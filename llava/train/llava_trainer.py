import os
import torch

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    ShardedDDPOption,
    logger,
)
from typing import List, Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        self._maybe_log_name_memory_loss_breakdown(model, loss, outputs)
        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def _unwrap_module(model):
        module = model
        seen = set()
        while hasattr(module, "module") and id(module) not in seen:
            seen.add(id(module))
            module = module.module
        return module

    @staticmethod
    def _scalar(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return float(value.detach().float().mean().item())
        return float(value)

    @staticmethod
    def _format_route_vector(route_summary, prefix, n_items):
        values = [route_summary.get(f"{prefix}_{idx}") for idx in range(n_items)]
        if any(value is None for value in values):
            return "[]"
        return "[" + ",".join(f"{value:.4f}" for value in values) + "]"

    def _maybe_log_name_memory_loss_breakdown(self, model, loss, outputs):
        if str(os.environ.get("NAME_MEMORY_DEBUG_LOSS", "")).strip().lower() not in {"1", "true", "yes", "on"}:
            return
        if not self.is_world_process_zero():
            return
        loss_dict = getattr(outputs, "name_memory_losses", None)
        if loss_dict is None and isinstance(outputs, dict):
            loss_dict = outputs.get("name_memory_losses")
        if not loss_dict or "loss_vqa" not in loss_dict:
            return

        every_step = str(os.environ.get("NAME_MEMORY_DEBUG_ROUTER_EVERY_STEP", "")).strip().lower() in {"1", "true", "yes", "on"}
        compact_router = str(os.environ.get("NAME_MEMORY_DEBUG_ROUTER_COMPACT", "")).strip().lower() in {"1", "true", "yes", "on"}
        logging_steps = 1 if every_step else max(1, int(getattr(self.args, "logging_steps", 1) or 1))
        step = int(getattr(self.state, "global_step", 0))
        if step % logging_steps != 0:
            return
        micro_step = int(getattr(self, "_name_memory_loss_micro_step", 0))
        self._name_memory_loss_micro_step = micro_step + 1

        module = self._unwrap_module(model)
        route_summary = {}
        adapter_stack = getattr(module, "module2_adapter_stack", None)
        if adapter_stack is not None and hasattr(adapter_stack, "route_debug_summary"):
            route_summary = adapter_stack.route_debug_summary()

        if compact_router:
            pref_route_count = int(getattr(adapter_stack, "num_shared_pref_experts", 5)) if adapter_stack is not None else 5
            pref_values = [
                float(route_summary.get(f"pref_factor_{idx}", 0.0))
                for idx in range(pref_route_count)
            ]
            if pref_route_count < 5:
                pref_values.extend([0.0] * (5 - pref_route_count))
            print(
                "[router7] "
                f"global_step={step} "
                f"task_pref={route_summary.get('task_pref', 0.0):.4f} "
                f"task_profile={route_summary.get('task_profile', 0.0):.4f} "
                f"pref_e0={pref_values[0]:.4f} "
                f"pref_e1={pref_values[1]:.4f} "
                f"pref_e2={pref_values[2]:.4f} "
                f"pref_e3={pref_values[3]:.4f} "
                f"pref_e4={pref_values[4]:.4f}",
                flush=True,
            )
            return

        aux_scale = float(module._current_aux_scale()) if hasattr(module, "_current_aux_scale") else 1.0
        weights = {
            "loss_consistency_img_desc": float(getattr(module, "loss_weight_consistency", 0.0)),
            "loss_contrast_profile_img": float(getattr(module, "loss_weight_profile_img", 0.0)),
            "loss_contrast_description": float(getattr(module, "loss_weight_description", 0.0)),
            "loss_pref_factor_contrast": float(getattr(module, "loss_weight_pref_factor", 0.0)),
            "loss_pref_decorrelation": float(getattr(module, "loss_weight_pref_decorrelation", 0.0)),
            "loss_pref_focal": float(getattr(module, "loss_weight_pref_focal", 0.0)),
            "loss_task_router": float(getattr(module, "loss_weight_task_router", 0.0)),
            "loss_pref_router": float(getattr(module, "loss_weight_pref_router", 0.0)),
        }

        total_value = self._scalar(loss)
        vqa_value = self._scalar(loss_dict.get("loss_vqa"))
        if total_value is None or vqa_value is None:
            return
        aux_value = total_value - vqa_value
        denom = total_value if abs(total_value) > 1e-8 else 1e-8

        logs = {
            "nm/loss_total": total_value,
            "nm/loss_vqa": vqa_value,
            "nm/loss_aux_weighted": aux_value,
            "nm/vqa_frac": vqa_value / denom,
            "nm/aux_frac": aux_value / denom,
            "nm/aux_scale": aux_scale,
        }
        for loss_name, weight in weights.items():
            raw_value = self._scalar(loss_dict.get(loss_name))
            if raw_value is None:
                continue
            short_name = loss_name.replace("loss_", "")
            logs[f"nm/{short_name}"] = raw_value
            logs[f"nm/{short_name}_weighted"] = raw_value * weight * aux_scale

        for key, value in route_summary.items():
            logs[f"nm/module2_{key}"] = value

        if getattr(self, "_last_name_memory_loss_trainer_log_step", None) != step:
            self._last_name_memory_loss_trainer_log_step = step
            self.log(logs)

        pref_route_count = int(getattr(adapter_stack, "num_shared_pref_experts", 5)) if adapter_stack is not None else 5
        pref_route = self._format_route_vector(route_summary, "pref_factor", pref_route_count)
        print(
            "[name-memory-loss] "
            f"global_step={step} micro_step={micro_step} total={total_value:.6f} vqa={vqa_value:.6f} "
            f"aux_weighted={aux_value:.6f} vqa_frac={logs['nm/vqa_frac']:.4f} "
            f"aux_frac={logs['nm/aux_frac']:.4f} aux_scale={aux_scale:.4f} "
            f"cons_w={logs.get('nm/consistency_img_desc_weighted', 0.0):.6f} "
            f"img_w={logs.get('nm/contrast_profile_img_weighted', 0.0):.6f} "
            f"desc_w={logs.get('nm/contrast_description_weighted', 0.0):.6f} "
            f"pref_factor_w={logs.get('nm/pref_factor_contrast_weighted', 0.0):.6f} "
            f"pref_dec_w={logs.get('nm/pref_decorrelation_weighted', 0.0):.6f} "
            f"pref_focal_w={logs.get('nm/pref_focal_weighted', 0.0):.6f} "
            f"task_router_w={logs.get('nm/task_router_weighted', 0.0):.6f} "
            f"pref_router_w={logs.get('nm/pref_router_weighted', 0.0):.6f} "
            f"m2_task_pref={route_summary.get('task_pref', 0.0):.4f} "
            f"m2_task_profile={route_summary.get('task_profile', 0.0):.4f} "
            f"m2_task_raw_pref={route_summary.get('task_raw_pref', 0.0):.4f} "
            f"m2_task_raw_profile={route_summary.get('task_raw_profile', 0.0):.4f} "
            f"m2_task_acc={route_summary.get('task_router_acc', 0.0):.4f} "
            f"m2_task_entropy={route_summary.get('task_router_entropy', 0.0):.4f} "
            f"m2_pref_acc={route_summary.get('pref_router_acc', 0.0):.4f} "
            f"m2_pref_entropy={route_summary.get('pref_router_entropy', 0.0):.4f} "
            f"m2_profile_img={route_summary.get('profile_image', 0.0):.4f} "
            f"m2_profile_desc={route_summary.get('profile_description', 0.0):.4f} "
            f"m2_pref_route={pref_route} "
            f"m2_site_calls={route_summary.get('site_calls', 0.0):.0f} "
            f"m2_by_concept={route_summary.get('by_concept', '')}",
            flush=True,
        )

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            return super().create_optimizer()

        opt_model = self.model

        # for name, param in opt_model.named_parameters():
        #     for k in ['lora_router.default.weight']:
        #         if str(k) in name:
        #             param.requires_grad = False

        # txt_file_path = "params.txt"
        # for name, param in opt_model.named_parameters():
        #     for k in ['loraA.0.', 'loraB.0.', 'loraA.2.', 'loraB.2.', 'lora_router.default.weight']:
        #         if str(k) in name:
        #             with open(txt_file_path, "a") as txt_file:
        #                 txt_file.write(f"{param}")
        
        # for name, p in opt_model.named_parameters():
        #     print(name, p.requires_grad)

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if self._is_lightweight_name_memory_checkpoint():
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            module = self._unwrap_module(self.model)
            base_model = getattr(module, "base_model", module)
            keys_to_match = ["mm_projector", "vision_resampler"]
            mm_projector_state = get_mm_adapter_state_maybe_zero_3(base_model.named_parameters(), keys_to_match)
            name_memory_state = module.get_name_memory_state_dict()

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                os.makedirs(output_dir, exist_ok=True)
                base_model.config.save_pretrained(output_dir)
                torch.save(mm_projector_state, os.path.join(output_dir, "mm_projector.bin"))
                torch.save(name_memory_state, os.path.join(output_dir, "name_memory_trainables.bin"))
                self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
        elif getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _is_lightweight_name_memory_checkpoint(self):
        if bool(getattr(self.args, "lora_enable", False)):
            return False
        module = self._unwrap_module(self.model)
        config = getattr(module, "config", None)
        return bool(getattr(config, "use_name_memory", False)) and hasattr(module, "get_name_memory_state_dict")

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

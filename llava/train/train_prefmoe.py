# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
import math
from dataclasses import dataclass, field
import json, deepspeed
import logging
import pathlib, random
from typing import Dict, Optional, Sequence, List

import torch
import sys
import transformers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PrefMoE.peft import (
    PeftModel,
    TaskType,
    LoraConfig,
    get_peft_model,
    PrefMoEMOELoraConfig,
    WEIGHTS_NAME,
    set_peft_model_state_dict,
)

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from llava.train.dataset import (
        DataArguments,
        make_supervised_prefmoe_data_module,
        make_supervised_data_module,
        rank0_print,
    )
except ImportError:
    from dataset import make_supervised_data_module, DataArguments, rank0_print, make_supervised_prefmoe_data_module

NAME_MEMORY_TRAIN_MODES = {
    "name_memory",
    "name_memory_hmoe",
    "name_memory_static_prefix_projector",
}


def maybe_force_non_reentrant_checkpointing(enabled: bool) -> bool:
    if not enabled:
        return False
    import torch.utils.checkpoint as torch_checkpoint

    current_checkpoint = torch_checkpoint.checkpoint
    if getattr(current_checkpoint, "_llava_non_reentrant_default", False):
        return False

    def checkpoint_with_non_reentrant_default(function, *args, **kwargs):
        kwargs.setdefault("use_reentrant", False)
        return current_checkpoint(function, *args, **kwargs)

    checkpoint_with_non_reentrant_default._llava_non_reentrant_default = True
    checkpoint_with_non_reentrant_default._llava_original_checkpoint = current_checkpoint
    torch_checkpoint.checkpoint = checkpoint_with_non_reentrant_default
    return True

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    previous_task_model_path: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")

    task_embedding_dim: Optional[int] = field(default=64)
    expert_num: Optional[int] = field(default=4)
    task: Optional[str] = field(default="")
    train_mode: Optional[str] = field(default="prefmllm")
    name_memory_top_layers: int = field(default=8)
    name_memory_factor_margin: float = field(default=0.2)
    name_memory_num_pseudo_users: int = field(default=50)
    name_memory_pseudo_csv_path: str = field(default="data/mmpb_clean/pseudo_users.csv")
    name_memory_mem_use_weight: float = field(default=0.10)
    name_memory_rank_margin_pseudo: float = field(default=0.2)
    name_memory_rank_margin_null: float = field(default=0.2)
    name_memory_shared_pref_lora_rank: int = field(default=64)
    name_memory_user_pref_rank: int = field(default=64)
    name_memory_user_profile_rank: int = field(default=32)
    name_memory_shared_pref_expert_count: int = field(default=5)
    name_memory_profile_expert_count: int = field(default=2)
    name_memory_module2_pref_profile_mix: float = field(default=0.7)
    name_memory_module2_arch: str = field(default="legacy_embedded")
    name_memory_use_backbone_lora: bool = field(default=True)
    name_memory_hmoe_arch_override: str = field(default="")
    name_memory_hmoe_keep_backbone_lora: bool = field(default=False)
    name_memory_module1_mode: str = field(default="token_bank")
    name_memory_token_bank_path: str = field(default="")
    name_memory_token_delta_scale: float = field(default=0.1)
    name_memory_build_bank_only: bool = field(default=False)
    name_memory_require_cached_bank: bool = field(default=True)
    name_memory_enable_prefix: bool = field(default=True)
    name_memory_enable_module2: bool = field(default=True)
    name_memory_enable_counterfactual: bool = field(default=True)
    name_memory_routing_mode: str = field(default="hierarchical")
    name_memory_task_router_mode: str = field(default="query")
    name_memory_task_router_fixed_pref_weight: float = field(default=0.60)
    name_memory_task_router_target_confidence: float = field(default=0.80)
    name_memory_pref_router_mode: str = field(default="learned")
    name_memory_pref_router_fixed_weights: str = field(default="")
    name_memory_pref_router_supervision: str = field(default="none")
    name_memory_pref_router_loss_weight: float = field(default=0.0)
    name_memory_pref_router_target_confidence: float = field(default=0.90)
    name_memory_pref_context_mode: str = field(default="all_factors")
    name_memory_profile_router_mode: str = field(default="learned")
    name_memory_profile_router_fixed_weights: str = field(default="")
    name_memory_hier_lora_context_mode: str = field(default="none")
    name_memory_hier_lora_target_modules: str = field(default="")
    name_memory_enable_user_pref_adapter: bool = field(default=True)
    name_memory_use_profile_image: bool = field(default=True)
    name_memory_use_description: bool = field(default=True)
    name_memory_use_preference: bool = field(default=True)
    name_memory_use_factorized_preference_memory: bool = field(default=True)
    name_memory_prefix_compose_mode: str = field(default="split13")
    name_memory_loss_weight_consistency: float = field(default=0.05)
    name_memory_loss_weight_profile_img: float = field(default=0.10)
    name_memory_loss_weight_description: float = field(default=0.10)
    name_memory_loss_weight_pref_factor: float = field(default=0.20)
    name_memory_loss_weight_pref_decorrelation: float = field(default=0.05)
    name_memory_loss_weight_pref_focal: float = field(default=0.20)
    name_memory_task_router_supervision: str = field(default="none")
    name_memory_task_router_loss_weight: float = field(default=0.0)
    name_memory_contrastive_temperature: float = field(default=0.07)
    name_memory_contrastive_real_weight: float = field(default=1.0)
    name_memory_contrastive_pseudo_weight: float = field(default=0.5)
    name_memory_focal_gamma: float = field(default=2.0)
    name_memory_pref_residual_loss_mode: str = field(default="density_focal")
    name_memory_pref_density_eta: float = field(default=1.0)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_full_state_maybe_zero_3(named_params):
    return {
        key: maybe_zero_3(value, ignore_status=True).cpu()
        for key, value in named_params
    }


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def write_trainable_sanity_summary(model, output_dir: str, training_args):
    if not (training_args.local_rank == 0 or training_args.local_rank == -1):
        return
    os.makedirs(output_dir, exist_ok=True)
    lines = []
    total_params = 0
    trainable_params = 0
    groups = {
        "base_model": [0, 0],
        "name_memory_module": [0, 0],
        "module2_adapter_stack": [0, 0],
        "mm_projector": [0, 0],
        "peft_lora": [0, 0],
        "other": [0, 0],
    }
    trainable_names = []
    unexpected_base_trainable_names = []
    for name, param in model.named_parameters():
        count = int(param.numel())
        total_params += count
        if name.startswith("module2_adapter_stack."):
            group = "module2_adapter_stack"
        elif name.startswith("name_memory_module."):
            group = "name_memory_module"
        elif "mm_projector" in name:
            group = "mm_projector"
        elif "lora_" in name:
            group = "peft_lora"
        elif name.startswith("base_model."):
            group = "base_model"
        else:
            group = "other"
        groups[group][0] += count
        if param.requires_grad:
            trainable_params += count
            groups[group][1] += count
            trainable_names.append((name, count, str(param.dtype), tuple(param.shape)))
            if name.startswith("base_model.") and "mm_projector" not in name and "lora_" not in name:
                unexpected_base_trainable_names.append(name)

    lines.append("[name-memory sanity]")
    lines.append(f"module2_arch={getattr(model.config, 'name_memory_module2_arch', 'n/a')}")
    lines.append(f"use_name_memory={getattr(model.config, 'use_name_memory', False)}")
    lines.append(f"name_memory_version={getattr(model.config, 'name_memory_version', 'n/a')}")
    lines.append(f"prefix_compose_mode={getattr(model.config, 'name_memory_prefix_compose_mode', 'n/a')}")
    lines.append(f"pref_residual_loss_mode={getattr(model.config, 'name_memory_pref_residual_loss_mode', 'density_focal')}")
    lines.append(f"pref_density_eta={getattr(model.config, 'name_memory_pref_density_eta', 'n/a')}")
    lines.append(f"use_backbone_lora={getattr(model.config, 'name_memory_use_backbone_lora', 'n/a')}")
    lines.append(f"total_params={total_params}")
    lines.append(f"trainable_params={trainable_params}")
    lines.append(f"trainable_ratio={trainable_params / max(1, total_params):.8f}")
    lines.append("")
    lines.append("[groups total/trainable]")
    for group, (group_total, group_trainable) in groups.items():
        lines.append(f"{group}: total={group_total} trainable={group_trainable}")
    lines.append("")
    lines.append(f"unexpected_trainable_base_params={len(unexpected_base_trainable_names)}")
    for name in unexpected_base_trainable_names[:100]:
        lines.append(f"  {name}")
    lines.append("")
    lines.append(f"trainable_parameter_names={len(trainable_names)}")
    for name, count, dtype, shape in trainable_names:
        lines.append(f"{name}\tparams={count}\tdtype={dtype}\tshape={shape}")

    text = "\n".join(lines) + "\n"
    path = os.path.join(output_dir, "sanity_check.txt")
    with open(path, "w", encoding="utf-8") as fout:
        fout.write(text)
    if str(os.environ.get("NAME_MEMORY_DEBUG_ROUTER_COMPACT", "")).strip().lower() not in {"1", "true", "yes", "on"}:
        print(text)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def load_model_from_previous_task(model, previous_task_model_path):
    token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
    # if model.lm_head.weight.shape[0] != token_num:
    #     model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
    #     model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

    print('Loading additional LLaVA weights...')
    if os.path.exists(os.path.join(previous_task_model_path, 'non_lora_trainables.bin')):
        non_lora_trainables = torch.load(os.path.join(previous_task_model_path, 'non_lora_trainables.bin'), map_location='cpu')
    else:
        # this is probably from HF Hub
        from huggingface_hub import hf_hub_download
        def load_from_hf(repo_id, filename, subfolder=None):
            cache_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder)
            return torch.load(cache_file, map_location='cpu')
        non_lora_trainables = load_from_hf(previous_task_model_path, 'non_lora_trainables.bin')
    non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
    if any(k.startswith('model.model.') for k in non_lora_trainables):
        non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
    model.load_state_dict(non_lora_trainables, strict=False)

    from peft import PeftModel
    print('Loading LoRA weights...')
    filename = os.path.join(previous_task_model_path, WEIGHTS_NAME)
    adapters_weights = torch.load(filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    load_result = set_peft_model_state_dict(model, adapters_weights, adapter_name="default")
    print('Model is loaded...')

def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    if maybe_force_non_reentrant_checkpointing(training_args.gradient_checkpointing):
        print("[gradient_checkpointing] forcing torch checkpoint default to use_reentrant=False")
    train_mode = str(getattr(model_args, "train_mode", "prefmoe")).strip().lower()
    if train_mode in {"finetune", "fine-tune"}:
        train_mode = "fine_tune"
    valid_train_modes = {
        "prefmllm",
        "prefmoe",
        "lora",
        "fine_tune",
        "llava",
        "name_memory",
        "name_memory_hmoe",
        "name_memory_static_prefix_projector",
    }
    if train_mode not in valid_train_modes:
        raise ValueError(f"Unsupported train_mode={train_mode}, valid={sorted(valid_train_modes)}")
    if train_mode == "prefmllm":
        train_mode = "name_memory_hmoe"

    if train_mode == "lora":
        if not training_args.lora_enable:
            rank0_print("[train_mode=lora] force lora_enable=True")
            training_args.lora_enable = True
    elif train_mode == "name_memory":
        if not training_args.lora_enable:
            rank0_print("[train_mode=name_memory] force lora_enable=True")
            training_args.lora_enable = True
        model_args.tune_mm_mlp_adapter = False
        model_args.name_memory_module2_arch = "legacy_embedded"
        model_args.name_memory_use_backbone_lora = True
    elif train_mode == "name_memory_hmoe":
        hmoe_arch_override = str(model_args.name_memory_hmoe_arch_override or "").strip().lower()
        keep_backbone_lora = bool(model_args.name_memory_hmoe_keep_backbone_lora)
        if hmoe_arch_override:
            rank0_print(f"[train_mode=name_memory_hmoe] use module2_arch override: {hmoe_arch_override}")
            model_args.name_memory_module2_arch = hmoe_arch_override
        else:
            model_args.name_memory_module2_arch = "hierarchical_moe_lora"
        if training_args.lora_enable and not keep_backbone_lora:
            rank0_print("[train_mode=name_memory_hmoe] use PrefMLLM hierarchical MoE adapters without backbone LoRA")
            training_args.lora_enable = False
        model_args.tune_mm_mlp_adapter = False
        model_args.name_memory_use_backbone_lora = bool(keep_backbone_lora and training_args.lora_enable)
        hier_context_mode = str(model_args.name_memory_hier_lora_context_mode or "none").strip().lower()
        if (
            model_args.name_memory_module2_arch == "hierarchical_moe_lora"
            and hier_context_mode in {"none", "off", "disabled", "false", "0"}
            and int(model_args.name_memory_shared_pref_lora_rank) == 64
        ):
            rank0_print("[train_mode=name_memory_hmoe] use Hierarchical MoE preference LoRA rank=8")
            model_args.name_memory_shared_pref_lora_rank = 8
        if (
            model_args.name_memory_module2_arch == "hierarchical_moe_lora"
            and hier_context_mode in {"none", "off", "disabled", "false", "0"}
            and int(model_args.name_memory_user_profile_rank) == 32
        ):
            rank0_print("[train_mode=name_memory_hmoe] use Hierarchical MoE profile LoRA rank=8")
            model_args.name_memory_user_profile_rank = 8
        module1_mode = str(model_args.name_memory_module1_mode or "").strip().lower()
        prefix_compose_mode = str(model_args.name_memory_prefix_compose_mode or "").strip().lower()
        if module1_mode in {"role_binder", "role_binder_v2", "v2_role_binder", "legacy_role_binder"}:
            model_args.name_memory_prefix_compose_mode = "role_binder_v2"
        elif prefix_compose_mode in {"role_binder", "role_binder_v2", "v2_role_binder", "binder8", "legacy_binder8"}:
            model_args.name_memory_module1_mode = "role_binder_v2"
            model_args.name_memory_prefix_compose_mode = "role_binder_v2"
        data_args.name_memory_use_concept_id = True
    elif train_mode == "name_memory_static_prefix_projector":
        if training_args.lora_enable:
            rank0_print("[train_mode=name_memory_static_prefix_projector] force lora_enable=False")
            training_args.lora_enable = False
        model_args.tune_mm_mlp_adapter = True
        model_args.name_memory_module2_arch = "legacy_embedded"
        model_args.name_memory_use_backbone_lora = False
        model_args.name_memory_enable_prefix = True
        model_args.name_memory_enable_module2 = False
        model_args.name_memory_enable_counterfactual = False
        model_args.name_memory_task_router_supervision = "none"
        model_args.name_memory_task_router_loss_weight = 0.0
        model_args.name_memory_pref_router_supervision = "none"
        model_args.name_memory_pref_router_loss_weight = 0.0
        model_args.name_memory_loss_weight_consistency = 0.0
        model_args.name_memory_loss_weight_profile_img = 0.0
        model_args.name_memory_loss_weight_description = 0.0
        model_args.name_memory_loss_weight_pref_factor = 0.0
        model_args.name_memory_loss_weight_pref_decorrelation = 0.0
        model_args.name_memory_loss_weight_pref_focal = 0.0
        data_args.name_memory_use_concept_id = True
    elif train_mode == "fine_tune":
        if training_args.lora_enable:
            rank0_print("[train_mode=fine_tune] force lora_enable=False")
            training_args.lora_enable = False
        model_args.tune_mm_mlp_adapter = False
    elif train_mode == "llava":
        if training_args.lora_enable:
            rank0_print("[train_mode=llava] force lora_enable=False")
            training_args.lora_enable = False
        model_args.tune_mm_mlp_adapter = True

    rank0_print(
        f"[train_mode={train_mode}] lora_enable={training_args.lora_enable}, "
        f"tune_mm_mlp_adapter={model_args.tune_mm_mlp_adapter}"
    )
    if train_mode == "name_memory_hmoe" and "hmoe" not in os.path.basename(str(training_args.output_dir or "")).lower():
        rank0_print(
            f"[train_mode=name_memory_hmoe] output_dir='{training_args.output_dir}' does not contain 'hmoe'; "
            "please keep PrefMLLM HMoE outputs in a dedicated directory."
        )
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args,
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        if train_mode in {"prefmoe", "name_memory", "name_memory_hmoe"}:
            kwargs = {
                "task_embedding_dim": model_args.task_embedding_dim,
                "expert_num": model_args.expert_num,
            }
            lora_config = PrefMoEMOELoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(model),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type=TaskType.CAUSAL_LM_PrefMoE,
                **kwargs
            )
        elif train_mode == "lora":
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(model),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type=TaskType.CAUSAL_LM,
            )
        else:
            raise ValueError(f"lora_enable=True is incompatible with train_mode={train_mode}")
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print(f"Adding LoRA adapters... mode={train_mode}")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    #######################################################################################
    # if model_args.previous_task_model_path is not None:
    #     # load model from previous task
    #     load_model_from_previous_task(model, model_args.previous_task_model_path)

    ########################################################################################
    # data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    data_args.name_memory_enable = bool(train_mode in NAME_MEMORY_TRAIN_MODES)
    data_module = make_supervised_prefmoe_data_module(tokenizer=tokenizer, data_args=data_args)
    if train_mode in NAME_MEMORY_TRAIN_MODES:
        model = LlavaLlamaNameMemoryWrapper(
            model,
            shared_pref_expert_count=model_args.name_memory_shared_pref_expert_count,
            top_layers=model_args.name_memory_top_layers,
            factor_margin=model_args.name_memory_factor_margin,
            num_pseudo_users=model_args.name_memory_num_pseudo_users,
            pseudo_csv_path=model_args.name_memory_pseudo_csv_path,
            shared_pref_rank=model_args.name_memory_shared_pref_lora_rank,
            user_pref_rank=model_args.name_memory_user_pref_rank,
            user_profile_rank=model_args.name_memory_user_profile_rank,
            profile_expert_count=model_args.name_memory_profile_expert_count,
            module2_pref_profile_mix=model_args.name_memory_module2_pref_profile_mix,
            module2_arch=model_args.name_memory_module2_arch,
            use_backbone_lora=model_args.name_memory_use_backbone_lora,
            module1_mode=model_args.name_memory_module1_mode,
            token_bank_path=model_args.name_memory_token_bank_path,
            token_delta_scale=model_args.name_memory_token_delta_scale,
            enable_prefix=model_args.name_memory_enable_prefix,
            enable_module2=model_args.name_memory_enable_module2,
            enable_counterfactual=model_args.name_memory_enable_counterfactual,
            routing_mode=model_args.name_memory_routing_mode,
            task_router_mode=model_args.name_memory_task_router_mode,
            task_router_fixed_pref_weight=model_args.name_memory_task_router_fixed_pref_weight,
            task_router_target_confidence=model_args.name_memory_task_router_target_confidence,
            pref_router_mode=model_args.name_memory_pref_router_mode,
            pref_router_fixed_weights=model_args.name_memory_pref_router_fixed_weights,
            pref_router_supervision=model_args.name_memory_pref_router_supervision,
            pref_router_loss_weight=model_args.name_memory_pref_router_loss_weight,
            pref_router_target_confidence=model_args.name_memory_pref_router_target_confidence,
            pref_context_mode=model_args.name_memory_pref_context_mode,
            profile_router_mode=model_args.name_memory_profile_router_mode,
            profile_router_fixed_weights=model_args.name_memory_profile_router_fixed_weights,
            hier_lora_context_mode=model_args.name_memory_hier_lora_context_mode,
            hier_lora_target_modules=model_args.name_memory_hier_lora_target_modules,
            enable_user_pref_adapter=model_args.name_memory_enable_user_pref_adapter,
            use_profile_image=model_args.name_memory_use_profile_image,
            use_description=model_args.name_memory_use_description,
            use_preference=model_args.name_memory_use_preference,
            use_factorized_preference_memory=model_args.name_memory_use_factorized_preference_memory,
            prefix_compose_mode=model_args.name_memory_prefix_compose_mode,
            loss_weight_consistency=model_args.name_memory_loss_weight_consistency,
            loss_weight_profile_img=model_args.name_memory_loss_weight_profile_img,
            loss_weight_description=model_args.name_memory_loss_weight_description,
            loss_weight_pref_factor=model_args.name_memory_loss_weight_pref_factor,
            loss_weight_pref_decorrelation=model_args.name_memory_loss_weight_pref_decorrelation,
            loss_weight_pref_focal=model_args.name_memory_loss_weight_pref_focal,
            task_router_supervision=model_args.name_memory_task_router_supervision,
            task_router_loss_weight=model_args.name_memory_task_router_loss_weight,
            contrastive_temperature=model_args.name_memory_contrastive_temperature,
            contrastive_real_weight=model_args.name_memory_contrastive_real_weight,
            contrastive_pseudo_weight=model_args.name_memory_contrastive_pseudo_weight,
            focal_gamma=model_args.name_memory_focal_gamma,
            pref_residual_loss_mode=model_args.name_memory_pref_residual_loss_mode,
            pref_density_eta=model_args.name_memory_pref_density_eta,
        )
        if bool(model_args.name_memory_build_bank_only) and torch.cuda.is_available():
            bank_dtype = compute_dtype
            rank0_print(
                f"[name_memory] build_bank_only=True: move wrapped model to {training_args.device} "
                f"dtype={bank_dtype} before token-bank encoding"
            )
            model.to(device=training_args.device, dtype=bank_dtype)
        if train_mode == "name_memory_hmoe":
            if model_args.name_memory_use_backbone_lora and training_args.lora_enable:
                rank0_print(
                    "[train_mode=name_memory_hmoe] keep base_model trainable via backbone LoRA "
                    "(skip blanket base_model.requires_grad_(False))"
                )
            else:
                model.base_model.requires_grad_(False)
                model.name_memory_module.requires_grad_(True)
                if bool(getattr(model, "enable_module2", True)):
                    model.module2_adapter_stack.requires_grad_(True)
                else:
                    model.module2_adapter_stack.requires_grad_(False)
        elif train_mode == "name_memory_static_prefix_projector":
            model.base_model.requires_grad_(False)
            model.name_memory_module.requires_grad_(False)
            model.module2_adapter_stack.requires_grad_(False)
            for p in model.base_model.get_model().mm_projector.parameters():
                p.requires_grad = True
        write_trainable_sanity_summary(model, training_args.output_dir, training_args)
    ########################################################################################
    n_folds = len(data_module["train_dataset"].data_split)
    fold_start = max(0, int(getattr(data_args, "train_fold_start", 0)))
    fold_end = int(getattr(data_args, "train_fold_end", -1))
    if fold_end < 0 or fold_end >= n_folds:
        fold_end = n_folds - 1
    if fold_start > fold_end:
        raise ValueError(f"Invalid fold range: start={fold_start}, end={fold_end}, n_folds={n_folds}")

    for i in range(fold_start, fold_end + 1):
        n_tasks = len(data_module["train_dataset"].data_split[i]["tasks"])
        task_start = max(0, int(getattr(data_args, "train_task_start", 0)))
        task_end = int(getattr(data_args, "train_task_end", -1))
        if task_end < 0 or task_end >= n_tasks:
            task_end = n_tasks - 1
        if task_start > task_end:
            raise ValueError(
                f"Invalid task range: start={task_start}, end={task_end}, "
                f"n_tasks={n_tasks}, fold={i}"
            )

        for j in range(task_start, task_end + 1):
            task_meta = data_module["train_dataset"].data_split[i]["tasks"][j]
            if "concepts" in task_meta:
                print(f"Training on fold {i}, task {j} with concepts: {task_meta['concepts']}")
            else:
                train_count = len(task_meta.get("train_idx", []))
                print(f"Training on fold {i}, task {j} with train samples: {train_count}")
            data_module["train_dataset"].fold_idx = i
            data_module["train_dataset"].task_idx = j  
            if train_mode in {"name_memory", "name_memory_hmoe"}:
                registry = data_module["train_dataset"].get_name_memory_registry(i)
                model.initialize_name_memory_from_registry(
                    registry=registry,
                    tokenizer=tokenizer,
                    image_aspect_ratio=data_args.image_aspect_ratio,
                    build_if_missing=not bool(model_args.name_memory_require_cached_bank),
                )
                if bool(model_args.name_memory_build_bank_only):
                    rank0_print(
                        f"[name_memory] token bank ready for fold {i}: "
                        f"{model.name_memory_module._resolve_token_bank_path(registry, model.name_memory_module._current_manifest_hash)}"
                    )
                    continue
                batch_size = max(1, int(training_args.per_device_train_batch_size))
                grad_acc = max(1, int(training_args.gradient_accumulation_steps))
                num_batches = max(1, math.ceil(len(data_module["train_dataset"]) / float(batch_size)))
                num_updates = max(1, math.ceil(num_batches / float(grad_acc)) * max(1, int(training_args.num_train_epochs)))
                model.set_loss_schedule(int(0.1 * num_updates))
            trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
            # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            #     trainer.train(resume_from_checkpoint=True)
            # else: 
            trainer.train()
            trainer.save_state()

            #############################################################################################
            model.config.use_cache = True

            current_save_dir = os.path.join(training_args.output_dir, f"fold_{i}_task_{j}")
            os.makedirs(current_save_dir, exist_ok=True)

            if training_args.lora_enable:
                if train_mode in NAME_MEMORY_TRAIN_MODES:
                    state_dict = get_peft_state_maybe_zero_3(model.base_model.named_parameters(), training_args.lora_bias)
                    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.base_model.named_parameters())
                    name_memory_state_dict = model.get_name_memory_state_dict()
                else:
                    state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
                    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
                
                if training_args.local_rank == 0 or training_args.local_rank == -1:
                    model.config.save_pretrained(current_save_dir)
                    model.save_pretrained(current_save_dir, state_dict=state_dict) # save lora parameters
                    torch.save(non_lora_state_dict, os.path.join(current_save_dir, 'non_lora_trainables.bin'))
                    if train_mode in NAME_MEMORY_TRAIN_MODES:
                        torch.save(name_memory_state_dict, os.path.join(current_save_dir, 'name_memory_trainables.bin'))
                    print(f"Model for fold {i} task {j} saved to {current_save_dir}")
            else:
                if train_mode in NAME_MEMORY_TRAIN_MODES:
                    name_memory_state_dict = model.get_name_memory_state_dict()
                    mm_projector_state_dict = get_mm_adapter_state_maybe_zero_3(
                        model.base_model.named_parameters(),
                        ["mm_projector", "vision_resampler"],
                    )
                    if training_args.local_rank == 0 or training_args.local_rank == -1:
                        model.base_model.config.save_pretrained(current_save_dir)
                        torch.save(mm_projector_state_dict, os.path.join(current_save_dir, 'mm_projector.bin'))
                        torch.save(name_memory_state_dict, os.path.join(current_save_dir, 'name_memory_trainables.bin'))
                        print(f"Model for fold {i} task {j} saved to {current_save_dir}")
                else:
                    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=current_save_dir)
            model.config.use_cache = False
    ################################################################################################

if __name__ == "__main__":
    train()

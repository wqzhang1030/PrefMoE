from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.constants import IGNORE_INDEX
from llava.model.memory import (
    HMoEModule2AdapterStack,
    HierarchicalMoELoraAdapterStack,
    NAME_MEMORY_BINDER_BOTTLENECK,
    NAME_MEMORY_DEFAULT_PSEUDO_CSV,
    NAME_MEMORY_FACTORS,
    NAME_MEMORY_UNKNOWN_SLOT_ID,
    Module2AdapterStack,
    NameMemoryModule,
)


def _normalize_pref_residual_loss_mode(mode: str) -> str:
    normalized = str(mode or "density_focal").strip().lower().replace("-", "_")
    if normalized in {"1", "paper", "eq6", "density", "density_focal", "annotation_density_focal", "residual_density_focal"}:
        return "density_focal"
    if normalized in {"0", "prototype"}:
        return "prototype"
    raise ValueError(f"Unsupported preference residual loss mode: {mode!r}")


#送module进去的入口
class LlavaLlamaNameMemoryWrapper(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        num_slots: int = 1,
        num_factors: int = len(NAME_MEMORY_FACTORS),
        shared_pref_expert_count: int = len(NAME_MEMORY_FACTORS),
        top_layers: int = 8,
        factor_margin: float = 0.2,
        num_pseudo_users: int = 50,
        pseudo_csv_path: str = NAME_MEMORY_DEFAULT_PSEUDO_CSV,
        shared_pref_rank: int = 64,
        user_pref_rank: int = 64,
        user_profile_rank: int = 32,
        profile_expert_count: int = 2,
        module2_pref_profile_mix: float = 0.7,
        module2_arch: str = "legacy_embedded",
        use_backbone_lora: bool = True,
        module1_mode: str = "token_bank",
        token_bank_path: str = "",
        token_delta_scale: float = 0.1,
        enable_prefix: bool = True,
        enable_module2: bool = True,
        enable_counterfactual: bool = True,
        routing_mode: str = "hierarchical",
        task_router_mode: str = "memory_only",
        task_router_fixed_pref_weight: float = 0.6,
        task_router_target_confidence: float = 1.0,
        pref_router_mode: str = "learned",
        pref_router_fixed_weights: str = "",
        pref_router_supervision: str = "none",
        pref_router_loss_weight: float = 0.0,
        pref_router_target_confidence: float = 0.9,
        pref_context_mode: str = "all_factors",
        profile_router_mode: str = "learned",
        profile_router_fixed_weights: str = "",
        hier_lora_context_mode: str = "none",
        hier_lora_target_modules: str = "",
        enable_user_pref_adapter: bool = True,
        use_profile_image: bool = True,
        use_description: bool = True,
        use_preference: bool = True,
        use_factorized_preference_memory: bool = True,
        prefix_compose_mode: str = "split13",
        loss_weight_consistency: float = 0.05,
        loss_weight_profile_img: float = 0.10,
        loss_weight_description: float = 0.10,
        loss_weight_pref_factor: float = 0.20,
        loss_weight_pref_decorrelation: float = 0.05,
        loss_weight_pref_focal: float = 0.20,
        task_router_supervision: str = "none",
        task_router_loss_weight: float = 0.0,
        contrastive_temperature: float = 0.07,
        contrastive_real_weight: float = 1.0,
        contrastive_pseudo_weight: float = 0.5,
        focal_gamma: float = 2.0,
        pref_residual_loss_mode: str = "density_focal",
        pref_density_eta: float = 1.0,
    ):
        super().__init__()
        self.base_model = base_model
        hidden_size = int(self.base_model.config.hidden_size)
         ##module1的初始化
        self.name_memory_module = NameMemoryModule(
            hidden_size=hidden_size,
            num_factors=num_factors,
            num_pseudo_users=num_pseudo_users,
            pseudo_csv_path=pseudo_csv_path,
            token_bank_path=token_bank_path,
            token_delta_scale=token_delta_scale,
            module1_mode=module1_mode,
            use_profile_image=use_profile_image,
            use_description=use_description,
            use_preference=use_preference,
            use_factorized_preference_memory=use_factorized_preference_memory,
            prefix_compose_mode=prefix_compose_mode,
        )
        self.module2_arch = str(module2_arch or "legacy_embedded").strip().lower()
        self._uses_hierarchical_lora_moe = self.module2_arch in {
            "hierarchical_moe_lora",
            "hier_moe_lora",
            "linear_hmoe_lora",
        }
        self.use_backbone_lora = bool(use_backbone_lora)
        self.task_router_supervision = str(task_router_supervision or "none").strip().lower()
        self.loss_weight_task_router = float(task_router_loss_weight)
        self.pref_router_supervision = str(pref_router_supervision or "none").strip().lower()
        self.loss_weight_pref_router = float(pref_router_loss_weight)
        self.pref_router_target_confidence = float(pref_router_target_confidence)
        self.pref_context_mode = str(pref_context_mode or "all_factors").strip().lower()
        ##module2的初始化
        if self._uses_hierarchical_lora_moe:
            self.module2_adapter_stack = HierarchicalMoELoraAdapterStack(
                hidden_size=hidden_size,
                num_layers=top_layers,
                num_factors=num_factors,
                num_shared_pref_experts=shared_pref_expert_count,
                num_profile_experts=profile_expert_count,
                shared_pref_rank=shared_pref_rank,
                user_pref_rank=user_pref_rank,
                user_profile_rank=user_profile_rank,
                enable_user_pref_adapter=enable_user_pref_adapter,
                task_router_supervision=self.task_router_supervision,
                task_router_mode=task_router_mode,
                task_router_fixed_pref_weight=task_router_fixed_pref_weight,
                task_router_target_confidence=task_router_target_confidence,
                pref_router_mode=pref_router_mode,
                pref_router_fixed_weights=pref_router_fixed_weights,
                pref_router_supervision=self.pref_router_supervision,
                pref_router_target_confidence=self.pref_router_target_confidence,
                pref_context_mode=self.pref_context_mode,
                profile_router_mode=profile_router_mode,
                profile_router_fixed_weights=profile_router_fixed_weights,
                context_mode=hier_lora_context_mode,
                target_modules=hier_lora_target_modules,
            )
        elif self.module2_arch == "hmoe":
            self.module2_adapter_stack = HMoEModule2AdapterStack(
                hidden_size=hidden_size,
                num_layers=top_layers,
                num_factors=num_factors,
                num_shared_pref_experts=shared_pref_expert_count,
                num_profile_experts=profile_expert_count,
                shared_pref_rank=shared_pref_rank,
                user_profile_rank=user_profile_rank,
                routing_mode=routing_mode,
                task_router_supervision=self.task_router_supervision,
                task_router_mode=task_router_mode,
                task_router_fixed_pref_weight=task_router_fixed_pref_weight,
                task_router_target_confidence=task_router_target_confidence,
            )
        else:
            self.module2_adapter_stack = Module2AdapterStack(
                hidden_size=hidden_size,
                num_layers=top_layers,
                num_factors=num_factors,
                num_shared_pref_experts=shared_pref_expert_count,
                shared_pref_rank=shared_pref_rank,
                user_pref_rank=user_pref_rank,
                user_profile_rank=user_profile_rank,
                pref_profile_mix=module2_pref_profile_mix,
                routing_mode=routing_mode,
            )
        ##Module 2 挂到 decoder top layers 上
        self.module2_adapter_stack.attach(self._get_decoder_layers())
        self.register_buffer("name_memory_forward_steps", torch.zeros(1, dtype=torch.long), persistent=False)

        self.loss_warmup_steps = 0
        self.factor_margin = float(factor_margin)
        self.num_slots = int(num_slots)
        self.enable_prefix = bool(enable_prefix)
        self.enable_module2 = bool(enable_module2)
        if not self.enable_module2:
            self.module2_adapter_stack.requires_grad_(False)
        self.enable_counterfactual = bool(enable_counterfactual)
        self.prefix_compose_mode = self.name_memory_module.prefix_compose_mode
        self.task_router_mode = str(task_router_mode or "memory_only").strip().lower()
        self.task_router_fixed_pref_weight = float(task_router_fixed_pref_weight)
        self.task_router_target_confidence = float(task_router_target_confidence)
        self.pref_router_mode = str(pref_router_mode or "learned").strip().lower()
        self.pref_router_fixed_weights = str(pref_router_fixed_weights or "")
        self.pref_router_supervision = str(pref_router_supervision or "none").strip().lower()
        self.pref_router_target_confidence = float(pref_router_target_confidence)
        self.profile_router_mode = str(profile_router_mode or "learned").strip().lower()
        self.profile_router_fixed_weights = str(profile_router_fixed_weights or "")
        self.hier_lora_context_mode = str(hier_lora_context_mode or "none").strip().lower()
        self.hier_lora_target_modules = str(hier_lora_target_modules or "")
        self.enable_user_pref_adapter = bool(enable_user_pref_adapter)
        self.loss_weight_consistency = float(loss_weight_consistency)
        self.loss_weight_profile_img = float(loss_weight_profile_img)
        self.loss_weight_description = float(loss_weight_description)
        self.loss_weight_pref_factor = float(loss_weight_pref_factor)
        self.loss_weight_pref_decorrelation = float(loss_weight_pref_decorrelation)
        self.loss_weight_pref_focal = float(loss_weight_pref_focal)
        self.contrastive_temperature = float(contrastive_temperature)
        self.contrastive_real_weight = float(contrastive_real_weight)
        self.contrastive_pseudo_weight = float(contrastive_pseudo_weight)
        self.focal_gamma = float(focal_gamma)
        self.pref_residual_loss_mode = _normalize_pref_residual_loss_mode(pref_residual_loss_mode)
        self.pref_density_eta = float(pref_density_eta)

        self.config.use_name_memory = True
        if self._uses_hierarchical_lora_moe:
            self.config.name_memory_version = "hierarchical_moe_lora_v1"
        elif self.module2_arch == "hmoe":
            self.config.name_memory_version = "hmoe_v1"
        else:
            self.config.name_memory_version = "v2"
        self.config.name_memory_num_slots = int(num_slots)
        self.config.name_memory_num_factors = int(num_factors)
        self.config.name_memory_shared_pref_expert_count = int(shared_pref_expert_count)
        self.config.name_memory_top_layers = int(top_layers)
        self.config.name_memory_prefix_compose_mode = self.prefix_compose_mode
        if self.prefix_compose_mode in {"sum8", "role_binder_v2"}:
            self.config.name_memory_prefix_num_tokens = 3 + int(num_factors)
        else:
            self.config.name_memory_prefix_num_tokens = 3 + 2 * int(num_factors)
        self.config.name_memory_num_pseudo_users = int(num_pseudo_users)
        self.config.name_memory_pseudo_csv_path = str(pseudo_csv_path)
        self.config.name_memory_factor_margin = float(factor_margin)
        self.config.name_memory_module2_arch = self.module2_arch
        self.config.name_memory_shared_pref_lora_rank = int(shared_pref_rank)
        self.config.name_memory_user_pref_rank = int(user_pref_rank)
        self.config.name_memory_user_profile_rank = int(user_profile_rank)
        self.config.name_memory_profile_expert_count = int(profile_expert_count)
        self.config.name_memory_module2_pref_profile_mix = float(module2_pref_profile_mix)
        if self._uses_hierarchical_lora_moe:
            self.config.name_memory_router_mode = "hierarchical_lora_moe"
        elif self.module2_arch == "hmoe":
            self.config.name_memory_router_mode = "query_memory_conditioned_hmoe"
        else:
            self.config.name_memory_router_mode = "uid_dominant_memory_conditioned"
        self.config.name_memory_use_backbone_lora = bool(self.use_backbone_lora)
        self.config.name_memory_module1_mode = str(module1_mode)
        self.config.name_memory_token_bank_path = str(token_bank_path or "")
        self.config.name_memory_token_delta_scale = float(token_delta_scale)
        if self.prefix_compose_mode == "role_binder_v2":
            self.config.name_memory_binding_mode = "role_binder_v2"
            self.config.name_memory_binding_bottleneck = int(NAME_MEMORY_BINDER_BOTTLENECK)
        else:
            self.config.name_memory_binding_mode = (
                "shared_point_offset_sum8" if self.prefix_compose_mode == "sum8" else "shared_point_offset"
            )
            self.config.name_memory_binding_bottleneck = 0
        self.config.name_memory_enable_prefix = self.enable_prefix
        self.config.name_memory_enable_module2 = self.enable_module2
        self.config.name_memory_enable_counterfactual = self.enable_counterfactual
        self.config.name_memory_routing_mode = str(routing_mode or "hierarchical")
        self.config.name_memory_task_router_mode = self.task_router_mode
        self.config.name_memory_task_router_fixed_pref_weight = self.task_router_fixed_pref_weight
        self.config.name_memory_task_router_target_confidence = self.task_router_target_confidence
        self.config.name_memory_pref_router_mode = self.pref_router_mode
        self.config.name_memory_pref_router_fixed_weights = self.pref_router_fixed_weights
        self.config.name_memory_pref_router_supervision = self.pref_router_supervision
        self.config.name_memory_pref_router_target_confidence = self.pref_router_target_confidence
        self.config.name_memory_pref_context_mode = self.pref_context_mode
        self.config.name_memory_profile_router_mode = self.profile_router_mode
        self.config.name_memory_profile_router_fixed_weights = self.profile_router_fixed_weights
        self.config.name_memory_hier_lora_context_mode = self.hier_lora_context_mode
        self.config.name_memory_hier_lora_target_modules = self.hier_lora_target_modules
        self.config.name_memory_enable_user_pref_adapter = self.enable_user_pref_adapter
        self.config.name_memory_use_profile_image = bool(use_profile_image)
        self.config.name_memory_use_description = bool(use_description)
        self.config.name_memory_use_preference = bool(use_preference)
        self.config.name_memory_use_factorized_preference_memory = bool(use_factorized_preference_memory)
        self.config.name_memory_loss_weight_consistency = self.loss_weight_consistency
        self.config.name_memory_loss_weight_profile_img = self.loss_weight_profile_img
        self.config.name_memory_loss_weight_description = self.loss_weight_description
        self.config.name_memory_loss_weight_pref_factor = self.loss_weight_pref_factor
        self.config.name_memory_loss_weight_pref_decorrelation = self.loss_weight_pref_decorrelation
        self.config.name_memory_loss_weight_pref_focal = self.loss_weight_pref_focal
        self.config.name_memory_task_router_supervision = self.task_router_supervision
        self.config.name_memory_task_router_loss_weight = self.loss_weight_task_router
        self.config.name_memory_pref_router_loss_weight = self.loss_weight_pref_router
        self.config.name_memory_contrastive_temperature = self.contrastive_temperature
        self.config.name_memory_contrastive_real_weight = self.contrastive_real_weight
        self.config.name_memory_contrastive_pseudo_weight = self.contrastive_pseudo_weight
        self.config.name_memory_focal_gamma = self.focal_gamma
        self.config.name_memory_pref_residual_loss_mode = self.pref_residual_loss_mode
        self.config.name_memory_pref_density_eta = self.pref_density_eta

    @property
    def config(self):
        return self.base_model.config

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def save_pretrained(self, output_dir: str, state_dict=None):
        return self.base_model.save_pretrained(output_dir, state_dict=state_dict)

    def _unwrap_llava_backbone(self):
        if hasattr(self.base_model, "get_model"):
            return self.base_model.get_model()
        if hasattr(self.base_model, "base_model") and hasattr(self.base_model.base_model, "model"):
            return self.base_model.base_model.model
        if hasattr(self.base_model, "model"):
            return self.base_model.model
        raise AttributeError("Unable to locate underlying LLaVA backbone.")

    def _get_decoder_layers(self):
        backbone = self._unwrap_llava_backbone()
        if hasattr(backbone, "layers"):
            return backbone.layers
        if hasattr(backbone, "model") and hasattr(backbone.model, "layers"):
            return backbone.model.layers
        raise AttributeError("Unable to locate decoder layers for Module 2 attachment.")

    def _get_prepare_inputs_owner(self):
        if hasattr(self.base_model, "prepare_inputs_labels_for_multimodal"):
            return self.base_model
        if hasattr(self.base_model, "base_model") and hasattr(self.base_model.base_model, "prepare_inputs_labels_for_multimodal"):
            return self.base_model.base_model
        raise AttributeError("Unable to locate prepare_inputs_labels_for_multimodal.")

    def _get_embedding_layer(self):
        if hasattr(self.base_model, "get_input_embeddings"):
            return self.base_model.get_input_embeddings()
        if hasattr(self.base_model, "base_model") and hasattr(self.base_model.base_model, "get_input_embeddings"):
            return self.base_model.base_model.get_input_embeddings()
        raise AttributeError("Unable to locate token embedding layer.")

    def _get_text_backbone(self):
        return self._unwrap_llava_backbone()

    def _get_vision_modules(self):
        backbone = self._unwrap_llava_backbone()
        if hasattr(backbone, "get_vision_tower"):
            vision_tower = backbone.get_vision_tower()
        elif hasattr(self.base_model, "get_vision_tower"):
            vision_tower = self.base_model.get_vision_tower()
        else:
            raise AttributeError("Unable to locate vision tower.")
        mm_projector = backbone.mm_projector
        return vision_tower, mm_projector

    def set_loss_schedule(self, warmup_steps: int) -> None:
        self.loss_warmup_steps = max(0, int(warmup_steps))

    def get_name_memory_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if not key.startswith("base_model.")
        }

    def load_name_memory_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        target_param = next(self.base_model.parameters())
        target_device = target_param.device
        target_dtype = target_param.dtype

        slot_img_key = "name_memory_module.real_proto_tokens"
        pseudo_key = "name_memory_module.pseudo_proto_tokens"
        num_slots = None
        num_pseudo = None
        if slot_img_key in state_dict:
            num_slots = int(state_dict[slot_img_key].shape[0])
        if pseudo_key in state_dict:
            num_pseudo = int(state_dict[pseudo_key].shape[0])

        if num_slots is not None:
            self.name_memory_module.resize_real_bank(num_slots)
            self.num_slots = int(num_slots)
            self.config.name_memory_num_slots = int(num_slots)
        if num_pseudo is not None:
            self.name_memory_module.resize_pseudo_bank(num_pseudo)
            self.config.name_memory_num_pseudo_users = int(num_pseudo)

        self.name_memory_module.to(device=target_device, dtype=target_dtype)
        self.module2_adapter_stack.to(device=target_device, dtype=target_dtype)
        self.load_state_dict(state_dict, strict=False)

    def initialize_name_memory_from_registry(
        self,
        registry: Dict[str, object],
        tokenizer,
        image_aspect_ratio: str = "square",
        build_if_missing: bool = True,
    ) -> None:
        vision_tower, mm_projector = self._get_vision_modules()
        text_backbone = self._get_text_backbone()
        target_dtype = next(self.base_model.parameters()).dtype
        self.name_memory_module.to(device=self.device, dtype=target_dtype)
        self.module2_adapter_stack.to(device=self.device, dtype=target_dtype)
        self.name_memory_module.initialize_from_registry(
            registry=registry,
            tokenizer=tokenizer,
            text_backbone=text_backbone,
            vision_tower=vision_tower,
            mm_projector=mm_projector,
            image_aspect_ratio=image_aspect_ratio,
            device=self.device,
            model_identifier=str(getattr(self.base_model.config, "_name_or_path", "")),
            tokenizer_identifier=str(getattr(tokenizer, "name_or_path", "")),
            build_if_missing=build_if_missing,
        )
        self.name_memory_module.set_pref_annotation_ids_from_registry(registry, device=self.device)
        self.num_slots = int(registry.get("num_slots", 1))
        self.config.name_memory_num_slots = self.num_slots
        self.config.name_memory_num_pseudo_users = int(self.name_memory_module.num_pseudo_users)

    def _prepare_base_inputs(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
    ):
        prepare_owner = self._get_prepare_inputs_owner()
        if inputs_embeds is None:
            (
                _,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = prepare_owner.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
            )

        batch_size = inputs_embeds.shape[0]
        seq_len = inputs_embeds.shape[1]
        device = inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        return inputs_embeds, attention_mask, position_ids, past_key_values, labels

    def _append_name_memory_prefix(
        self,
        base_inputs_embeds: torch.FloatTensor,
        base_attention_mask: torch.Tensor,
        base_position_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor],
        user_slot_id: torch.LongTensor,
        concept_id: torch.LongTensor,
        source_kind: str,
    ):
        batch_size = user_slot_id.shape[0]
        device = base_inputs_embeds.device
        bundle = self.name_memory_module.build_prefix(
            user_slot_id=user_slot_id.to(device=device, dtype=torch.long),
            concept_id=concept_id.to(device=device, dtype=torch.long),
            source_kind=source_kind,
        )
        bundle["query_summary"] = self._build_query_summary(
            base_inputs_embeds=base_inputs_embeds,
            base_attention_mask=base_attention_mask,
            labels=labels,
        )
        if not self.enable_prefix:
            bundle["prefix_len"] = base_inputs_embeds.new_tensor(0, dtype=torch.long)
            return base_inputs_embeds, base_attention_mask, base_position_ids, labels, bundle
        prefix_embeds = bundle["prefix_embeds"].to(device=device, dtype=base_inputs_embeds.dtype)
        prefix_len = prefix_embeds.shape[1]
        bundle["prefix_len"] = prefix_embeds.new_tensor(prefix_len, dtype=torch.long)
        prefix_mask = torch.ones(batch_size, prefix_len, dtype=base_attention_mask.dtype, device=device)
        inputs_embeds = torch.cat([prefix_embeds, base_inputs_embeds], dim=1)
        attention_mask = torch.cat([prefix_mask, base_attention_mask], dim=1)
        prefix_positions = torch.arange(prefix_len, device=device, dtype=base_position_ids.dtype).unsqueeze(0).expand(batch_size, -1)
        position_ids = torch.cat([prefix_positions, base_position_ids + prefix_len], dim=1)

        if labels is None:
            prefixed_labels = None
        else:
            prefix_labels = torch.full((batch_size, prefix_len), IGNORE_INDEX, dtype=labels.dtype, device=labels.device)
            prefixed_labels = torch.cat([prefix_labels, labels], dim=1)
        return inputs_embeds, attention_mask, position_ids, prefixed_labels, bundle

    def _build_query_summary(
        self,
        *,
        base_inputs_embeds: torch.FloatTensor,
        base_attention_mask: torch.Tensor,
        labels: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        query_mask = base_attention_mask.to(device=base_inputs_embeds.device, dtype=torch.bool)
        if labels is not None:
            prompt_mask = labels.to(device=base_inputs_embeds.device).eq(IGNORE_INDEX)
            query_mask = query_mask & prompt_mask
        weights = query_mask.to(dtype=base_inputs_embeds.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (base_inputs_embeds * weights).sum(dim=1) / denom

    def _uses_annotation_density_focal_loss(self) -> bool:
        return _normalize_pref_residual_loss_mode(self.pref_residual_loss_mode) == "density_focal"

    def _build_hidden_query_summary(
        self,
        outputs: CausalLMOutputWithPast,
        *,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        bundle: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None or len(hidden_states) <= 0:
            return None
        hidden = hidden_states[-1]
        query_mask = attention_mask.to(device=hidden.device, dtype=torch.bool)
        if labels is not None and labels.shape[:2] == hidden.shape[:2]:
            query_mask = query_mask & labels.to(device=hidden.device).eq(IGNORE_INDEX)
        prefix_len_value = int(bundle.get("prefix_len", hidden.new_zeros((), dtype=torch.long)).item())
        if prefix_len_value > 0:
            query_mask = query_mask.clone()
            query_mask[:, :prefix_len_value] = False
        if not query_mask.any():
            query_mask = attention_mask.to(device=hidden.device, dtype=torch.bool)
            if prefix_len_value > 0:
                query_mask = query_mask.clone()
                query_mask[:, :prefix_len_value] = False
        weights = query_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (hidden * weights).sum(dim=1) / denom

    def _merge_context_bundles(self, bundles) -> Dict[str, torch.Tensor]:
        keys = (
            "name_token",
            "uid_token",
            "image_token",
            "description_token",
            "shared_pref_tokens",
            "offset_pref_tokens",
            "final_pref_tokens",
            "pref_tokens",
            "pref_gate",
            "profile_summary",
            "preference_summary",
            "query_summary",
            "concept_id",
        )
        merged = {}
        for key in keys:
            merged[key] = torch.cat([bundle[key] for bundle in bundles], dim=0)
        if "router_task_label" in bundles[0]:
            merged["router_task_label"] = torch.cat([bundle["router_task_label"] for bundle in bundles], dim=0)
        return merged

    def _run_base_forward(
        self,
        *,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values,
        labels: Optional[torch.LongTensor],
        bundle: Dict[str, torch.Tensor],
        use_cache: Optional[bool],
        output_attentions: Optional[bool],
        output_hidden_states: Optional[bool],
        return_dict: Optional[bool],
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if self.enable_module2:
            self.module2_adapter_stack.set_context(bundle)
        else:
            self.module2_adapter_stack.clear_context()
        need_hidden_states = self._uses_annotation_density_focal_loss() and labels is not None
        outputs = self.base_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True if need_hidden_states else output_hidden_states,
            return_dict=True if return_dict is None else return_dict,
            **kwargs,
        )
        return outputs
  ##contrasive的loss实现anchor 和 positive 先归一化
    def _mean_exp_similarity(
        self,
        anchor_vec: torch.Tensor,
        pool: torch.Tensor,
    ) -> torch.Tensor:
        if pool.numel() <= 0:
            return anchor_vec.new_zeros(())
        pool = F.normalize(pool.to(device=anchor_vec.device, dtype=anchor_vec.dtype), dim=-1, eps=1e-6)
        logits = torch.matmul(anchor_vec.unsqueeze(0), pool.t()).squeeze(0) / max(self.contrastive_temperature, 1e-6)
        return torch.exp(logits).mean()

    def _weighted_infonce_from_pools(
        self,
        anchor_vec: torch.Tensor,
        positive_vec: torch.Tensor,
        real_pool: torch.Tensor,
        pseudo_pool: torch.Tensor,
    ) -> torch.Tensor:
        pos_logit = torch.dot(anchor_vec, positive_vec) / max(self.contrastive_temperature, 1e-6)
        pos_term = torch.exp(pos_logit)
        denom = pos_term
        if real_pool.numel() > 0 and self.contrastive_real_weight > 0.0:
            denom = denom + self.contrastive_real_weight * self._mean_exp_similarity(anchor_vec, real_pool)
        if pseudo_pool.numel() > 0 and self.contrastive_pseudo_weight > 0.0:
            denom = denom + self.contrastive_pseudo_weight * self._mean_exp_similarity(anchor_vec, pseudo_pool)
        return -torch.log(pos_term / denom.clamp_min(1e-12))

    def _contrastive_infonce_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        user_slot_id: torch.Tensor,
        concept_id: torch.Tensor,
        pseudo_kind: str,
    ) -> torch.Tensor:
        valid = user_slot_id.ne(NAME_MEMORY_UNKNOWN_SLOT_ID)
        if valid.sum() <= 0:
            return anchor.new_zeros(())

        anchor_valid = F.normalize(anchor[valid], dim=-1, eps=1e-6)
        positive_valid = F.normalize(positive[valid], dim=-1, eps=1e-6)
        user_valid = user_slot_id[valid]
        concept_valid = concept_id[valid]
        losses = []
        for idx in range(anchor_valid.shape[0]):
            real_neg_mask = user_valid.ne(user_valid[idx])
            real_pool = positive_valid[real_neg_mask] if real_neg_mask.any() else anchor_valid.new_zeros((0, anchor_valid.shape[-1]))

            pseudo_candidates = self.name_memory_module.get_matching_pseudo_features(
                kind=pseudo_kind,
                concept_id=int(concept_valid[idx].item()),
            ) if self.enable_counterfactual else anchor_valid.new_zeros((0, anchor_valid.shape[-1]))
            losses.append(
                self._weighted_infonce_from_pools(
                    anchor_vec=anchor_valid[idx],
                    positive_vec=positive_valid[idx],
                    real_pool=real_pool,
                    pseudo_pool=pseudo_candidates,
                )
            )

        if not losses:
            return anchor.new_zeros(())
        return torch.stack(losses, dim=0).mean()

    def _single_pref_factor_loss(
        self,
        anchor_pref: torch.Tensor,
        proto_pref: torch.Tensor,
        user_slot_id: torch.Tensor,
        factor_idx: int,
    ) -> torch.Tensor:
        valid = user_slot_id.ne(NAME_MEMORY_UNKNOWN_SLOT_ID)
        if valid.sum() <= 0:
            return anchor_pref.new_zeros(())

        anchor_pref = anchor_pref[valid]
        proto_pref = proto_pref[valid]
        user_slot_id = user_slot_id[valid]

        anchor = F.normalize(anchor_pref, dim=-1, eps=1e-6)
        positive = F.normalize(proto_pref, dim=-1, eps=1e-6)

        losses = []
        for idx in range(anchor.shape[0]):
            real_neg_mask = user_slot_id.ne(user_slot_id[idx])
            real_pool = positive[real_neg_mask] if real_neg_mask.any() else anchor.new_zeros((0, anchor.shape[-1]))

            pseudo_pool = self.name_memory_module.get_matching_pseudo_features(
                kind="pref",
                concept_id=-1,
                factor_idx=factor_idx,
            ) if self.enable_counterfactual else anchor.new_zeros((0, anchor.shape[-1]))
            losses.append(
                self._weighted_infonce_from_pools(
                    anchor_vec=anchor[idx],
                    positive_vec=positive[idx],
                    real_pool=real_pool,
                    pseudo_pool=pseudo_pool,
                )
            )

        if not losses:
            return anchor.new_zeros(())
        return torch.stack(losses, dim=0).mean()

    def _preference_decorrelation_loss(
        self,
        slot_offset: torch.Tensor,
        shared_pref: torch.Tensor,
        user_slot_id: torch.Tensor,
    ) -> torch.Tensor:
        valid = user_slot_id.ne(NAME_MEMORY_UNKNOWN_SLOT_ID)
        if valid.sum() <= 0:
            return slot_offset.new_zeros(())
        residual = F.normalize(slot_offset[valid], dim=-1, eps=1e-6)
        num_factors = int(residual.shape[1])
        if num_factors <= 1:
            return slot_offset.new_zeros(())
        gram = torch.matmul(residual, residual.transpose(1, 2))
        pair_indices = torch.triu_indices(num_factors, num_factors, offset=1, device=gram.device)
        if pair_indices.numel() <= 0:
            return slot_offset.new_zeros(())
        pairwise = gram[:, pair_indices[0], pair_indices[1]]
        residual_pair_term = pairwise.pow(2).mean()

        prototype = shared_pref.to(device=residual.device, dtype=residual.dtype)
        if prototype.ndim == 2:
            prototype = prototype.unsqueeze(0).expand(residual.shape[0], -1, -1)
        elif prototype.shape[0] != residual.shape[0]:
            prototype = prototype[:1].expand(residual.shape[0], -1, -1)
        prototype = F.normalize(prototype[:, :num_factors, :], dim=-1, eps=1e-6)
        prototype_term = (residual[:, : prototype.shape[1], :] * prototype).sum(dim=-1).pow(2).mean()
        return residual_pair_term + prototype_term

    def _pref_factor_losses(
        self,
        slot_pref: torch.Tensor,
        proto_pref: torch.Tensor,
        user_slot_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        factor_terms = []
        for factor_idx, factor_name in enumerate(NAME_MEMORY_FACTORS):
            factor_loss = self._single_pref_factor_loss(
                anchor_pref=slot_pref[:, factor_idx, :],
                proto_pref=proto_pref[:, factor_idx, :],
                user_slot_id=user_slot_id,
                factor_idx=factor_idx,
            )
            losses[f"loss_pref_{factor_name}"] = factor_loss
            factor_terms.append(factor_loss)
        if factor_terms:
            losses["loss_pref_factor_contrast"] = torch.stack(factor_terms, dim=0).mean()
        else:
            losses["loss_pref_factor_contrast"] = slot_pref.new_zeros(())
        return losses

    def _pairwise_focal_logits_loss(
        self,
        positive_logit: torch.Tensor,
        negative_logits: torch.Tensor,
    ) -> torch.Tensor:
        pos_prob = torch.sigmoid(positive_logit)
        pos_loss = -((1.0 - pos_prob).pow(self.focal_gamma) * torch.log(pos_prob.clamp_min(1e-12)))
        if negative_logits.numel() <= 0:
            return pos_loss
        neg_prob = torch.sigmoid(negative_logits)
        neg_loss = -(neg_prob.pow(self.focal_gamma) * torch.log((1.0 - neg_prob).clamp_min(1e-12)))
        return 0.5 * (pos_loss + neg_loss.mean())

    def _single_pref_offset_focal_loss(
        self,
        slot_offset: torch.Tensor,
        proto_offset: torch.Tensor,
        user_slot_id: torch.Tensor,
    ) -> torch.Tensor:
        valid = user_slot_id.ne(NAME_MEMORY_UNKNOWN_SLOT_ID)
        if valid.sum() <= 0:
            return slot_offset.new_zeros(())

        slot_offset = F.normalize(slot_offset[valid], dim=-1, eps=1e-6)
        proto_offset = F.normalize(proto_offset[valid], dim=-1, eps=1e-6)
        user_slot_id = user_slot_id[valid]
        losses = []
        for idx in range(slot_offset.shape[0]):
            positive_logit = torch.dot(slot_offset[idx], proto_offset[idx]) / max(self.contrastive_temperature, 1e-6)
            real_neg_mask = user_slot_id.ne(user_slot_id[idx])
            if real_neg_mask.any():
                negative_logits = torch.matmul(
                    slot_offset[idx].unsqueeze(0),
                    proto_offset[real_neg_mask].t(),
                ).squeeze(0) / max(self.contrastive_temperature, 1e-6)
            else:
                negative_logits = slot_offset.new_zeros((0,))
            losses.append(self._pairwise_focal_logits_loss(positive_logit, negative_logits))
        if not losses:
            return slot_offset.new_zeros(())
        return torch.stack(losses, dim=0).mean()

    def _pref_offset_focal_loss(
        self,
        slot_offset: torch.Tensor,
        proto_offset: torch.Tensor,
        user_slot_id: torch.Tensor,
    ) -> torch.Tensor:
        factor_terms = []
        for factor_idx in range(slot_offset.shape[1]):
            factor_terms.append(
                self._single_pref_offset_focal_loss(
                    slot_offset=slot_offset[:, factor_idx, :],
                    proto_offset=proto_offset[:, factor_idx, :],
                    user_slot_id=user_slot_id,
                )
            )
        if not factor_terms:
            return slot_offset.new_zeros(())
        return torch.stack(factor_terms, dim=0).mean()

    def _query_conditioned_pref_activations(
        self,
        slot_offset: torch.Tensor,
        hidden_query_summary: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_query_summary is None:
            return slot_offset
        query = hidden_query_summary.to(device=slot_offset.device, dtype=slot_offset.dtype).unsqueeze(1)
        return slot_offset + query

    def _annotation_density_focal_pref_losses(
        self,
        slot_offset: torch.Tensor,
        user_slot_id: torch.Tensor,
        hidden_query_summary: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        factor_terms = []
        focal_terms = []
        valid_user = user_slot_id.ne(NAME_MEMORY_UNKNOWN_SLOT_ID)
        if valid_user.sum() <= 0:
            zero = slot_offset.new_zeros(())
            for factor_name in NAME_MEMORY_FACTORS:
                losses[f"loss_pref_{factor_name}"] = zero
            losses["loss_pref_factor_contrast"] = zero
            losses["loss_pref_focal"] = zero
            losses["loss_pref_residual_density_focal"] = zero
            return losses

        annotation_bank = self.name_memory_module.slot_pref_annotation_ids.to(device=user_slot_id.device)
        safe_slot_id = user_slot_id.clamp(min=0, max=max(0, annotation_bank.shape[0] - 1))
        annotation_ids = annotation_bank.index_select(0, safe_slot_id)
        activations = self._query_conditioned_pref_activations(slot_offset, hidden_query_summary)
        temperature = max(self.contrastive_temperature, 1e-6)
        eta = max(0.0, self.pref_density_eta)

        for factor_idx, factor_name in enumerate(NAME_MEMORY_FACTORS):
            if factor_idx >= activations.shape[1] or factor_idx >= annotation_ids.shape[1]:
                factor_loss = slot_offset.new_zeros(())
                losses[f"loss_pref_{factor_name}"] = factor_loss
                factor_terms.append(factor_loss)
                focal_terms.append(factor_loss)
                continue

            labels = annotation_ids[:, factor_idx]
            valid = valid_user & labels.ge(0)
            if valid.sum() <= 0:
                factor_loss = slot_offset.new_zeros(())
                losses[f"loss_pref_{factor_name}"] = factor_loss
                factor_terms.append(factor_loss)
                focal_terms.append(factor_loss)
                continue

            real_activations = activations[valid, factor_idx, :]
            factor_act = F.normalize(real_activations, dim=-1, eps=1e-6)
            factor_labels = labels[valid]

            pseudo_offsets_valid = factor_act.new_zeros((0, factor_act.shape[-1]))
            pseudo_labels = factor_labels.new_zeros((0,))
            if self.enable_counterfactual and getattr(self.name_memory_module, "num_pseudo_users", 0) > 0:
                pseudo_offsets = self.name_memory_module.get_pseudo_residual_features(factor_idx=factor_idx)
                pseudo_label_bank = getattr(self.name_memory_module, "pseudo_pref_annotation_ids", None)
                if pseudo_label_bank is not None and pseudo_offsets.numel() > 0:
                    pseudo_labels_all = pseudo_label_bank.to(device=factor_labels.device, dtype=torch.long)[:, factor_idx]
                    pseudo_valid = pseudo_labels_all.ge(0)
                    if pseudo_valid.any():
                        pseudo_offsets_valid = pseudo_offsets.to(device=real_activations.device, dtype=real_activations.dtype)[pseudo_valid]
                        pseudo_labels = pseudo_labels_all[pseudo_valid]

            p_values = []
            rho_values = []
            valid_query = (
                hidden_query_summary.to(device=real_activations.device, dtype=real_activations.dtype)[valid]
                if hidden_query_summary is not None
                else None
            )
            for anchor_idx in range(factor_act.shape[0]):
                candidate_act = factor_act
                candidate_labels = factor_labels
                if pseudo_offsets_valid.numel() > 0:
                    query_for_pseudo = (
                        valid_query[anchor_idx].unsqueeze(0)
                        if valid_query is not None
                        else pseudo_offsets_valid.new_zeros((1, pseudo_offsets_valid.shape[-1]))
                    )
                    pseudo_act = F.normalize(pseudo_offsets_valid + query_for_pseudo, dim=-1, eps=1e-6)
                    candidate_act = torch.cat([candidate_act, pseudo_act], dim=0)
                    candidate_labels = torch.cat([candidate_labels, pseudo_labels], dim=0)
                logits = torch.matmul(candidate_act, factor_act[anchor_idx].unsqueeze(-1)).squeeze(-1) / temperature
                logits = logits - logits.max()
                exp_logits = torch.exp(logits)
                same_group = candidate_labels.eq(factor_labels[anchor_idx])
                numerator = exp_logits[same_group].sum()
                denominator = exp_logits.sum().clamp_min(1e-12)
                p_values.append((numerator / denominator).clamp(min=1e-6, max=1.0))
                rho_values.append(factor_labels.eq(factor_labels[anchor_idx]).to(dtype=factor_act.dtype).mean().clamp(min=0.0, max=1.0))
            p_same = torch.stack(p_values, dim=0)
            rho = torch.stack(rho_values, dim=0)
            density_weight = (1.0 - rho).pow(eta)
            factor_contrast = -torch.log(p_same).mean()
            factor_focal = -(density_weight * (1.0 - p_same) * torch.log(p_same)).mean()
            losses[f"loss_pref_{factor_name}"] = factor_focal
            factor_terms.append(factor_contrast)
            focal_terms.append(factor_focal)

        contrast_loss = torch.stack(factor_terms, dim=0).mean() if factor_terms else slot_offset.new_zeros(())
        focal_loss = torch.stack(focal_terms, dim=0).mean() if focal_terms else slot_offset.new_zeros(())
        losses["loss_pref_factor_contrast"] = contrast_loss
        losses["loss_pref_focal"] = focal_loss
        losses["loss_pref_residual_density_focal"] = focal_loss
        return losses

    def _compute_aux_losses(
        self,
        bundle: Dict[str, torch.Tensor],
        user_slot_id: torch.Tensor,
        concept_id: torch.Tensor,
        hidden_query_summary: Optional[torch.Tensor] = None,
    ):
        slot_img = bundle["slot_img"]
        slot_desc = bundle["slot_desc"]
        slot_pref = bundle["final_pref_tokens"]
        proto_img = bundle["proto_img"]
        proto_desc = bundle["proto_desc"]
        proto_pref = bundle["final_proto_pref_tokens"]
        slot_pref_offset = bundle["raw_slot_pref"]
        proto_pref_offset = bundle["raw_proto_pref"]
        valid = user_slot_id.ne(NAME_MEMORY_UNKNOWN_SLOT_ID)

        if valid.any():
            consistency = (1.0 - F.cosine_similarity(slot_img[valid], slot_desc[valid], dim=-1)).mean()
        else:
            consistency = slot_img.new_zeros(())

        img_contrast = self._contrastive_infonce_loss(
            slot_img,
            proto_img,
            user_slot_id,
            concept_id,
            pseudo_kind="img",
        )
        desc_contrast = self._contrastive_infonce_loss(
            slot_desc,
            proto_desc,
            user_slot_id,
            concept_id,
            pseudo_kind="desc",
        )
        if self._uses_annotation_density_focal_loss():
            pref_losses = self._annotation_density_focal_pref_losses(slot_pref_offset, user_slot_id, hidden_query_summary)
            pref_focal = slot_pref_offset.new_zeros(())
        else:
            pref_losses = self._pref_factor_losses(slot_pref, proto_pref, user_slot_id)
            pref_focal = self._pref_offset_focal_loss(slot_pref_offset, proto_pref_offset, user_slot_id)
        pref_decorrelation = self._preference_decorrelation_loss(slot_pref_offset, bundle["shared_pref_tokens"], user_slot_id)
        losses = {
            "loss_consistency_img_desc": consistency,
            "loss_contrast_profile_img": img_contrast,
            "loss_contrast_description": desc_contrast,
            "loss_pref_decorrelation": pref_decorrelation,
            "loss_pref_focal": pref_focal,
        }
        losses.update(pref_losses)
        return losses

    def _current_aux_scale(self) -> float:
        if self.loss_warmup_steps <= 0:
            return 1.0
        step = int(self.name_memory_forward_steps.item())
        return min(1.0, float(step) / float(max(1, self.loss_warmup_steps)))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        user_slot_id: Optional[torch.LongTensor] = None,
        concept_id: Optional[torch.LongTensor] = None,
        router_task_label: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if user_slot_id is None:
            raise ValueError("user_slot_id is required for name-memory forward.")
        if concept_id is None:
            concept_id = torch.full_like(user_slot_id, -1)

        base_inputs_embeds, base_attention_mask, base_position_ids, past_key_values, labels = self._prepare_base_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            images=images,
        )
        inputs_embeds, attention_mask, position_ids, labels, bundle = self._append_name_memory_prefix(
            base_inputs_embeds,
            base_attention_mask,
            base_position_ids,
            labels,
            user_slot_id,
            concept_id,
            source_kind="real",
        )
        if router_task_label is not None:
            bundle["router_task_label"] = router_task_label.to(self.device, dtype=torch.long)

        outputs = self._run_base_forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            bundle=bundle,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        module2_losses = self.module2_adapter_stack.get_runtime_aux_losses()
        hidden_query_summary = self._build_hidden_query_summary(
            outputs,
            attention_mask=attention_mask,
            labels=labels,
            bundle=bundle,
        ) if self._uses_annotation_density_focal_loss() and labels is not None else None

        aux_losses = self._compute_aux_losses(
            bundle=bundle,
            user_slot_id=user_slot_id.to(self.device, dtype=torch.long),
            concept_id=concept_id.to(self.device, dtype=torch.long),
            hidden_query_summary=hidden_query_summary,
        )
        vqa_loss = outputs.loss
        total_loss = None
        if vqa_loss is not None:
            aux_scale = self._current_aux_scale()
            zero_aux = torch.zeros((), device=vqa_loss.device, dtype=vqa_loss.dtype)
            loss_task_router = module2_losses.get("loss_task_router", zero_aux)
            loss_pref_router = module2_losses.get("loss_pref_router", zero_aux)
            total_loss = vqa_loss + aux_scale * (
                self.loss_weight_consistency * aux_losses["loss_consistency_img_desc"]
                + self.loss_weight_profile_img * aux_losses["loss_contrast_profile_img"]
                + self.loss_weight_description * aux_losses["loss_contrast_description"]
                + self.loss_weight_pref_factor * aux_losses["loss_pref_factor_contrast"]
                + self.loss_weight_pref_decorrelation * aux_losses["loss_pref_decorrelation"]
                + self.loss_weight_pref_focal * aux_losses["loss_pref_focal"]
                + self.loss_weight_task_router * loss_task_router
                + self.loss_weight_pref_router * loss_pref_router
            )

        if self.training:
            self.name_memory_forward_steps += 1

        outputs.loss = total_loss
        outputs.name_memory_losses = {
            "loss_vqa": vqa_loss.detach() if vqa_loss is not None else torch.zeros((), device=self.device),
            **aux_losses,
            **module2_losses,
        }
        return outputs

    def generate(
        self,
        input_ids: torch.LongTensor = None,
        images: Optional[torch.FloatTensor] = None,
        user_slot_id: Optional[torch.LongTensor] = None,
        concept_id: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids is required for name-memory generation.")

        max_new_tokens = int(kwargs.pop("max_new_tokens", 32))
        min_new_tokens = int(kwargs.pop("min_new_tokens", 1))
        do_sample = bool(kwargs.pop("do_sample", False))
        temperature = float(kwargs.pop("temperature", 1.0) or 1.0)
        eos_token_id = kwargs.pop("eos_token_id", getattr(self.config, "eos_token_id", None))
        kwargs.pop("use_cache", None)
        if kwargs:
            unsupported = ", ".join(sorted(kwargs.keys()))
            raise NotImplementedError(f"Unsupported generation kwargs for name-memory manual decode: {unsupported}")

        generated_ids = input_ids.to(self.device)
        images = images.to(self.device, dtype=self.dtype) if images is not None else None
        if attention_mask is None:
            attention_mask = torch.ones_like(generated_ids, dtype=torch.long, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)
        user_slot_id = user_slot_id.to(self.device, dtype=torch.long) if user_slot_id is not None else None
        concept_id = concept_id.to(self.device, dtype=torch.long) if concept_id is not None else None
        blocked_first_token_ids = set()
        for token_id in (
            eos_token_id,
            getattr(self.config, "pad_token_id", None),
            getattr(self.config, "bos_token_id", None),
        ):
            if token_id is None:
                continue
            if isinstance(token_id, (list, tuple, set)):
                blocked_first_token_ids.update(int(x) for x in token_id if x is not None)
            else:
                blocked_first_token_ids.add(int(token_id))

        for step in range(max_new_tokens):
            outputs = self.forward(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                labels=None,
                images=images,
                user_slot_id=user_slot_id,
                concept_id=concept_id,
                use_cache=False,
                return_dict=True,
            )
            next_token_logits = outputs.logits[:, -1, :].clone()
            if step < min_new_tokens and blocked_first_token_ids:
                valid_block_ids = [idx for idx in blocked_first_token_ids if 0 <= idx < next_token_logits.shape[-1]]
                if valid_block_ids:
                    next_token_logits[:, valid_block_ids] = -torch.inf
            if do_sample:
                probs = torch.softmax(next_token_logits / max(temperature, 1e-5), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)],
                dim=1,
            )

            if eos_token_id is not None and bool(torch.all(next_token.eq(int(eos_token_id)))):
                break

        return generated_ids

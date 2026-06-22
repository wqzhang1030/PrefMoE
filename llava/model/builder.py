#    Copyright 2023 Haotian Liu
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


import json
import os, sys
import warnings
import shutil
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _maybe_wrap_name_memory_model(model, model_path: str):
    use_name_memory = bool(getattr(getattr(model, 'config', None), 'use_name_memory', False))
    trainables_path = os.path.join(model_path, 'name_memory_trainables.bin')
    if (not use_name_memory) and (not os.path.exists(trainables_path)):
        return model

    num_slots = int(getattr(model.config, 'name_memory_num_slots', 1) or 1)
    num_factors = int(getattr(model.config, 'name_memory_num_factors', 5) or 5)
    shared_pref_expert_count = int(
        getattr(model.config, 'name_memory_shared_pref_expert_count', num_factors) or num_factors
    )
    top_layers = int(getattr(model.config, 'name_memory_top_layers', 8) or 8)
    factor_margin = float(getattr(model.config, 'name_memory_factor_margin', 0.2) or 0.2)
    num_pseudo_users = int(getattr(model.config, 'name_memory_num_pseudo_users', 50) or 50)
    pseudo_csv_path = str(getattr(model.config, 'name_memory_pseudo_csv_path', 'data/mmpb_clean/pseudo_users.csv'))
    shared_pref_rank = int(getattr(model.config, 'name_memory_shared_pref_lora_rank', 64) or 64)
    user_pref_rank = int(getattr(model.config, 'name_memory_user_pref_rank', 64) or 64)
    user_profile_rank = int(getattr(model.config, 'name_memory_user_profile_rank', 32) or 32)
    profile_expert_count = int(getattr(model.config, 'name_memory_profile_expert_count', 2) or 2)
    module2_pref_profile_mix = float(getattr(model.config, 'name_memory_module2_pref_profile_mix', 0.7) or 0.7)
    module2_arch = str(getattr(model.config, 'name_memory_module2_arch', 'legacy_embedded') or 'legacy_embedded')
    use_backbone_lora = bool(getattr(model.config, 'name_memory_use_backbone_lora', True))
    module1_mode = str(getattr(model.config, 'name_memory_module1_mode', 'token_bank') or 'token_bank')
    token_bank_path = str(getattr(model.config, 'name_memory_token_bank_path', '') or '')
    token_delta_scale = float(getattr(model.config, 'name_memory_token_delta_scale', 0.1) or 0.1)
    enable_prefix = bool(getattr(model.config, 'name_memory_enable_prefix', True))
    enable_module2 = bool(getattr(model.config, 'name_memory_enable_module2', True))
    enable_counterfactual = bool(getattr(model.config, 'name_memory_enable_counterfactual', True))
    routing_mode = str(getattr(model.config, 'name_memory_routing_mode', 'hierarchical') or 'hierarchical')
    task_router_mode = str(getattr(model.config, 'name_memory_task_router_mode', 'memory_only') or 'memory_only')
    task_router_fixed_pref_weight = float(
        getattr(model.config, 'name_memory_task_router_fixed_pref_weight', 0.6) or 0.6
    )
    task_router_target_confidence = float(
        getattr(model.config, 'name_memory_task_router_target_confidence', 1.0) or 1.0
    )
    pref_router_mode = str(getattr(model.config, 'name_memory_pref_router_mode', 'learned') or 'learned')
    pref_router_fixed_weights = str(getattr(model.config, 'name_memory_pref_router_fixed_weights', '') or '')
    pref_router_supervision = str(getattr(model.config, 'name_memory_pref_router_supervision', 'none') or 'none')
    pref_router_loss_weight = float(getattr(model.config, 'name_memory_pref_router_loss_weight', 0.0) or 0.0)
    pref_router_target_confidence = float(
        getattr(model.config, 'name_memory_pref_router_target_confidence', 0.9) or 0.9
    )
    pref_context_mode = str(getattr(model.config, 'name_memory_pref_context_mode', 'all_factors') or 'all_factors')
    profile_router_mode = str(getattr(model.config, 'name_memory_profile_router_mode', 'learned') or 'learned')
    profile_router_fixed_weights = str(getattr(model.config, 'name_memory_profile_router_fixed_weights', '') or '')
    hier_lora_context_mode = str(getattr(model.config, 'name_memory_hier_lora_context_mode', 'none') or 'none')
    hier_lora_target_modules = str(getattr(model.config, 'name_memory_hier_lora_target_modules', '') or '')
    use_profile_image = bool(getattr(model.config, 'name_memory_use_profile_image', True))
    use_description = bool(getattr(model.config, 'name_memory_use_description', True))
    use_preference = bool(getattr(model.config, 'name_memory_use_preference', True))
    use_factorized_preference_memory = bool(getattr(model.config, 'name_memory_use_factorized_preference_memory', True))
    prefix_compose_mode = str(
        getattr(
            model.config,
            'name_memory_prefix_compose_mode',
            getattr(model.config, 'name_memory_binding_mode', 'split13'),
        )
        or 'split13'
    )
    task_router_supervision = str(getattr(model.config, 'name_memory_task_router_supervision', 'none') or 'none')
    task_router_loss_weight = float(getattr(model.config, 'name_memory_task_router_loss_weight', 0.0) or 0.0)
    loss_weight_consistency = float(getattr(model.config, 'name_memory_loss_weight_consistency', 0.05) or 0.0)
    loss_weight_profile_img = float(getattr(model.config, 'name_memory_loss_weight_profile_img', 0.10) or 0.0)
    loss_weight_description = float(getattr(model.config, 'name_memory_loss_weight_description', 0.10) or 0.0)
    loss_weight_pref_factor = float(getattr(model.config, 'name_memory_loss_weight_pref_factor', 0.20) or 0.0)
    loss_weight_pref_decorrelation = float(getattr(model.config, 'name_memory_loss_weight_pref_decorrelation', 0.05) or 0.0)
    loss_weight_pref_focal = float(getattr(model.config, 'name_memory_loss_weight_pref_focal', 0.20) or 0.0)
    contrastive_temperature = float(getattr(model.config, 'name_memory_contrastive_temperature', 0.07) or 0.07)
    contrastive_real_weight = float(getattr(model.config, 'name_memory_contrastive_real_weight', 1.0) or 0.0)
    contrastive_pseudo_weight = float(getattr(model.config, 'name_memory_contrastive_pseudo_weight', 0.5) or 0.0)
    focal_gamma = float(getattr(model.config, 'name_memory_focal_gamma', 2.0) or 2.0)
    pref_residual_loss_mode = str(getattr(model.config, 'name_memory_pref_residual_loss_mode', 'density_focal') or 'density_focal')
    pref_density_eta = float(getattr(model.config, 'name_memory_pref_density_eta', 1.0) or 0.0)
    wrapped = LlavaLlamaNameMemoryWrapper(
        model,
        num_slots=num_slots,
        num_factors=num_factors,
        shared_pref_expert_count=shared_pref_expert_count,
        top_layers=top_layers,
        factor_margin=factor_margin,
        num_pseudo_users=num_pseudo_users,
        pseudo_csv_path=pseudo_csv_path,
        shared_pref_rank=shared_pref_rank,
        user_pref_rank=user_pref_rank,
        user_profile_rank=user_profile_rank,
        profile_expert_count=profile_expert_count,
        module2_pref_profile_mix=module2_pref_profile_mix,
        module2_arch=module2_arch,
        use_backbone_lora=use_backbone_lora,
        module1_mode=module1_mode,
        token_bank_path=token_bank_path,
        token_delta_scale=token_delta_scale,
        enable_prefix=enable_prefix,
        enable_module2=enable_module2,
        enable_counterfactual=enable_counterfactual,
        routing_mode=routing_mode,
        task_router_mode=task_router_mode,
        task_router_fixed_pref_weight=task_router_fixed_pref_weight,
        task_router_target_confidence=task_router_target_confidence,
        pref_router_mode=pref_router_mode,
        pref_router_fixed_weights=pref_router_fixed_weights,
        pref_router_supervision=pref_router_supervision,
        pref_router_loss_weight=pref_router_loss_weight,
        pref_router_target_confidence=pref_router_target_confidence,
        pref_context_mode=pref_context_mode,
        profile_router_mode=profile_router_mode,
        profile_router_fixed_weights=profile_router_fixed_weights,
        hier_lora_context_mode=hier_lora_context_mode,
        hier_lora_target_modules=hier_lora_target_modules,
        use_profile_image=use_profile_image,
        use_description=use_description,
        use_preference=use_preference,
        use_factorized_preference_memory=use_factorized_preference_memory,
        prefix_compose_mode=prefix_compose_mode,
        task_router_supervision=task_router_supervision,
        task_router_loss_weight=task_router_loss_weight,
        loss_weight_consistency=loss_weight_consistency,
        loss_weight_profile_img=loss_weight_profile_img,
        loss_weight_description=loss_weight_description,
        loss_weight_pref_factor=loss_weight_pref_factor,
        loss_weight_pref_decorrelation=loss_weight_pref_decorrelation,
        loss_weight_pref_focal=loss_weight_pref_focal,
        contrastive_temperature=contrastive_temperature,
        contrastive_real_weight=contrastive_real_weight,
        contrastive_pseudo_weight=contrastive_pseudo_weight,
        focal_gamma=focal_gamma,
        pref_residual_loss_mode=pref_residual_loss_mode,
        pref_density_eta=pref_density_eta,
    )
    _load_name_memory_original_module_weights(wrapped, model_path)
    if os.path.exists(trainables_path):
        state_dict = torch.load(trainables_path, map_location='cpu')
        wrapped.load_name_memory_state_dict(state_dict)
    return wrapped


def _load_name_memory_original_module_weights(model, model_path: str):
    """Load wrapped top-layer backbone weights saved under `.original_module.` keys.

    Name-memory checkpoints keep the injected top layers under wrapper-owned
    `original_module` paths. A plain `LlavaLlamaForCausalLM.from_pretrained(...)`
    call cannot consume those keys, so the wrapped model needs a second pass.
    """
    shard_paths = []
    bin_index = os.path.join(model_path, 'pytorch_model.bin.index.json')
    if os.path.exists(bin_index):
        with open(bin_index, 'r', encoding='utf-8') as f:
            weight_map = json.load(f).get('weight_map', {})
        shard_paths = sorted(
            {os.path.join(model_path, shard) for key, shard in weight_map.items() if '.original_module.' in key}
        )
    else:
        single_bin = os.path.join(model_path, 'pytorch_model.bin')
        if os.path.exists(single_bin):
            shard_paths = [single_bin]

    if not shard_paths:
        return

    original_module_state = {}
    for shard_path in shard_paths:
        shard_state = torch.load(shard_path, map_location='cpu')
        for key, value in shard_state.items():
            if '.original_module.' in key:
                original_module_state[key] = value

    if not original_module_state:
        return

    target_model = getattr(model, 'base_model', model)
    _, unexpected = target_model.load_state_dict(original_module_state, strict=False)
    if unexpected:
        warnings.warn(
            f'Unexpected original_module keys when restoring wrapped name-memory weights: {unexpected[:5]}'
        )


def _has_full_model_weights(model_path: str) -> bool:
    """Return True when `model_path` stores a full HF checkpoint, not just adapter/projector deltas."""
    index_files = {
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "model.safetensors",
        "model.safetensors.index.json",
    }
    if any(os.path.exists(os.path.join(model_path, name)) for name in index_files):
        return True

    try:
        filenames = os.listdir(model_path)
    except OSError:
        return False

    for name in filenames:
        if name.startswith("pytorch_model-") and name.endswith(".bin"):
            return True
        if name.startswith("model-") and name.endswith(".safetensors"):
            return True
    return False


def _is_name_memory_only_checkpoint(model_path: str) -> bool:
    """Name-memory checkpoint that stores our Module2 weights, not PEFT LoRA."""
    trainables_path = os.path.join(model_path, "name_memory_trainables.bin")
    if not os.path.exists(trainables_path):
        return False
    peft_weight_names = {
        "adapter_model.bin",
        "adapter_model.safetensors",
    }
    return not any(os.path.exists(os.path.join(model_path, name)) for name in peft_weight_names)


def _load_llava_tokenizer(model_path: str, model_base: str = None, use_fast: bool = False):
    """Load tokenizer from checkpoint when available, otherwise fall back to base model.

    Full LLaVA checkpoints in this repo may omit tokenizer files and only keep model
    weights/config. In that case `AutoTokenizer.from_pretrained(model_path)` can fail
    with config-mapping errors such as `KeyError: LlavaConfig` during eval.
    """
    try:
        return AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)
    except Exception as exc:
        if model_base is None:
            raise
        warnings.warn(
            f"Falling back to tokenizer from model_base because loading tokenizer from "
            f"{model_path} failed: {type(exc).__name__}: {exc}"
        )
        return AutoTokenizer.from_pretrained(model_base, use_fast=use_fast)

def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", **kwargs):
    kwargs = {"device_map": device_map, **kwargs}
    is_name_memory_only = _is_name_memory_only_checkpoint(model_path)

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None and not is_name_memory_only:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from PrefMoE.peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            if _has_full_model_weights(model_path):
                print('Loading full LLaVA checkpoint from model path...')
                if 'mpt' in model_name.lower():
                    tokenizer = _load_llava_tokenizer(model_path, model_base=model_base, use_fast=True)
                    model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                else:
                    tokenizer = _load_llava_tokenizer(model_path, model_base=model_base, use_fast=False)
                    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                # this may be mm projector only
                print('Loading LLaVA from base model...')
                if 'mpt' in model_name.lower():
                    if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                        shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                    cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                    model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                    cfg_pretrained = AutoConfig.from_pretrained(model_path)
                    model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

                mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
                mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
                model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from PrefMoE.peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        model = _maybe_wrap_name_memory_model(model, model_path)
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


def load_pretrained_model_v2(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", **kwargs):
    kwargs = {"device_map": device_map, **kwargs}
    is_name_memory_only = _is_name_memory_only_checkpoint(model_path)

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None and not is_name_memory_only:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from PrefMoE.peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            if _has_full_model_weights(model_path):
                print('Loading full LLaVA checkpoint from model path...')
                if 'mpt' in model_name.lower():
                    tokenizer = _load_llava_tokenizer(model_path, model_base=model_base, use_fast=True)
                    model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                else:
                    tokenizer = _load_llava_tokenizer(model_path, model_base=model_base, use_fast=False)
                    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                # this may be mm projector only
                print('Loading LLaVA from base model...')
                if 'mpt' in model_name.lower():
                    if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                        shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                    cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                    model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                    cfg_pretrained = AutoConfig.from_pretrained(model_path)
                    model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

                mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
                mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
                model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from PrefMoE.peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        model = _maybe_wrap_name_memory_model(model, model_path)
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len

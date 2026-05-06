import hashlib
import json
import os
import os.path as osp
import re
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .mmpb_clean_user_bank import load_pseudo_user_bank
##
##1.映射空间的定义
##
NAME_MEMORY_FACTORS = (
    "entertainment",
    "travel",
    "lifestyle",
    "shopping",
    "fashion",
)
NAME_MEMORY_FACTOR_TO_ID = {name: idx for idx, name in enumerate(NAME_MEMORY_FACTORS)}
NAME_MEMORY_UNKNOWN_SLOT_ID = 0
NAME_MEMORY_DEFAULT_PSEUDO_CSV = "data/mmpb_clean/pseudo_users.csv"
NAME_MEMORY_DEFAULT_TOKEN_BANK_DIR = "outputs/token_banks"
NAME_MEMORY_BINDER_BOTTLENECK = 256
##总共8token输入
NAME_MEMORY_ROLE_NAMES = (
    "name",
    "image",
    "description",
    "shared_pref_entertainment",
    "offset_pref_entertainment",
    "shared_pref_travel",
    "offset_pref_travel",
    "shared_pref_lifestyle",
    "offset_pref_lifestyle",
    "shared_pref_shopping",
    "offset_pref_shopping",
    "shared_pref_fashion",
    "offset_pref_fashion",
)
NAME_MEMORY_TOKEN_BUILD_VERSION = "v3_shared_point_offset_20260424"
NAME_MEMORY_ROLE_BINDER_BUILD_VERSION = "v2_role_binder_20260318"
NAME_MEMORY_BINDER_ROLE_NAMES = (
    "name",
    "image",
    "description",
    "pref_entertainment",
    "pref_travel",
    "pref_lifestyle",
    "pref_shopping",
    "pref_fashion",
)
##没必要但是就这样吧，泛化能力强点
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".ppm")
_FACTOR_PATTERN = re.compile(r"In terms of ([^,]+), (.*?)(?=In terms of [^,]+,|$)", re.IGNORECASE | re.DOTALL)


def _project_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").exists() and (parent / "llava").is_dir():
            return parent
    return path.parents[3]


def _get_text_embedding_layer(text_backbone):
    if hasattr(text_backbone, "get_input_embeddings"):
        try:
            layer = text_backbone.get_input_embeddings()
        except Exception:
            layer = None
        if layer is not None:
            return layer
    layer = getattr(text_backbone, "embed_tokens", None)
    if layer is not None:
        return layer
    return None


def _gathered_params_if_needed(parameters):
    params = [param for param in parameters if isinstance(param, torch.nn.Parameter)]
    if not params:
        return nullcontext()
    try:
        from deepspeed import zero as deepspeed_zero
    except Exception:
        return nullcontext()
    if not any(hasattr(param, "ds_id") for param in params):
        return nullcontext()
    return deepspeed_zero.GatheredParameters(params, modifier_rank=None)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_json_hash(payload: Dict[str, object]) -> str:
    return _sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def _sha256_file(path: str) -> str:
    target = Path(path)
    if not target.exists() or not target.is_file():
        return ""
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def resolve_description_field(prompt_type: str) -> str:
    prompt_type = str(prompt_type or "hard_moderate").strip().lower()
    mapping = {
        "hard_moderate": "description_moderate",
        "hard_detailed": "description_detailed",
        "hard_simple": "description_simple",
        "hard_super_detailed": "description_super_detailed",
    }
    return mapping.get(prompt_type, "description_moderate")

##按照concept切preference——对照nameid
def concept_to_factor_id(concept) -> int:
    key = str(concept or "").strip().lower()
    return NAME_MEMORY_FACTOR_TO_ID.get(key, -1)


def split_preference_factors(preference_text: str) -> List[str]:
    text = str(preference_text or "").strip()
    if not text:
        return [""] * len(NAME_MEMORY_FACTORS)

    matched: Dict[str, str] = {}
    for factor_name, body in _FACTOR_PATTERN.findall(text):
        key = str(factor_name).strip().lower()
        if key in NAME_MEMORY_FACTOR_TO_ID:
            matched[key] = f"In terms of {key}, {str(body).strip()}".strip()

    if not matched:
        return [text] * len(NAME_MEMORY_FACTORS)
    return [matched.get(factor_name, text) for factor_name in NAME_MEMORY_FACTORS]


def _select_profile_image_path(row: pd.Series) -> str:
    for idx in range(1, 6):
        key = f"injection_image_{idx}"
        val = str(row.get(key, "")).strip()
        if val:
            return val
    return str(row.get("image_path", "")).strip()


def _resolve_image_path(image_folder: str, rel_or_abs: str) -> str:
    path = str(rel_or_abs or "").strip()
    if not path:
        return path
    if osp.isabs(path):
        return path

    candidate = osp.normpath(osp.join(image_folder, path))
    if osp.exists(candidate):
        return candidate

    fallback = osp.normpath(osp.join(osp.dirname(image_folder), path))
    if osp.exists(fallback):
        return fallback
    return candidate


def _normalize_preprocessed_image(processed):
    if torch.is_tensor(processed):
        if processed.dim() > 0 and processed.shape[0] == 1:
            return processed[0]
        return processed
    if isinstance(processed, Mapping):
        normalized = {}
        for key, value in processed.items():
            if not torch.is_tensor(value):
                continue
            if value.dim() > 0 and value.shape[0] == 1:
                normalized[key] = value[0]
            else:
                normalized[key] = value
        if "pixel_values" not in normalized:
            raise ValueError("processor.preprocess must return pixel_values for name-memory image encoding")
        return normalized if len(normalized) > 1 else normalized["pixel_values"]
    raise TypeError("Unsupported image processor output type: {}".format(type(processed).__name__))


def _preprocess_pil_image(image: Image.Image, processor, image_aspect_ratio: str):
    if image_aspect_ratio == "pad" and hasattr(processor, "image_mean"):
        width, height = image.size
        if width != height:
            if width > height:
                padded = Image.new(image.mode, (width, width), tuple(int(x * 255) for x in processor.image_mean))
                padded.paste(image, (0, (width - height) // 2))
                image = padded
            else:
                padded = Image.new(image.mode, (height, height), tuple(int(x * 255) for x in processor.image_mean))
                padded.paste(image, ((height - width) // 2, 0))
                image = padded
    return _normalize_preprocessed_image(processor.preprocess(image, return_tensors="pt"))


def _preprocess_image(image_path: str, processor, image_aspect_ratio: str):
    image = Image.open(image_path).convert("RGB")
    return _preprocess_pil_image(image, processor, image_aspect_ratio)


def build_fold_train_visible_user_registry(
    data_frame: pd.DataFrame,
    data_split,
    fold_idx: int,
    image_folder: str,
    description_field: str,
) -> Dict[str, object]:
    fold_tasks = data_split[int(fold_idx)]["tasks"]
    train_indices = sorted({int(idx) for task in fold_tasks for idx in task.get("train_idx", [])})
    visible_df = data_frame.iloc[train_indices].copy()
    if "index" not in visible_df.columns:
        visible_df = visible_df.reset_index().rename(columns={"index": "index"})

    visible_df["name"] = visible_df["name"].astype(str)
    slot_by_name = {}
    records = [
        {
            "slot_id": NAME_MEMORY_UNKNOWN_SLOT_ID,
            "name": "",
            "profile_image_path": "",
            "description_text": "",
            "preference_text": "",
            "factor_texts": [""] * len(NAME_MEMORY_FACTORS),
        }
    ]

    names = sorted(x for x in visible_df["name"].dropna().unique().tolist() if str(x).strip())
    for slot_id, name in enumerate(names, start=1):
        user_rows = visible_df[visible_df["name"] == name].sort_values("index")
        row = user_rows.iloc[0]
        description_text = str(row.get(description_field, "") or "")
        preference_text = str(row.get("preference", "") or "")
        records.append(
            {
                "slot_id": slot_id,
                "name": str(name),
                "profile_image_path": _resolve_image_path(image_folder, _select_profile_image_path(row)),
                "description_text": description_text,
                "preference_text": preference_text,
                "factor_texts": split_preference_factors(preference_text),
            }
        )
        slot_by_name[str(name)] = slot_id

    return {
        "fold_idx": int(fold_idx),
        "description_field": description_field,
        "slot_by_name": slot_by_name,
        "records": records,
        "num_slots": len(records),
    }


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden_states * mask).sum(dim=1) / denom


def _format_name_prompt(name: str) -> str:
    return f"User name: {_normalize_text(name)}."


def _format_description_prompt(text: str) -> str:
    return f"User description: {_normalize_text(text)}."


def _format_preference_prompt(factor_name: str, text: str) -> str:
    return f"User preference in {factor_name}: {_normalize_text(text)}."


class _RoleAwareTokenBinder(nn.Module):
    def __init__(self, hidden_size: int, bottleneck_dim: int = NAME_MEMORY_BINDER_BOTTLENECK):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        num_heads = 4 if bottleneck_dim % 4 == 0 else 1
        self.down = nn.Linear(hidden_size, bottleneck_dim, bias=False)
        self.norm1 = nn.LayerNorm(bottleneck_dim)
        self.attn = nn.MultiheadAttention(bottleneck_dim, num_heads=num_heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(bottleneck_dim)
        self.mlp = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim, bias=False),
        )
        self.up = nn.Linear(bottleneck_dim, hidden_size, bias=False)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, tokens_with_roles: torch.Tensor) -> torch.Tensor:
        low = self.down(tokens_with_roles)
        attn_in = self.norm1(low)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        low = low + attn_out
        mlp_in = self.norm2(low)
        low = low + self.mlp(mlp_in)
        delta = torch.tanh(self.scale) * self.up(low)
        delta = delta.clone()
        delta[:, 0, :] = 0.0
        return tokens_with_roles + delta


class NameMemoryModule(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_factors: int = len(NAME_MEMORY_FACTORS),
        num_pseudo_users: int = 50,
        pseudo_csv_path: str = NAME_MEMORY_DEFAULT_PSEUDO_CSV,
        token_bank_path: str = "",
        token_delta_scale: float = 0.1,
        module1_mode: str = "token_bank",
        use_profile_image: bool = True,
        use_description: bool = True,
        use_preference: bool = True,
        use_factorized_preference_memory: bool = True,
        prefix_compose_mode: str = "split13",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_factors = num_factors
        self.num_pseudo_users = int(num_pseudo_users)
        self.pseudo_csv_path = str(pseudo_csv_path)
        self.token_bank_path = str(token_bank_path or "").strip()
        self.token_delta_scale = float(token_delta_scale)
        self.module1_mode = str(module1_mode or "token_bank")
        self.use_profile_image = bool(use_profile_image)
        self.use_description = bool(use_description)
        self.use_preference = bool(use_preference)
        self.use_factorized_preference_memory = bool(use_factorized_preference_memory)
        self.prefix_compose_mode = self._normalize_prefix_compose_mode(prefix_compose_mode)
        if str(self.module1_mode or "").strip().lower() in {
            "role_binder",
            "role_binder_v2",
            "v2_role_binder",
            "legacy_role_binder",
        }:
            self.prefix_compose_mode = "role_binder_v2"

        self.pref_gate_bias = nn.Parameter(torch.zeros(self.num_factors))
        self.shared_pref_points = nn.Parameter(torch.zeros(self.num_factors, hidden_size))
        self.real_content_delta = nn.Parameter(torch.zeros(1, 7, hidden_size))
        self.token_role_embeddings = nn.Parameter(torch.zeros(len(NAME_MEMORY_BINDER_ROLE_NAMES), hidden_size))
        self.token_binder = _RoleAwareTokenBinder(hidden_size=hidden_size, bottleneck_dim=NAME_MEMORY_BINDER_BOTTLENECK)

        self.register_buffer("real_proto_tokens", torch.zeros(1, 8, hidden_size), persistent=True)
        self.register_buffer("pseudo_proto_tokens", torch.zeros(self.num_pseudo_users, 8, hidden_size), persistent=True)
        self.register_buffer(
            "slot_pref_annotation_ids",
            torch.full((1, self.num_factors), -1, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "pseudo_pref_annotation_ids",
            torch.full((self.num_pseudo_users, self.num_factors), -1, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "pseudo_family_ids",
            torch.zeros(self.num_pseudo_users, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer("pseudo_cycle_cursor", torch.zeros(1, dtype=torch.long), persistent=False)
        self.initialized_num_slots = 1
        self._registry_cache_key = None
        self._current_manifest_hash = ""
        self._pseudo_pref_texts = []

    @staticmethod
    def _normalize_prefix_compose_mode(mode: str) -> str:
        mode = str(mode or "split13").strip().lower()
        if mode in {"split", "split13", "shared_offset_split", "shared_point_offset"}:
            return "split13"
        if mode in {"sum", "sum8", "shared_offset_sum", "shared_point_offset_sum8"}:
            return "sum8"
        if mode in {"role_binder", "role_binder_v2", "v2_role_binder", "binder8", "legacy_binder8"}:
            return "role_binder_v2"
        raise ValueError(f"Unsupported name-memory prefix_compose_mode={mode!r}")

    def _use_role_binder_prefix(self) -> bool:
        module1_mode = str(self.module1_mode or "").strip().lower()
        return self.prefix_compose_mode == "role_binder_v2" or module1_mode in {
            "role_binder",
            "role_binder_v2",
            "v2_role_binder",
            "legacy_role_binder",
        }

    def resize_real_bank(self, num_slots: int) -> None:
        num_slots = max(1, int(num_slots))
        self.real_proto_tokens = self.real_proto_tokens.new_zeros((num_slots, 8, self.hidden_size))
        self.real_content_delta = nn.Parameter(
            self.real_content_delta.data.new_zeros((num_slots, 7, self.hidden_size))
        )
        self.slot_pref_annotation_ids = self.slot_pref_annotation_ids.new_full((num_slots, self.num_factors), -1)
        self.initialized_num_slots = num_slots

    @staticmethod
    def _is_valid_preference_annotation(text: str) -> bool:
        value = _normalize_text(text).strip().lower()
        return bool(value) and value not in {"nan", "none", "unknown", "unknown preferences"}

    def set_pref_annotation_ids_from_registry(self, registry: Dict[str, object], device=None) -> None:
        records = list(registry.get("records", []))
        num_slots = max(1, int(registry.get("num_slots", len(records) or 1)))
        target_device = device if device is not None else self.slot_pref_annotation_ids.device
        annotation_ids = torch.full((num_slots, self.num_factors), -1, dtype=torch.long, device=target_device)
        per_factor_maps = [dict() for _ in range(self.num_factors)]

        for rec in records:
            slot_id = int(rec.get("slot_id", NAME_MEMORY_UNKNOWN_SLOT_ID))
            if slot_id <= NAME_MEMORY_UNKNOWN_SLOT_ID or slot_id >= num_slots:
                continue
            factor_texts = list(rec.get("factor_texts", [""] * self.num_factors))
            for factor_idx in range(self.num_factors):
                text = factor_texts[factor_idx] if factor_idx < len(factor_texts) else ""
                if not self._is_valid_preference_annotation(text):
                    continue
                key = _normalize_text(text).lower()
                label_map = per_factor_maps[factor_idx]
                if key not in label_map:
                    label_map[key] = len(label_map)
                annotation_ids[slot_id, factor_idx] = int(label_map[key])

        self.slot_pref_annotation_ids = annotation_ids
        pseudo_ids = torch.full((self.num_pseudo_users, self.num_factors), -1, dtype=torch.long, device=target_device)
        for pseudo_idx, factor_texts in enumerate(getattr(self, "_pseudo_pref_texts", [])):
            if pseudo_idx >= self.num_pseudo_users:
                break
            for factor_idx in range(self.num_factors):
                text = factor_texts[factor_idx] if factor_idx < len(factor_texts) else ""
                if not self._is_valid_preference_annotation(text):
                    continue
                key = _normalize_text(text).lower()
                label_map = per_factor_maps[factor_idx]
                if key not in label_map:
                    label_map[key] = len(label_map)
                pseudo_ids[pseudo_idx, factor_idx] = int(label_map[key])
        self.pseudo_pref_annotation_ids = pseudo_ids

    def resize_pseudo_bank(self, num_pseudo_users: int) -> None:
        num_pseudo_users = max(0, int(num_pseudo_users))
        self.num_pseudo_users = num_pseudo_users
        self.pseudo_proto_tokens = self.pseudo_proto_tokens.new_zeros((num_pseudo_users, 8, self.hidden_size))
        self.pseudo_pref_annotation_ids = self.pseudo_pref_annotation_ids.new_full(
            (num_pseudo_users, self.num_factors),
            -1,
            dtype=torch.long,
        )
        self.pseudo_family_ids = self.pseudo_family_ids.new_zeros((num_pseudo_users,), dtype=torch.long)

    def build_pref_gate(self, concept_id: torch.Tensor) -> torch.Tensor:
        batch_size = concept_id.shape[0]
        logits = self.pref_gate_bias.unsqueeze(0).expand(batch_size, -1).clone()
        valid = concept_id.ge(0)
        if valid.any():
            boost = torch.zeros_like(logits)
            boost[valid, concept_id[valid]] = 4.0
            logits = logits + boost
        return torch.softmax(logits, dim=-1)

    def _bind_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        role_aug = tokens + self.token_role_embeddings.unsqueeze(0).to(device=tokens.device, dtype=tokens.dtype)
        return self.token_binder(role_aug)

    def _encode_text_tokens(
        self,
        texts: List[str],
        tokenizer,
        text_backbone,
        device,
        dtype,
        max_length: int = 192,
    ) -> torch.Tensor:
        if not texts:
            return torch.zeros(0, self.hidden_size, device=device, dtype=dtype)

        backbone_param = next(text_backbone.parameters())
        model_device = backbone_param.device
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        embed_only = str(os.environ.get("HOSTSWAP_NAME_MEMORY_TEXT_EMBED_ONLY", "0")).strip() == "1"
        embedding_layer = _get_text_embedding_layer(text_backbone)
        if embedding_layer is not None:
            with _gathered_params_if_needed(list(embedding_layer.parameters(recurse=False))):
                embedding_weight = getattr(embedding_layer, "weight", None)
                if embed_only or (embedding_weight is not None and getattr(embedding_weight, "ndim", 0) != 2):
                    if embedding_weight is None:
                        model_device = next(embedding_layer.parameters()).device
                    else:
                        model_device = embedding_weight.device
                    input_ids = tokenized["input_ids"].to(model_device)
                    attention_mask = tokenized["attention_mask"].to(model_device)
                    token_embeds = embedding_layer(input_ids)
                    pooled = masked_mean_pool(token_embeds, attention_mask)
                    return pooled.to(device=device, dtype=dtype)
        input_ids = tokenized["input_ids"].to(model_device)
        attention_mask = tokenized["attention_mask"].to(model_device)
        was_training = text_backbone.training
        text_backbone.eval()
        with torch.no_grad():
            outputs = text_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            pooled = masked_mean_pool(hidden_states, attention_mask)
        if was_training:
            text_backbone.train()
        return pooled.to(device=device, dtype=dtype)

    def _encode_profile_images(
        self,
        image_paths: List[str],
        vision_tower,
        mm_projector,
        image_aspect_ratio: str,
        device,
        dtype,
    ) -> torch.Tensor:
        if not image_paths:
            return torch.zeros(0, self.hidden_size, device=device, dtype=dtype)

        projector_params = list(mm_projector.parameters())
        if projector_params:
            projector_device = projector_params[0].device
            projector_dtype = projector_params[0].dtype
        else:
            projector_device = device
            projector_dtype = dtype

        vision_param = next(vision_tower.parameters())
        vision_device = vision_param.device
        vision_dtype = vision_param.dtype
        processor = vision_tower.image_processor

        prepared_inputs = []
        for path in image_paths:
            if path and osp.exists(path):
                prepared_inputs.append(_preprocess_image(path, processor, image_aspect_ratio))
                continue
            size = getattr(processor, "crop_size", {"height": 448, "width": 448})
            width = int(size.get("width", size.get("height", 448)))
            height = int(size.get("height", size.get("width", 448)))
            mean = getattr(processor, "image_mean", [0.5, 0.5, 0.5])
            blank = Image.new("RGB", (width, height), tuple(int(x * 255) for x in mean))
            prepared_inputs.append(_preprocess_pil_image(blank, processor, image_aspect_ratio))

        plain_tensors = all(torch.is_tensor(item) for item in prepared_inputs)
        same_shape = plain_tensors and all(item.shape == prepared_inputs[0].shape for item in prepared_inputs)

        was_vt_training = vision_tower.training
        was_projector_training = mm_projector.training
        vision_tower.eval()
        mm_projector.eval()
        with torch.no_grad():
            if same_shape:
                image_batch = torch.stack(prepared_inputs, dim=0).to(device=vision_device, dtype=vision_dtype)
                image_features = vision_tower(image_batch)
                if image_features.dim() == 2:
                    image_features = image_features.unsqueeze(0)
                image_features = image_features.to(device=projector_device, dtype=projector_dtype)
                image_features = mm_projector(image_features)
                pooled = image_features.mean(dim=1)
            else:
                pooled_features = []
                for item in prepared_inputs:
                    extra_kwargs = {}
                    if torch.is_tensor(item):
                        pixel_values = item
                    else:
                        pixel_values = item["pixel_values"]
                        for key, value in item.items():
                            if key == "pixel_values":
                                continue
                            if torch.is_tensor(value):
                                extra_kwargs[key] = value.unsqueeze(0) if value.dim() == 1 else value
                    if pixel_values.dim() == 3:
                        pixel_values = pixel_values.unsqueeze(0)
                    pixel_values = pixel_values.to(device=vision_device, dtype=vision_dtype)
                    image_features = vision_tower(pixel_values, **extra_kwargs)
                    if image_features.dim() == 2:
                        image_features = image_features.unsqueeze(0)
                    image_features = image_features.to(device=projector_device, dtype=projector_dtype)
                    image_features = mm_projector(image_features)
                    pooled_features.append(image_features.mean(dim=1)[0])
                pooled = torch.stack(pooled_features, dim=0)
        if was_vt_training:
            vision_tower.train()
        if was_projector_training:
            mm_projector.train()
        return pooled.to(device=device, dtype=dtype)

    def _synthesize_pseudo_image_tokens(
        self,
        real_image_tokens: torch.Tensor,
        real_desc_tokens: torch.Tensor,
        real_pref_tokens: torch.Tensor,
        pseudo_desc_tokens: torch.Tensor,
        pseudo_pref_tokens: torch.Tensor,
        pseudo_family_ids: torch.Tensor,
        image_seeds: List[int],
    ) -> torch.Tensor:
        num_pseudo = pseudo_desc_tokens.shape[0]
        if num_pseudo == 0:
            return pseudo_desc_tokens.new_zeros((0, self.hidden_size))

        positive_real_img = real_image_tokens[1:] if real_image_tokens.shape[0] > 1 else real_image_tokens[:0]
        positive_real_desc = real_desc_tokens[1:] if real_desc_tokens.shape[0] > 1 else real_desc_tokens[:0]
        positive_real_pref = real_pref_tokens[1:] if real_pref_tokens.shape[0] > 1 else real_pref_tokens[:0]
        if positive_real_img.shape[0] == 0:
            return pseudo_desc_tokens.clone()

        outputs = []
        for idx in range(num_pseudo):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(image_seeds[idx]))
            family_idx = int(pseudo_family_ids[idx].item())
            if 0 <= family_idx < self.num_factors and positive_real_pref.shape[0] > 0:
                target = F.normalize(pseudo_pref_tokens[idx, family_idx].detach().cpu(), dim=0, eps=1e-6)
                candidate_pool = F.normalize(positive_real_pref[:, family_idx, :].detach().cpu(), dim=-1, eps=1e-6)
            else:
                target = F.normalize(pseudo_desc_tokens[idx].detach().cpu(), dim=0, eps=1e-6)
                candidate_pool = F.normalize(positive_real_desc.detach().cpu(), dim=-1, eps=1e-6)

            similarity = torch.matmul(candidate_pool, target)
            topk = min(8, int(similarity.shape[0]))
            if topk > 0:
                donor_pool = torch.topk(similarity, k=topk, largest=True).indices
            else:
                donor_pool = torch.arange(positive_real_img.shape[0], dtype=torch.long)
            donor_a = int(donor_pool[int(torch.randint(donor_pool.shape[0], (1,), generator=generator).item())].item())
            donor_b = int(donor_pool[int(torch.randint(donor_pool.shape[0], (1,), generator=generator).item())].item())
            mixed = 0.65 * positive_real_img[donor_a].detach().cpu() + 0.35 * positive_real_img[donor_b].detach().cpu()
            text_hint = pseudo_desc_tokens[idx].detach().cpu()
            if 0 <= family_idx < self.num_factors:
                text_hint = text_hint + pseudo_pref_tokens[idx, family_idx].detach().cpu()
            noise = torch.randn(self.hidden_size, generator=generator, dtype=mixed.dtype)
            token = mixed + 0.10 * text_hint + 0.01 * noise
            outputs.append(token)
        return torch.stack(outputs, dim=0).to(device=real_image_tokens.device, dtype=real_image_tokens.dtype)

    def _build_manifest_payload(
        self,
        registry: Dict[str, object],
        *,
        model_identifier: str,
        tokenizer_identifier: str,
    ) -> Dict[str, object]:
        pseudo_path = self.pseudo_csv_path
        if not osp.isabs(pseudo_path):
            pseudo_path = str((_project_root() / pseudo_path).resolve())
        records_payload = []
        for rec in registry.get("records", []):
            records_payload.append(
                {
                    "slot_id": int(rec.get("slot_id", 0)),
                    "name": _normalize_text(rec.get("name", "")),
                    "profile_image_path": str(rec.get("profile_image_path", "")),
                    "description_text": _normalize_text(rec.get("description_text", "")),
                    "factor_texts": [_normalize_text(x) for x in rec.get("factor_texts", [""] * self.num_factors)],
                }
            )
        token_build_version = (
            NAME_MEMORY_ROLE_BINDER_BUILD_VERSION if self._use_role_binder_prefix() else NAME_MEMORY_TOKEN_BUILD_VERSION
        )
        return {
            "token_build_version": token_build_version,
            "module1_mode": self.module1_mode,
            "prefix_compose_mode": self.prefix_compose_mode,
            "hidden_size": int(self.hidden_size),
            "num_factors": int(self.num_factors),
            "num_pseudo_users": int(self.num_pseudo_users),
            "fold_idx": int(registry.get("fold_idx", -1)),
            "description_field": str(registry.get("description_field", "")),
            "model_identifier": str(model_identifier or ""),
            "tokenizer_identifier": str(tokenizer_identifier or ""),
            "pseudo_csv_path": pseudo_path,
            "pseudo_csv_sha256": _sha256_file(pseudo_path),
            "records": records_payload,
        }

    def _resolve_token_bank_path(self, registry: Dict[str, object], manifest_hash: str) -> Path:
        raw = str(self.token_bank_path or "").strip()
        if raw:
            path = Path(raw)
        else:
            fold_idx = int(registry.get("fold_idx", -1))
            path = Path(NAME_MEMORY_DEFAULT_TOKEN_BANK_DIR) / f"name_memory_v2_fold_{fold_idx}_{manifest_hash[:12]}.pt"
        if not path.is_absolute():
            path = _project_root() / path
        return path

    def _has_explicit_token_bank_path(self) -> bool:
        return bool(str(self.token_bank_path or "").strip())

    def _save_token_bank_artifact(self, registry: Dict[str, object], manifest: Dict[str, object], manifest_hash: str) -> None:
        path = self._resolve_token_bank_path(registry, manifest_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        slot_to_name = [str(rec.get("name", "")) for rec in registry.get("records", [])]
        payload = {
            "manifest": manifest,
            "manifest_hash": manifest_hash,
            "factor_order": list(NAME_MEMORY_FACTORS),
            "slot_to_name": slot_to_name,
            "name_to_slot": dict(registry.get("slot_by_name", {})),
            "real_proto_tokens": self.real_proto_tokens.detach().cpu(),
            "pseudo_proto_tokens": self.pseudo_proto_tokens.detach().cpu(),
            "pseudo_family_ids": self.pseudo_family_ids.detach().cpu(),
        }
        tmp_path = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
        with open(tmp_path, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)

    def _initialize_shared_pref_points(self, real_tokens: torch.Tensor) -> None:
        with torch.no_grad():
            if real_tokens.ndim != 3 or real_tokens.shape[1] < 8:
                self.shared_pref_points.zero_()
                return
            positive_real = real_tokens[1:, 3:, :] if real_tokens.shape[0] > 1 else real_tokens[:, 3:, :]
            if positive_real.numel() <= 0:
                self.shared_pref_points.zero_()
                return
            self.shared_pref_points.copy_(positive_real.mean(dim=0))

    def _try_load_token_bank_artifact(
        self,
        registry: Dict[str, object],
        manifest: Dict[str, object],
        manifest_hash: str,
        device,
        dtype,
    ) -> bool:
        path = self._resolve_token_bank_path(registry, manifest_hash)
        if not path.exists():
            return False
        payload = torch.load(path, map_location="cpu")
        factor_order = tuple(payload.get("factor_order", ()))
        if factor_order != NAME_MEMORY_FACTORS:
            return False
        slot_to_name = [str(rec.get("name", "")) for rec in registry.get("records", [])]
        if list(payload.get("slot_to_name", [])) != slot_to_name:
            return False
        real_tokens = payload.get("real_proto_tokens")
        pseudo_tokens = payload.get("pseudo_proto_tokens")
        pseudo_family_ids = payload.get("pseudo_family_ids")
        if real_tokens is None or pseudo_tokens is None or pseudo_family_ids is None:
            return False

        payload_manifest = payload.get("manifest") or {}
        strict_manifest_match = (
            str(payload.get("manifest_hash", "")) == manifest_hash and payload.get("manifest") == manifest
        )
        if not strict_manifest_match:
            if not self._has_explicit_token_bank_path():
                return False

            payload_build_version = str(payload_manifest.get("token_build_version", ""))
            current_build_version = str(manifest.get("token_build_version", ""))
            known_build_versions = {
                NAME_MEMORY_ROLE_BINDER_BUILD_VERSION,
                NAME_MEMORY_TOKEN_BUILD_VERSION,
            }
            compatible_build_version = (
                payload_build_version in known_build_versions
                and current_build_version in known_build_versions
            )
            compatible_core_shape = (
                int(payload_manifest.get("hidden_size", -1)) == int(self.hidden_size)
                and int(payload_manifest.get("num_factors", -1)) == int(self.num_factors)
                and int(payload_manifest.get("fold_idx", -999)) == int(registry.get("fold_idx", -1))
                and str(payload_manifest.get("description_field", ""))
                == str(manifest.get("description_field", ""))
                and tuple(real_tokens.shape[:2]) == (int(registry.get("num_slots", len(slot_to_name))), 8)
                and int(real_tokens.shape[-1]) == int(self.hidden_size)
                and int(pseudo_tokens.shape[-1]) == int(self.hidden_size)
            )
            if not compatible_build_version or not compatible_core_shape:
                return False
            print(
                "[name_memory] loading explicitly provided compatible token bank "
                f"despite manifest id mismatch: {path}"
            )

        self.resize_real_bank(int(real_tokens.shape[0]))
        self.resize_pseudo_bank(int(pseudo_tokens.shape[0]))
        self.real_proto_tokens = real_tokens.to(device=device, dtype=dtype)
        self.pseudo_proto_tokens = pseudo_tokens.to(device=device, dtype=dtype)
        self.pseudo_family_ids = pseudo_family_ids.to(device=device, dtype=torch.long)
        with torch.no_grad():
            self.real_content_delta.zero_()
        self._initialize_shared_pref_points(self.real_proto_tokens)
        self.initialized_num_slots = int(real_tokens.shape[0])
        self._current_manifest_hash = manifest_hash
        return True

    def _wait_for_token_bank_artifact(
        self,
        registry: Dict[str, object],
        manifest: Dict[str, object],
        manifest_hash: str,
        device,
        dtype,
        *,
        retries: int = 600,
        sleep_seconds: float = 2.0,
    ) -> bool:
        import time

        for attempt in range(retries + 1):
            if self._try_load_token_bank_artifact(
                registry,
                manifest,
                manifest_hash,
                device=device,
                dtype=dtype,
            ):
                return True
            if attempt < retries:
                time.sleep(sleep_seconds)
        return False
  ##这里开始建tokenmap
    def _build_token_bank_from_registry(
        self,
        registry: Dict[str, object],
        tokenizer,
        text_backbone,
        vision_tower,
        mm_projector,
        image_aspect_ratio: str,
        device,
        model_dtype,
        manifest: Dict[str, object],
        manifest_hash: str,
        registry_key: str,
    ) -> None:
        records = list(registry.get("records", []))
        num_slots = int(registry.get("num_slots", len(records)))
        self.resize_real_bank(num_slots)
     ##这里定义了8个位置，token怎么建立num_slots, 8，8个位置对应什么，怎么编码
        real_tokens = torch.zeros(num_slots, 8, self.hidden_size, device=device, dtype=model_dtype)
        valid_records = [rec for rec in records if int(rec.get("slot_id", 0)) > 0]
        if valid_records:
            slot_ids = torch.tensor([int(rec["slot_id"]) for rec in valid_records], device=device, dtype=torch.long)
            name_tokens = self._encode_text_tokens(
                [_format_name_prompt(rec.get("name", "")) for rec in valid_records],
                tokenizer=tokenizer,
                text_backbone=text_backbone,
                device=device,
                dtype=model_dtype,
            )
            image_tokens = self._encode_profile_images(
                [rec.get("profile_image_path", "") for rec in valid_records],
                vision_tower=vision_tower,
                mm_projector=mm_projector,
                image_aspect_ratio=image_aspect_ratio,
                device=device,
                dtype=model_dtype,
            )
            desc_tokens = self._encode_text_tokens(
                [_format_description_prompt(rec.get("description_text", "")) for rec in valid_records],
                tokenizer=tokenizer,
                text_backbone=text_backbone,
                device=device,
                dtype=model_dtype,
            )

            real_tokens.index_copy_(0, slot_ids, real_tokens.index_select(0, slot_ids))
            real_tokens[slot_ids, 0, :] = name_tokens
            real_tokens[slot_ids, 1, :] = image_tokens
            real_tokens[slot_ids, 2, :] = desc_tokens
            for factor_idx, factor_name in enumerate(NAME_MEMORY_FACTORS):
                factor_tokens = self._encode_text_tokens(
                    [
                        _format_preference_prompt(
                            factor_name,
                            rec.get("factor_texts", [""] * self.num_factors)[factor_idx],
                        )
                        for rec in valid_records
                    ],
                    tokenizer=tokenizer,
                    text_backbone=text_backbone,
                    device=device,
                    dtype=model_dtype,
                )
                real_tokens[slot_ids, 3 + factor_idx, :] = factor_tokens

        self.real_proto_tokens = real_tokens.detach()
        with torch.no_grad():
            self.real_content_delta.zero_()
        self._initialize_shared_pref_points(self.real_proto_tokens)

        pseudo_frame = load_pseudo_user_bank(self.pseudo_csv_path, expected_rows=self.num_pseudo_users)
        self.resize_pseudo_bank(len(pseudo_frame))
        if len(pseudo_frame) > 0:
            pseudo_tokens = torch.zeros(len(pseudo_frame), 8, self.hidden_size, device=device, dtype=model_dtype)
            pseudo_tokens[:, 0, :] = self._encode_text_tokens(
                [_format_name_prompt(text) for text in pseudo_frame["pseudo_name"].astype(str).tolist()],
                tokenizer=tokenizer,
                text_backbone=text_backbone,
                device=device,
                dtype=model_dtype,
            )
            pseudo_tokens[:, 2, :] = self._encode_text_tokens(
                [_format_description_prompt(text) for text in pseudo_frame["description_text"].astype(str).tolist()],
                tokenizer=tokenizer,
                text_backbone=text_backbone,
                device=device,
                dtype=model_dtype,
            )
            for factor_idx, factor_name in enumerate(NAME_MEMORY_FACTORS):
                pseudo_tokens[:, 3 + factor_idx, :] = self._encode_text_tokens(
                    [
                        _format_preference_prompt(factor_name, text)
                        for text in pseudo_frame[f"pref_{factor_name}"].astype(str).tolist()
                    ],
                    tokenizer=tokenizer,
                    text_backbone=text_backbone,
                    device=device,
                    dtype=model_dtype,
                )
            self._pseudo_pref_texts = [
                [
                    str(row[f"pref_{factor_name}"])
                    for factor_name in NAME_MEMORY_FACTORS
                ]
                for _, row in pseudo_frame.iterrows()
            ]

            pseudo_family_ids = torch.tensor(
                [concept_to_factor_id(name) for name in pseudo_frame["concept_family"].tolist()],
                device=device,
                dtype=torch.long,
            )
            pseudo_tokens[:, 1, :] = self._synthesize_pseudo_image_tokens(
                real_image_tokens=real_tokens[:, 1, :],
                real_desc_tokens=real_tokens[:, 2, :],
                real_pref_tokens=real_tokens[:, 3:, :],
                pseudo_desc_tokens=pseudo_tokens[:, 2, :],
                pseudo_pref_tokens=pseudo_tokens[:, 3:, :],
                pseudo_family_ids=pseudo_family_ids,
                image_seeds=pseudo_frame["image_seed"].tolist(),
            )
            self.pseudo_proto_tokens = pseudo_tokens.detach()
            self.pseudo_family_ids = pseudo_family_ids.detach()

        self.initialized_num_slots = num_slots
        self._registry_cache_key = registry_key
        self._current_manifest_hash = manifest_hash
        self._save_token_bank_artifact(registry, manifest, manifest_hash)

    def initialize_from_registry(
        self,
        registry: Dict[str, object],
        tokenizer,
        text_backbone,
        vision_tower,
        mm_projector,
        image_aspect_ratio: str,
        device,
        model_identifier: str = "",
        tokenizer_identifier: str = "",
        build_if_missing: bool = True,
    ) -> None:
        manifest = self._build_manifest_payload(
            registry,
            model_identifier=model_identifier,
            tokenizer_identifier=tokenizer_identifier,
        )
        manifest_hash = _stable_json_hash(manifest)
        registry_key = manifest_hash
        if self._registry_cache_key == registry_key and int(self.real_proto_tokens.shape[0]) == int(registry.get("num_slots", 1)):
            return

        model_dtype = next(self.parameters()).dtype
        dist_ready = torch.distributed.is_available() and torch.distributed.is_initialized()
        if dist_ready:
            rank = torch.distributed.get_rank()
            status_device = torch.device(device)
            if torch.distributed.get_backend() == "nccl" and status_device.type != "cuda":
                status_device = torch.device("cuda", torch.cuda.current_device())
            missing_status = torch.zeros(1, device=status_device, dtype=torch.int32)
            if rank == 0:
                if not self._try_load_token_bank_artifact(
                    registry,
                    manifest,
                    manifest_hash,
                    device=device,
                    dtype=model_dtype,
                ):
                    if build_if_missing:
                        self._build_token_bank_from_registry(
                            registry=registry,
                            tokenizer=tokenizer,
                            text_backbone=text_backbone,
                            vision_tower=vision_tower,
                            mm_projector=mm_projector,
                            image_aspect_ratio=image_aspect_ratio,
                            device=device,
                            model_dtype=model_dtype,
                            manifest=manifest,
                            manifest_hash=manifest_hash,
                            registry_key=registry_key,
                        )
                    else:
                        missing_status.fill_(1)
                else:
                    self._registry_cache_key = registry_key
            torch.distributed.broadcast(missing_status, src=0)
            if int(missing_status.item()) != 0:
                raise FileNotFoundError(
                    f"Required token-bank artifact not found at "
                    f"{self._resolve_token_bank_path(registry, manifest_hash)}"
                )
            if rank == 0:
                return

            if self._wait_for_token_bank_artifact(
                registry,
                manifest,
                manifest_hash,
                device=device,
                dtype=model_dtype,
            ):
                self._registry_cache_key = registry_key
                return
            raise FileNotFoundError(
                f"Rank 0 did not materialize token-bank artifact at "
                f"{self._resolve_token_bank_path(registry, manifest_hash)}"
            )

        if self._try_load_token_bank_artifact(registry, manifest, manifest_hash, device=device, dtype=model_dtype):
            self._registry_cache_key = registry_key
            return

        if not build_if_missing:
            raise FileNotFoundError(
                f"Required token-bank artifact not found at "
                f"{self._resolve_token_bank_path(registry, manifest_hash)}"
            )

        self._build_token_bank_from_registry(
            registry=registry,
            tokenizer=tokenizer,
            text_backbone=text_backbone,
            vision_tower=vision_tower,
            mm_projector=mm_projector,
            image_aspect_ratio=image_aspect_ratio,
            device=device,
            model_dtype=model_dtype,
            manifest=manifest,
            manifest_hash=manifest_hash,
            registry_key=registry_key,
        )

    def _balanced_pseudo_indices(self, max_per_family: int = 2) -> torch.Tensor:
        if self.num_pseudo_users <= 0:
            return self.pseudo_family_ids.new_zeros((0,), dtype=torch.long)
        indices = []
        for family_idx in range(self.num_factors):
            family_members = torch.nonzero(self.pseudo_family_ids == family_idx, as_tuple=False).flatten()
            if family_members.numel() > 0:
                indices.append(family_members[:max_per_family])
        if not indices:
            return torch.arange(self.num_pseudo_users, device=self.pseudo_family_ids.device, dtype=torch.long)
        return torch.cat(indices, dim=0)

    def select_pseudo_ids(self, user_slot_id: torch.Tensor, concept_id: torch.Tensor) -> torch.Tensor:
        batch_size = user_slot_id.shape[0]
        if self.num_pseudo_users <= 0:
            return torch.zeros(batch_size, dtype=torch.long, device=user_slot_id.device)

        device = user_slot_id.device
        cycle_value = int(self.pseudo_cycle_cursor.item())
        pseudo_ids = []
        for idx in range(batch_size):
            family = int(concept_id[idx].item())
            if 0 <= family < self.num_factors:
                family_members = torch.nonzero(self.pseudo_family_ids == family, as_tuple=False).flatten()
            else:
                family_members = self._balanced_pseudo_indices().to(device=device)
            if family_members.numel() == 0:
                family_members = torch.arange(self.num_pseudo_users, device=device, dtype=torch.long)
            stable_index = int(
                (int(user_slot_id[idx].item()) * 17 + max(family, 0) * 7 + cycle_value + idx) % int(family_members.numel())
            )
            pseudo_ids.append(family_members[stable_index])
        self.pseudo_cycle_cursor.add_(1)
        return torch.stack(pseudo_ids, dim=0)

    def get_pseudo_features(self, kind: str, factor_idx: Optional[int] = None) -> torch.Tensor:
        pseudo_tokens = self.pseudo_proto_tokens
        if self._use_role_binder_prefix():
            bound_pseudo_tokens = self._bind_tokens(pseudo_tokens)
            if kind == "img":
                if not self.use_profile_image:
                    return pseudo_tokens.new_zeros((pseudo_tokens.shape[0], self.hidden_size))
                return bound_pseudo_tokens[:, 1, :]
            if kind == "desc":
                if not self.use_description:
                    return pseudo_tokens.new_zeros((pseudo_tokens.shape[0], self.hidden_size))
                return bound_pseudo_tokens[:, 2, :]
            if kind == "pref":
                if factor_idx is None:
                    raise ValueError("factor_idx is required when kind='pref'")
                if not self.use_preference:
                    return pseudo_tokens.new_zeros((pseudo_tokens.shape[0], self.hidden_size))
                return bound_pseudo_tokens[:, 3 + int(factor_idx), :]
            raise ValueError(f"Unsupported pseudo feature kind: {kind}")
        if kind == "img":
            if not self.use_profile_image:
                return pseudo_tokens.new_zeros((pseudo_tokens.shape[0], self.hidden_size))
            return pseudo_tokens[:, 1, :]
        if kind == "desc":
            if not self.use_description:
                return pseudo_tokens.new_zeros((pseudo_tokens.shape[0], self.hidden_size))
            return pseudo_tokens[:, 2, :]
        if kind == "pref":
            if factor_idx is None:
                raise ValueError("factor_idx is required when kind='pref'")
            if not self.use_preference:
                return pseudo_tokens.new_zeros((pseudo_tokens.shape[0], self.hidden_size))
            pref_offsets = pseudo_tokens[:, 3:, :]
            shared_pref = self.shared_pref_points.to(device=pref_offsets.device, dtype=pref_offsets.dtype)
            if not self.use_factorized_preference_memory:
                pref_offsets = self._collapse_preference_tokens(pref_offsets)
                shared_pref = self._collapse_preference_tokens(
                    shared_pref.unsqueeze(0).expand(pref_offsets.shape[0], -1, -1)
                )[0]
            return shared_pref[int(factor_idx)].unsqueeze(0) + pref_offsets[:, int(factor_idx), :]
        raise ValueError(f"Unsupported pseudo feature kind: {kind}")

    def get_matching_pseudo_features(
        self,
        kind: str,
        concept_id: int,
        factor_idx: Optional[int] = None,
    ) -> torch.Tensor:
        if self.num_pseudo_users <= 0:
            return self.pseudo_proto_tokens.new_zeros((0, self.hidden_size))

        if kind == "pref":
            base = self.get_pseudo_features(kind="pref", factor_idx=factor_idx)
            target_family = int(factor_idx) if factor_idx is not None else int(concept_id)
        else:
            base = self.get_pseudo_features(kind=kind)
            target_family = int(concept_id)

        if 0 <= target_family < self.num_factors:
            indices = torch.nonzero(self.pseudo_family_ids == target_family, as_tuple=False).flatten()
        else:
            indices = self._balanced_pseudo_indices()
        if indices.numel() == 0:
            indices = torch.arange(base.shape[0], device=base.device, dtype=torch.long)
        return base.index_select(0, indices.to(device=base.device))

    def get_pseudo_residual_features(self, factor_idx: int) -> torch.Tensor:
        if self.num_pseudo_users <= 0:
            return self.pseudo_proto_tokens.new_zeros((0, self.hidden_size))
        factor_idx = int(factor_idx)
        if factor_idx < 0 or factor_idx >= self.num_factors:
            raise ValueError(f"factor_idx out of range: {factor_idx}")
        if not self.use_preference:
            return self.pseudo_proto_tokens.new_zeros((self.num_pseudo_users, self.hidden_size))
        return self.pseudo_proto_tokens[:, 3 + factor_idx, :]

    def _lookup_real_tokens(self, slot_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        proto_tokens = self.real_proto_tokens.index_select(0, slot_ids)
        delta = self.real_content_delta.index_select(0, slot_ids)
        valid_mask = slot_ids.ne(NAME_MEMORY_UNKNOWN_SLOT_ID).to(dtype=delta.dtype).view(-1, 1, 1)
        effective_tokens = proto_tokens.clone()
        effective_tokens[:, 1:, :] = effective_tokens[:, 1:, :] + valid_mask * (
            self.token_delta_scale * torch.tanh(delta)
        )
        return {
            "proto_tokens": proto_tokens,
            "effective_tokens": effective_tokens,
        }

    def _collapse_preference_tokens(self, pref_tokens: torch.Tensor) -> torch.Tensor:
        if pref_tokens.ndim != 3 or pref_tokens.shape[1] == 0:
            return pref_tokens
        coarse_pref = pref_tokens.mean(dim=1, keepdim=True)
        return coarse_pref.expand(-1, self.num_factors, -1).contiguous()

    def _build_role_binder_prefix(
        self,
        *,
        source_kind: str,
        concept_id: torch.Tensor,
        selected_pseudo_id: Optional[torch.Tensor],
        proto_tokens: torch.Tensor,
        effective_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bound_tokens = self._bind_tokens(effective_tokens)
        bound_proto_tokens = self._bind_tokens(proto_tokens)
        pref_gate = self.build_pref_gate(concept_id)
        bound_pref_tokens = bound_tokens[:, 3:, :]
        bound_proto_pref_tokens = bound_proto_tokens[:, 3:, :]

        profile_summary = 0.5 * (bound_tokens[:, 1, :] + bound_tokens[:, 2, :])
        if not (self.use_profile_image or self.use_description):
            profile_summary = bound_tokens.new_zeros((bound_tokens.shape[0], self.hidden_size))
        elif self.use_profile_image and not self.use_description:
            profile_summary = bound_tokens[:, 1, :]
        elif self.use_description and not self.use_profile_image:
            profile_summary = bound_tokens[:, 2, :]
        preference_summary = (bound_pref_tokens * pref_gate.unsqueeze(-1)).sum(dim=1)
        shared_pref_tokens = bound_pref_tokens.new_zeros(bound_pref_tokens.shape)

        return {
            "source_kind": source_kind,
            "prefix_compose_mode": self.prefix_compose_mode,
            "concept_id": concept_id,
            "selected_pseudo_id": selected_pseudo_id,
            "pref_gate": pref_gate,
            "m_id": profile_summary,
            "m_pref_pool": preference_summary,
            "profile_summary": profile_summary,
            "preference_summary": preference_summary,
            "name_token": bound_tokens[:, 0, :],
            "uid_token": bound_tokens[:, 0, :],
            "image_token": bound_tokens[:, 1, :],
            "description_token": bound_tokens[:, 2, :],
            "shared_pref_tokens": shared_pref_tokens,
            "offset_pref_tokens": bound_pref_tokens,
            "final_pref_tokens": bound_pref_tokens,
            "final_proto_pref_tokens": bound_proto_pref_tokens,
            "pref_tokens": bound_pref_tokens,
            "prefix_embeds": bound_tokens,
            "slot_img": bound_tokens[:, 1, :],
            "slot_desc": bound_tokens[:, 2, :],
            "slot_pref": bound_pref_tokens,
            "proto_img": bound_proto_tokens[:, 1, :],
            "proto_desc": bound_proto_tokens[:, 2, :],
            "proto_pref": bound_proto_pref_tokens,
            "raw_effective_tokens": effective_tokens,
            "raw_proto_tokens": proto_tokens,
            "raw_slot_img": effective_tokens[:, 1, :],
            "raw_slot_desc": effective_tokens[:, 2, :],
            "raw_slot_pref": effective_tokens[:, 3:, :],
            "raw_proto_img": proto_tokens[:, 1, :],
            "raw_proto_desc": proto_tokens[:, 2, :],
            "raw_proto_pref": proto_tokens[:, 3:, :],
        }

    def build_prefix(
        self,
        user_slot_id: torch.Tensor,
        concept_id: torch.Tensor,
        source_kind: str = "real",
    ) -> Dict[str, torch.Tensor]:
        device = user_slot_id.device
        concept_id = concept_id.to(device=device, dtype=torch.long)
        real_lookup_ids = user_slot_id.clamp(min=0, max=max(0, self.initialized_num_slots - 1))
        selected_pseudo_id = None

        if source_kind == "real":
            looked_up = self._lookup_real_tokens(real_lookup_ids)
            proto_tokens = looked_up["proto_tokens"]
            effective_tokens = looked_up["effective_tokens"]
        elif source_kind == "null":
            null_ids = torch.zeros_like(real_lookup_ids)
            looked_up = self._lookup_real_tokens(null_ids)
            proto_tokens = looked_up["proto_tokens"]
            effective_tokens = looked_up["proto_tokens"]
        elif source_kind == "pseudo":
            selected_pseudo_id = self.select_pseudo_ids(real_lookup_ids, concept_id)
            proto_tokens = self.pseudo_proto_tokens.index_select(0, selected_pseudo_id)
            effective_tokens = proto_tokens
        else:
            raise ValueError(f"Unsupported source_kind={source_kind}")

        proto_tokens = proto_tokens.clone()
        effective_tokens = effective_tokens.clone()
        if not self.use_profile_image:
            proto_tokens[:, 1, :] = 0.0
            effective_tokens[:, 1, :] = 0.0
        if not self.use_description:
            proto_tokens[:, 2, :] = 0.0
            effective_tokens[:, 2, :] = 0.0
        if not self.use_preference:
            proto_tokens[:, 3:, :] = 0.0
            effective_tokens[:, 3:, :] = 0.0

        if self._use_role_binder_prefix():
            return self._build_role_binder_prefix(
                source_kind=source_kind,
                concept_id=concept_id,
                selected_pseudo_id=selected_pseudo_id,
                proto_tokens=proto_tokens,
                effective_tokens=effective_tokens,
            )

        profile_parts = []
        if self.use_profile_image:
            profile_parts.append(effective_tokens[:, 1, :])
        if self.use_description:
            profile_parts.append(effective_tokens[:, 2, :])
        if profile_parts:
            profile_summary = torch.stack(profile_parts, dim=0).mean(dim=0)
        else:
            profile_summary = effective_tokens.new_zeros((effective_tokens.shape[0], self.hidden_size))
        shared_pref_tokens = self.shared_pref_points.to(
            device=effective_tokens.device,
            dtype=effective_tokens.dtype,
        ).unsqueeze(0).expand(effective_tokens.shape[0], -1, -1)
        if self.use_preference:
            if not self.use_factorized_preference_memory:
                effective_tokens = effective_tokens.clone()
                proto_tokens = proto_tokens.clone()
                effective_tokens[:, 3:, :] = self._collapse_preference_tokens(effective_tokens[:, 3:, :])
                proto_tokens[:, 3:, :] = self._collapse_preference_tokens(proto_tokens[:, 3:, :])
                shared_pref_tokens = self._collapse_preference_tokens(shared_pref_tokens)
        else:
            shared_pref_tokens = effective_tokens.new_zeros((effective_tokens.shape[0], self.num_factors, self.hidden_size))
        offset_pref_tokens = effective_tokens[:, 3:, :]
        proto_offset_pref_tokens = proto_tokens[:, 3:, :]
        final_pref_tokens = shared_pref_tokens + offset_pref_tokens
        final_proto_pref_tokens = shared_pref_tokens + proto_offset_pref_tokens
        pref_gate = self.build_pref_gate(concept_id)
        preference_summary = (final_pref_tokens * pref_gate.unsqueeze(-1)).sum(dim=1)
        if self.prefix_compose_mode == "sum8":
            pref_prefix_embeds = final_pref_tokens
        else:
            pref_prefix_parts = []
            for factor_idx in range(self.num_factors):
                pref_prefix_parts.append(shared_pref_tokens[:, factor_idx : factor_idx + 1, :])
                pref_prefix_parts.append(offset_pref_tokens[:, factor_idx : factor_idx + 1, :])
            pref_prefix_embeds = torch.cat(pref_prefix_parts, dim=1) if pref_prefix_parts else effective_tokens.new_zeros(
                (effective_tokens.shape[0], 0, self.hidden_size)
            )
        prefix_embeds = torch.cat([effective_tokens[:, :3, :], pref_prefix_embeds], dim=1)
        bundle = {
            "source_kind": source_kind,
            "prefix_compose_mode": self.prefix_compose_mode,
            "concept_id": concept_id,
            "selected_pseudo_id": selected_pseudo_id,
            "pref_gate": pref_gate,
            "m_pref_pool": preference_summary,
            "profile_summary": profile_summary,
            "preference_summary": preference_summary,
            "name_token": effective_tokens[:, 0, :],
            "uid_token": effective_tokens[:, 0, :],
            "image_token": effective_tokens[:, 1, :],
            "description_token": effective_tokens[:, 2, :],
            "shared_pref_tokens": shared_pref_tokens,
            "offset_pref_tokens": offset_pref_tokens,
            "final_pref_tokens": final_pref_tokens,
            "final_proto_pref_tokens": final_proto_pref_tokens,
            "pref_tokens": final_pref_tokens,
            "prefix_embeds": prefix_embeds,
            "slot_img": effective_tokens[:, 1, :],
            "slot_desc": effective_tokens[:, 2, :],
            "slot_pref": final_pref_tokens,
            "proto_img": proto_tokens[:, 1, :],
            "proto_desc": proto_tokens[:, 2, :],
            "proto_pref": final_proto_pref_tokens,
            "raw_effective_tokens": effective_tokens,
            "raw_proto_tokens": proto_tokens,
            "raw_slot_img": effective_tokens[:, 1, :],
            "raw_slot_desc": effective_tokens[:, 2, :],
            "raw_slot_pref": offset_pref_tokens,
            "raw_proto_img": proto_tokens[:, 1, :],
            "raw_proto_desc": proto_tokens[:, 2, :],
            "raw_proto_pref": proto_offset_pref_tokens,
        }
        return bundle

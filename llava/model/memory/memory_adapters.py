import weakref
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .name_memory_modules import NAME_MEMORY_FACTORS


def _normalize_task_router_mode(mode: str) -> str:
    mode = str(mode or "memory_only").strip().lower()
    if mode in {"memory", "memory_only", "learned", "learned_memory"}:
        return "memory_only"
    if mode in {"query", "query_conditioned", "learned_query"}:
        return "query"
    if mode in {"fixed", "fixed_mix", "constant", "constant_mix"}:
        return "fixed"
    raise ValueError(f"Unsupported HMoE task_router_mode={mode!r}")


def _normalize_inner_router_mode(mode: str) -> str:
    mode = str(mode or "learned").strip().lower()
    if mode in {"learned", "softmax", "trainable"}:
        return "learned"
    if mode in {"fixed", "constant", "uniform", "fixed_uniform", "none"}:
        return "fixed"
    raise ValueError(f"Unsupported HMoE inner router mode={mode!r}")


def _parse_fixed_router_weights(weights, count: int):
    count = max(1, int(count))
    if weights is None:
        values = []
    elif isinstance(weights, str):
        values = [item.strip() for item in weights.split(",") if item.strip()]
    else:
        values = list(weights)
    if not values:
        return [1.0 / count for _ in range(count)]
    parsed = [max(0.0, float(value)) for value in values]
    if len(parsed) < count:
        parsed.extend([0.0 for _ in range(count - len(parsed))])
    parsed = parsed[:count]
    total = sum(parsed)
    if total <= 0.0:
        return [1.0 / count for _ in range(count)]
    return [value / total for value in parsed]


def _normalize_hier_lora_context_mode(mode: str) -> str:
    mode = str(mode or "none").strip().lower()
    if mode in {"none", "off", "disabled", "false", "0"}:
        return "none"
    if mode in {"bottleneck_bias", "rank_bias", "memory_bias", "context_bias"}:
        return "bottleneck_bias"
    raise ValueError(f"Unsupported Hierarchical MoE LoRA context mode={mode!r}")


def _normalize_pref_context_mode(mode: str) -> str:
    mode = str(mode or "all_factors").strip().lower()
    if mode in {"all", "all_factor", "all_factors", "factor", "factor_tokens"}:
        return "all_factors"
    if mode in {"selected", "concept", "concept_selected", "selected_factor", "pref_gate"}:
        return "concept_selected"
    if mode in {"hybrid", "selected_plus_factor", "factor_plus_selected"}:
        return "hybrid"
    if mode in {"legacy", "legacy_hmoe", "legacy_global", "global", "uid_concept"}:
        return "legacy_hmoe"
    raise ValueError(f"Unsupported Hierarchical MoE preference context mode={mode!r}")


def _parse_target_modules(target_modules):
    if target_modules is None:
        return None
    if isinstance(target_modules, str):
        values = [item.strip() for item in target_modules.split(",") if item.strip()]
        return tuple(values) if values else None
    values = [str(item).strip() for item in target_modules if str(item).strip()]
    return tuple(values) if values else None


####在 LLM 顶层插一组 adapter，用 Module 1 给的 memory context 去动态调 decoder
class _ResidualAdapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck_dim: int):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_dim, bias=False)
        self.up = nn.Linear(bottleneck_dim, hidden_size, bias=False)
        self.act = nn.GELU()
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        low_rank = self.down(hidden_states)
        if context is not None:
            low_rank = low_rank + context.unsqueeze(1)
        delta = self.up(self.act(low_rank))
        return torch.tanh(self.scale) * delta


class SharedPreferenceLoRAExpert(_ResidualAdapter):
    pass


class UserSpecificPreferenceAdapter(_ResidualAdapter):
    pass


class ProfileImageAdapter(_ResidualAdapter):
    pass


class ProfileDescriptionAdapter(_ResidualAdapter):
    pass


class _LoRAExpert(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, lora_alpha: float = None):
        super().__init__()
        self.rank = max(1, int(rank))
        self.scaling = float(self.rank if lora_alpha is None else lora_alpha) / float(self.rank)
        self.down = nn.Linear(in_features, self.rank, bias=False)
        self.up = nn.Linear(self.rank, out_features, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        low_rank = self.down(x)
        if context is not None:
            while context.ndim < low_rank.ndim:
                context = context.unsqueeze(1)
            low_rank = F.gelu(low_rank + context)
        return self.up(low_rank) * self.scaling


class _HierarchicalMoELoraSite(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        pref_expert_count: int,
        profile_expert_count: int,
        pref_rank: int,
        profile_rank: int,
        user_pref_rank: int = None,
        enable_user_pref_adapter: bool = True,
        lora_alpha: float = None,
    ):
        super().__init__()
        self.pref_experts = nn.ModuleList(
            [_LoRAExpert(in_features, out_features, pref_rank, lora_alpha=lora_alpha) for _ in range(pref_expert_count)]
        )
        self.profile_experts = nn.ModuleList(
            [_LoRAExpert(in_features, out_features, profile_rank, lora_alpha=lora_alpha) for _ in range(profile_expert_count)]
        )
        if enable_user_pref_adapter:
            self.user_pref_adapter = _LoRAExpert(
                in_features,
                out_features,
                pref_rank if user_pref_rank is None else user_pref_rank,
                lora_alpha=lora_alpha,
            )
        else:
            self.user_pref_adapter = None

    @staticmethod
    def _mix_experts(
        x: torch.Tensor,
        experts: nn.ModuleList,
        gate: torch.Tensor,
        contexts: torch.Tensor = None,
    ) -> torch.Tensor:
        mixed = x.new_zeros((*x.shape[:-1], experts[0].up.out_features))
        for idx, expert in enumerate(experts):
            weight = gate[:, idx]
            while weight.ndim < mixed.ndim:
                weight = weight.unsqueeze(-1)
            context = None if contexts is None else contexts[:, idx, :]
            mixed = mixed + weight * expert(x, context=context)
        return mixed

    def forward(
        self,
        x: torch.Tensor,
        *,
        pref_route: torch.Tensor,
        profile_gate: torch.Tensor,
        task_gate: torch.Tensor,
        pref_contexts: torch.Tensor = None,
        user_pref_context: torch.Tensor = None,
        profile_contexts: torch.Tensor = None,
    ) -> torch.Tensor:
        pref_branch = self._mix_experts(x, self.pref_experts, pref_route, contexts=pref_contexts)
        if self.user_pref_adapter is not None:
            pref_branch = pref_branch + self.user_pref_adapter(x, context=user_pref_context)
        profile_branch = self._mix_experts(x, self.profile_experts, profile_gate, contexts=profile_contexts)
        pref_weight = task_gate[:, 0]
        profile_weight = task_gate[:, 1]
        while pref_weight.ndim < pref_branch.ndim:
            pref_weight = pref_weight.unsqueeze(-1)
            profile_weight = profile_weight.unsqueeze(-1)
        return pref_weight * pref_branch + profile_weight * profile_branch


class _ContextProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.GELU(),
            nn.Linear(out_dim, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _flatten_pref_tokens(pref_tokens: torch.Tensor, num_factors: int) -> torch.Tensor:
    if pref_tokens.ndim != 3:
        raise ValueError("preference tokens must be a [B, F, H] tensor")
    batch_size = pref_tokens.shape[0]
    hidden_size = pref_tokens.shape[-1]
    trimmed = pref_tokens[:, :num_factors, :]
    if trimmed.shape[1] < num_factors:
        pad = trimmed.new_zeros((batch_size, num_factors - trimmed.shape[1], hidden_size))
        trimmed = torch.cat([trimmed, pad], dim=1)
    return trimmed.reshape(batch_size, num_factors * hidden_size)


class HierarchicalMoELoraAdapterStack(nn.Module):
    """Name-memory conditioned LoRA-MoE injected into target decoder Linear modules.

    This is the implementation that matches the yellow-box Hierarchical MoE design:
    five shared preference LoRA experts and two profile LoRA experts are mixed by
    token-conditioned routers, then a task router mixes preference/profile branches.
    """

    DEFAULT_TARGET_MODULES = ("q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 8,
        num_factors: int = 5,
        num_shared_pref_experts: int = 5,
        num_profile_experts: int = 2,
        shared_pref_rank: int = 8,
        user_pref_rank: int = None,
        user_profile_rank: int = 8,
        enable_user_pref_adapter: bool = True,
        task_router_supervision: str = "none",
        task_router_mode: str = "query",
        task_router_fixed_pref_weight: float = 0.6,
        task_router_target_confidence: float = 0.8,
        pref_router_mode: str = "learned",
        pref_router_fixed_weights=None,
        pref_router_supervision: str = "none",
        pref_router_target_confidence: float = 0.9,
        pref_context_mode: str = "all_factors",
        profile_router_mode: str = "learned",
        profile_router_fixed_weights=None,
        context_mode: str = "none",
        target_modules=None,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_factors = int(num_factors)
        self.num_shared_pref_experts = max(1, min(int(num_shared_pref_experts), int(num_factors)))
        self.num_profile_experts = max(1, int(num_profile_experts))
        self.shared_pref_rank = max(1, int(shared_pref_rank))
        self.user_pref_rank = self.shared_pref_rank if user_pref_rank is None else max(1, int(user_pref_rank))
        self.user_profile_rank = max(1, int(user_profile_rank))
        self.enable_user_pref_adapter = bool(enable_user_pref_adapter)
        self.task_router_supervision = str(task_router_supervision or "none").strip().lower()
        self.task_router_mode = _normalize_task_router_mode(task_router_mode)
        self.task_router_fixed_pref_weight = float(max(0.0, min(1.0, task_router_fixed_pref_weight)))
        self.task_router_target_confidence = float(max(0.5, min(1.0, task_router_target_confidence)))
        self.pref_router_mode = _normalize_inner_router_mode(pref_router_mode)
        self.pref_router_supervision = str(pref_router_supervision or "none").strip().lower()
        self.pref_router_target_confidence = float(
            max(1.0 / self.num_shared_pref_experts, min(1.0, pref_router_target_confidence))
        )
        self.pref_context_mode = _normalize_pref_context_mode(pref_context_mode)
        self.profile_router_mode = _normalize_inner_router_mode(profile_router_mode)
        self.pref_router_fixed_weights = _parse_fixed_router_weights(
            pref_router_fixed_weights,
            self.num_shared_pref_experts,
        )
        self.profile_router_fixed_weights = _parse_fixed_router_weights(
            profile_router_fixed_weights,
            self.num_profile_experts,
        )
        self.context_mode = _normalize_hier_lora_context_mode(context_mode)
        self.target_modules = tuple(_parse_target_modules(target_modules) or self.DEFAULT_TARGET_MODULES)
        self.gate_hidden_dim = max(32, min(128, hidden_size // 64))

        self.image_encoder = _ContextProjector(hidden_size, self.gate_hidden_dim)
        self.description_encoder = _ContextProjector(hidden_size, self.gate_hidden_dim)
        self.uid_encoder = _ContextProjector(hidden_size, self.gate_hidden_dim)
        self.query_encoder = _ContextProjector(hidden_size, self.gate_hidden_dim)
        self.profile_summary_encoder = _ContextProjector(hidden_size, self.gate_hidden_dim)
        self.pref_signature_encoder = _ContextProjector(hidden_size * self.num_factors, self.gate_hidden_dim)
        self.concept_embedding = nn.Embedding(self.num_factors + 1, self.gate_hidden_dim)

        task_router_input_dim = self.gate_hidden_dim * (3 if self.task_router_mode == "query" else 2)
        self.task_router = nn.Sequential(
            nn.Linear(task_router_input_dim, self.gate_hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.gate_hidden_dim, 2, bias=False),
        )
        self.pref_router = nn.Sequential(
            nn.Linear(hidden_size, self.gate_hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.gate_hidden_dim, 1, bias=False),
        )
        if self.enable_user_pref_adapter:
            self.user_pref_context_projector = _ContextProjector(self.gate_hidden_dim * 3, self.user_pref_rank)
        else:
            self.user_pref_context_projector = None
        self.profile_router = nn.Sequential(
            nn.Linear(self.gate_hidden_dim * 3, self.gate_hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.gate_hidden_dim, self.num_profile_experts, bias=False),
        )
        if self.context_mode == "bottleneck_bias":
            context_input_dim = hidden_size * 2
            self.pref_context_projectors = nn.ModuleList(
                [_ContextProjector(context_input_dim, self.shared_pref_rank) for _ in range(self.num_shared_pref_experts)]
            )
            self.pref_global_context_projector = _ContextProjector(self.gate_hidden_dim * 3, self.shared_pref_rank)
            self.profile_context_projectors = nn.ModuleList(
                [_ContextProjector(context_input_dim, self.user_profile_rank) for _ in range(self.num_profile_experts)]
            )
        else:
            self.pref_context_projectors = nn.ModuleList()
            self.pref_global_context_projector = None
            self.profile_context_projectors = nn.ModuleList()
        self.sites = nn.ModuleDict()

        self._context = None
        self._task_router_loss = None
        self._task_router_correct = 0.0
        self._task_router_entropy = 0.0
        self._task_router_count = 0
        self._pref_router_loss = None
        self._pref_router_correct = 0.0
        self._pref_router_entropy = 0.0
        self._pref_router_count = 0
        self._debug_route_sums = {}
        self._debug_route_count = 0
        self._debug_route_by_concept = {}

    def reset_route_debug_stats(self) -> None:
        self._debug_route_sums = {}
        self._debug_route_count = 0
        self._debug_route_by_concept = {}
        self._task_router_loss = None
        self._task_router_correct = 0.0
        self._task_router_entropy = 0.0
        self._task_router_count = 0
        self._pref_router_loss = None
        self._pref_router_correct = 0.0
        self._pref_router_entropy = 0.0
        self._pref_router_count = 0

    def _accumulate_route_debug(self, key: str, value: torch.Tensor) -> None:
        self._debug_route_sums[key] = self._debug_route_sums.get(key, 0.0) + float(
            value.detach().float().mean().item()
        )

    def _record_route_debug(self, concept_index: torch.Tensor, task_gate: torch.Tensor, profile_gate: torch.Tensor, pref_route: torch.Tensor) -> None:
        with torch.no_grad():
            self._debug_route_count += 1
            self._accumulate_route_debug("task_pref", task_gate[:, 0])
            self._accumulate_route_debug("task_profile", task_gate[:, 1])
            self._accumulate_route_debug("task_raw_pref", task_gate[:, 0])
            self._accumulate_route_debug("task_raw_profile", task_gate[:, 1])
            self._accumulate_route_debug("profile_image", profile_gate[:, 0])
            if profile_gate.shape[-1] > 1:
                self._accumulate_route_debug("profile_description", profile_gate[:, 1])
            for idx in range(self.num_shared_pref_experts):
                self._accumulate_route_debug(f"pref_factor_{idx}", pref_route[:, idx])
            pref_choice = pref_route.argmax(dim=-1)
            for row_idx in range(pref_route.shape[0]):
                concept_idx = int(concept_index[row_idx].item())
                concept_key = f"c{concept_idx}"
                stats = self._debug_route_by_concept.setdefault(
                    concept_key,
                    {
                        "count": 0,
                        "task_pref": 0.0,
                        "task_profile": 0.0,
                        "profile_image": 0.0,
                        "profile_description": 0.0,
                        "pref_choice": [0.0 for _ in range(self.num_shared_pref_experts)],
                        "pref_route": [0.0 for _ in range(self.num_shared_pref_experts)],
                    },
                )
                stats["count"] += 1
                stats["task_pref"] += float(task_gate[row_idx, 0].item())
                stats["task_profile"] += float(task_gate[row_idx, 1].item())
                stats["profile_image"] += float(profile_gate[row_idx, 0].item())
                stats["profile_description"] += float(profile_gate[row_idx, min(1, profile_gate.shape[-1] - 1)].item())
                chosen = int(pref_choice[row_idx].item())
                stats["pref_choice"][chosen] += 1.0
                for gate_idx in range(self.num_shared_pref_experts):
                    stats["pref_route"][gate_idx] += float(pref_route[row_idx, gate_idx].item())

    def route_debug_summary(self) -> dict:
        count = max(1, int(self._debug_route_count))
        summary = {key: value / count for key, value in self._debug_route_sums.items()}
        summary["site_calls"] = float(self._debug_route_count)
        summary["shared_pref_expert_count"] = float(self.num_shared_pref_experts)
        if self._task_router_count > 0:
            summary["task_router_acc"] = float(self._task_router_correct / self._task_router_count)
            summary["task_router_entropy"] = float(self._task_router_entropy / self._task_router_count)
        else:
            summary["task_router_acc"] = 0.0
            summary["task_router_entropy"] = 0.0
        if self._pref_router_count > 0:
            summary["pref_router_acc"] = float(self._pref_router_correct / self._pref_router_count)
            summary["pref_router_entropy"] = float(self._pref_router_entropy / self._pref_router_count)
        else:
            summary["pref_router_acc"] = 0.0
            summary["pref_router_entropy"] = 0.0
        concept_chunks = []
        for concept_key in sorted(self._debug_route_by_concept.keys()):
            stats = self._debug_route_by_concept[concept_key]
            concept_count = max(1, int(stats["count"]))
            concept_idx = int(concept_key[1:])
            concept_label = NAME_MEMORY_FACTORS[concept_idx] if 0 <= concept_idx < len(NAME_MEMORY_FACTORS) else "neutral"
            pref_route_mean = [value / concept_count for value in stats["pref_route"]]
            pref_choice_counts = stats["pref_choice"]
            chosen_idx = max(range(len(pref_choice_counts)), key=lambda idx: pref_choice_counts[idx])
            concept_chunks.append(
                f"{concept_label}:E{chosen_idx}"
                f"/task={stats['task_pref']/concept_count:.2f}|{stats['task_profile']/concept_count:.2f}"
                f"/prof={stats['profile_image']/concept_count:.2f}|{stats['profile_description']/concept_count:.2f}"
                f"/route=[{','.join(f'{value:.2f}' for value in pref_route_mean)}]"
            )
        summary["by_concept"] = " ; ".join(concept_chunks)
        return summary

    def _build_shared_pref_inputs(self, pref_tokens: torch.Tensor):
        if pref_tokens.ndim != 3 or pref_tokens.shape[1] <= 0:
            return [
                pref_tokens.new_zeros((pref_tokens.shape[0], pref_tokens.shape[-1]))
                for _ in range(self.num_shared_pref_experts)
            ]
        pref_tokens = pref_tokens[:, : self.num_factors, :]
        if self.num_shared_pref_experts >= pref_tokens.shape[1]:
            return [pref_tokens[:, idx, :] for idx in range(self.num_shared_pref_experts)]
        chunks = torch.chunk(pref_tokens, self.num_shared_pref_experts, dim=1)
        return [chunk.mean(dim=1) for chunk in chunks]

    def _build_bottleneck_contexts(
        self,
        *,
        shared_pref_inputs,
        selected_pref_token: torch.Tensor = None,
        image_token: torch.Tensor,
        description_token: torch.Tensor,
        query_summary: torch.Tensor,
        pref_global_context: torch.Tensor = None,
    ):
        if self.context_mode != "bottleneck_bias":
            return None, None
        pref_contexts = []
        context_pref_inputs = shared_pref_inputs
        if selected_pref_token is not None:
            if self.pref_context_mode == "concept_selected":
                context_pref_inputs = [selected_pref_token for _ in range(self.num_shared_pref_experts)]
            elif self.pref_context_mode == "hybrid":
                context_pref_inputs = [
                    0.5 * (shared_pref_inputs[idx] + selected_pref_token)
                    for idx in range(self.num_shared_pref_experts)
                ]
        for idx, projector in enumerate(self.pref_context_projectors):
            pref_context = projector(torch.cat([context_pref_inputs[idx], query_summary], dim=-1))
            if pref_global_context is not None:
                pref_context = pref_context + pref_global_context
            pref_contexts.append(pref_context)
        profile_inputs = [image_token, description_token]
        profile_contexts = []
        for idx, projector in enumerate(self.profile_context_projectors):
            token_idx = min(idx, len(profile_inputs) - 1)
            profile_contexts.append(projector(torch.cat([profile_inputs[token_idx], query_summary], dim=-1)))
        return torch.stack(pref_contexts, dim=1), torch.stack(profile_contexts, dim=1)

    def set_context(self, context: dict) -> None:
        debug_routes = str(os.environ.get("NAME_MEMORY_DEBUG_LOSS", "")).strip().lower() in {"1", "true", "yes", "on"}
        self.reset_route_debug_stats()
        concept_id = context["concept_id"].long()
        neutral_index = self.num_factors
        concept_index = concept_id.clone().clamp(min=-1, max=self.num_factors - 1)
        concept_index = torch.where(concept_index.ge(0), concept_index, torch.full_like(concept_index, neutral_index))

        pref_tokens = context["final_pref_tokens"] if "final_pref_tokens" in context else context["pref_tokens"]
        shared_pref_inputs = self._build_shared_pref_inputs(pref_tokens)
        pref_router_tokens = torch.stack(shared_pref_inputs, dim=1)
        selected_pref_token = context.get("preference_summary")
        if selected_pref_token is None and "pref_gate" in context:
            pref_gate = context["pref_gate"].to(device=pref_tokens.device, dtype=pref_tokens.dtype)
            selected_pref_token = (pref_tokens[:, : self.num_factors, :] * pref_gate[:, : self.num_factors].unsqueeze(-1)).sum(dim=1)
        pref_signature = self.pref_signature_encoder(_flatten_pref_tokens(pref_tokens, self.num_factors))
        uid_summary = self.uid_encoder(context["uid_token"])
        concept_embedding = self.concept_embedding(concept_index)
        profile_summary = self.profile_summary_encoder(context["profile_summary"])
        image_summary = self.image_encoder(context["image_token"])
        description_summary = self.description_encoder(context["description_token"])
        query_summary = self.query_encoder(
            context.get("query_summary", context["profile_summary"].new_zeros(context["profile_summary"].shape))
        )
        raw_query_summary = context.get(
            "query_summary",
            context["profile_summary"].new_zeros(context["profile_summary"].shape),
        )
        pref_global_context = None
        if self.context_mode == "bottleneck_bias" and self.pref_context_mode == "legacy_hmoe":
            pref_global_context = self.pref_global_context_projector(
                torch.cat([pref_signature, concept_embedding, uid_summary], dim=-1)
            )
        user_pref_context = None
        if self.user_pref_context_projector is not None:
            user_pref_context = self.user_pref_context_projector(
                torch.cat([pref_signature, concept_embedding, uid_summary], dim=-1)
            )
        pref_contexts, profile_contexts = self._build_bottleneck_contexts(
            shared_pref_inputs=shared_pref_inputs,
            selected_pref_token=selected_pref_token,
            image_token=context["image_token"],
            description_token=context["description_token"],
            query_summary=raw_query_summary,
            pref_global_context=pref_global_context,
        )

        if self.pref_router_mode == "fixed":
            pref_route = pref_router_tokens.new_tensor(self.pref_router_fixed_weights).view(1, -1)
            pref_route = pref_route.expand(pref_router_tokens.shape[0], -1)
        else:
            pref_route_logits = self.pref_router(pref_router_tokens).squeeze(-1)
            pref_route = torch.softmax(pref_route_logits, dim=-1)
            self._record_pref_router_supervision(
                pref_route_logits,
                pref_route,
                concept_id,
                task_labels=context.get("router_task_label"),
            )

        profile_router_input = torch.cat([image_summary, description_summary, profile_summary], dim=-1)
        if self.profile_router_mode == "fixed":
            profile_gate = profile_summary.new_tensor(self.profile_router_fixed_weights).view(1, -1)
            profile_gate = profile_gate.expand(profile_summary.shape[0], -1)
        else:
            profile_gate = torch.softmax(self.profile_router(profile_router_input), dim=-1)

        if self.task_router_mode == "fixed":
            fixed_pref = profile_summary.new_full((profile_summary.shape[0], 1), self.task_router_fixed_pref_weight)
            task_gate = torch.cat([fixed_pref, 1.0 - fixed_pref], dim=-1)
            task_logits = None
            self._record_fixed_task_router_stats(task_gate, labels=context.get("router_task_label"))
        else:
            task_router_parts = [profile_summary, pref_signature]
            if self.task_router_mode == "query":
                task_router_parts.append(query_summary)
            task_logits = self.task_router(torch.cat(task_router_parts, dim=-1))
            task_gate = torch.softmax(task_logits, dim=-1)
            self._record_task_router_supervision(task_logits, task_gate, labels=context.get("router_task_label"))

        self._context = {
            "pref_route": pref_route,
            "profile_gate": profile_gate,
            "task_gate": task_gate,
            "pref_contexts": pref_contexts,
            "user_pref_context": user_pref_context,
            "profile_contexts": profile_contexts,
            "concept_index": concept_index,
            "_debug_routes": debug_routes,
        }

    def clear_context(self) -> None:
        self._context = None

    def _record_task_router_supervision(self, task_logits: torch.Tensor, task_gate: torch.Tensor, labels=None) -> None:
        if labels is None or self.task_router_supervision != "category":
            return
        labels = labels.to(device=task_logits.device, dtype=torch.long)
        valid = labels.ge(0)
        if not valid.any():
            return
        valid_logits = task_logits[valid]
        valid_labels = labels[valid]
        if self.task_router_target_confidence < 1.0:
            num_classes = valid_logits.shape[-1]
            off_value = (1.0 - self.task_router_target_confidence) / max(1, num_classes - 1)
            soft_targets = valid_logits.new_full(valid_logits.shape, off_value)
            soft_targets.scatter_(1, valid_labels.view(-1, 1), self.task_router_target_confidence)
            self._task_router_loss = -(soft_targets * F.log_softmax(valid_logits, dim=-1)).sum(dim=-1).mean()
        else:
            self._task_router_loss = F.cross_entropy(valid_logits, valid_labels, reduction="mean")
        with torch.no_grad():
            valid_gate = task_gate[valid]
            pred = valid_logits.argmax(dim=-1)
            self._task_router_correct += float(pred.eq(valid_labels).float().sum().item())
            entropy = -(valid_gate * valid_gate.clamp_min(1e-8).log()).sum(dim=-1).mean()
            self._task_router_entropy += float(entropy.item()) * float(valid_labels.shape[0])
            self._task_router_count += int(valid_labels.shape[0])

    def _record_pref_router_supervision(
        self,
        pref_logits: torch.Tensor,
        pref_route: torch.Tensor,
        concept_id=None,
        task_labels=None,
    ) -> None:
        if concept_id is None or self.pref_router_supervision not in {"concept", "factor", "concept_id"}:
            return
        labels = concept_id.to(device=pref_logits.device, dtype=torch.long)
        valid = labels.ge(0) & labels.lt(self.num_shared_pref_experts)
        if task_labels is not None:
            task_labels = task_labels.to(device=pref_logits.device, dtype=torch.long)
            valid = valid & task_labels.eq(0)
        if not valid.any():
            return
        valid_logits = pref_logits[valid]
        valid_labels = labels[valid]
        if self.pref_router_target_confidence < 1.0:
            num_classes = valid_logits.shape[-1]
            off_value = (1.0 - self.pref_router_target_confidence) / max(1, num_classes - 1)
            soft_targets = valid_logits.new_full(valid_logits.shape, off_value)
            soft_targets.scatter_(1, valid_labels.view(-1, 1), self.pref_router_target_confidence)
            self._pref_router_loss = -(soft_targets * F.log_softmax(valid_logits, dim=-1)).sum(dim=-1).mean()
        else:
            self._pref_router_loss = F.cross_entropy(valid_logits, valid_labels, reduction="mean")
        with torch.no_grad():
            valid_route = pref_route[valid]
            pred = valid_logits.argmax(dim=-1)
            self._pref_router_correct += float(pred.eq(valid_labels).float().sum().item())
            entropy = -(valid_route * valid_route.clamp_min(1e-8).log()).sum(dim=-1).mean()
            self._pref_router_entropy += float(entropy.item()) * float(valid_labels.shape[0])
            self._pref_router_count += int(valid_labels.shape[0])

    def _record_fixed_task_router_stats(self, task_gate: torch.Tensor, labels=None) -> None:
        if labels is None:
            return
        labels = labels.to(device=task_gate.device, dtype=torch.long)
        valid = labels.ge(0)
        if not valid.any():
            return
        with torch.no_grad():
            valid_gate = task_gate[valid]
            pred = valid_gate.argmax(dim=-1)
            valid_labels = labels[valid]
            self._task_router_correct += float(pred.eq(valid_labels).float().sum().item())
            entropy = -(valid_gate * valid_gate.clamp_min(1e-8).log()).sum(dim=-1).mean()
            self._task_router_entropy += float(entropy.item()) * float(valid_labels.shape[0])
            self._task_router_count += int(valid_labels.shape[0])

    def get_runtime_aux_losses(self) -> dict:
        device = next(self.parameters()).device
        zero = torch.zeros((), device=device)
        loss_task_router = self._task_router_loss if self._task_router_loss is not None else zero
        loss_pref_router = self._pref_router_loss if self._pref_router_loss is not None else zero
        if self._task_router_count > 0:
            acc = torch.tensor(self._task_router_correct / self._task_router_count, device=device)
            entropy = torch.tensor(self._task_router_entropy / self._task_router_count, device=device)
        else:
            acc = zero
            entropy = zero
        if self._pref_router_count > 0:
            pref_acc = torch.tensor(self._pref_router_correct / self._pref_router_count, device=device)
            pref_entropy = torch.tensor(self._pref_router_entropy / self._pref_router_count, device=device)
        else:
            pref_acc = zero
            pref_entropy = zero
        return {
            "loss_task_router": loss_task_router,
            "task_router_acc": acc,
            "task_router_entropy": entropy,
            "loss_pref_router": loss_pref_router,
            "pref_router_acc": pref_acc,
            "pref_router_entropy": pref_entropy,
        }

    @staticmethod
    def _site_key(layer_index: int, parent_name: str, module_name: str) -> str:
        return f"layer{layer_index}_{parent_name}_{module_name}".replace(".", "_")

    def compute_delta(self, site_key: str, x: torch.Tensor) -> torch.Tensor:
        if self._context is None:
            out_features = self.sites[site_key].pref_experts[0].up.out_features
            return x.new_zeros((*x.shape[:-1], out_features))
        site = self.sites[site_key]
        delta = site(
            x,
            pref_route=self._context["pref_route"],
            profile_gate=self._context["profile_gate"],
            task_gate=self._context["task_gate"],
            pref_contexts=self._context.get("pref_contexts"),
            user_pref_context=self._context.get("user_pref_context"),
            profile_contexts=self._context.get("profile_contexts"),
        )
        if self._context.get("_debug_routes"):
            self._record_route_debug(
                concept_index=self._context["concept_index"],
                task_gate=self._context["task_gate"],
                profile_gate=self._context["profile_gate"],
                pref_route=self._context["pref_route"],
            )
        return delta

    def attach(self, decoder_layers) -> None:
        target_layers = list(decoder_layers)[-self.num_layers :]
        for layer_index, layer in enumerate(target_layers):
            for parent_name in ("self_attn", "mlp"):
                parent = getattr(layer, parent_name, None)
                if parent is None:
                    continue
                for module_name in self.target_modules:
                    original = getattr(parent, module_name, None)
                    if not isinstance(original, nn.Linear):
                        continue
                    site_key = self._site_key(layer_index, parent_name, module_name)
                    self.sites[site_key] = _HierarchicalMoELoraSite(
                        in_features=original.in_features,
                        out_features=original.out_features,
                        pref_expert_count=self.num_shared_pref_experts,
                        profile_expert_count=self.num_profile_experts,
                        pref_rank=self.shared_pref_rank,
                        user_pref_rank=self.user_pref_rank,
                        profile_rank=self.user_profile_rank,
                        enable_user_pref_adapter=self.enable_user_pref_adapter,
                    )
                    setattr(parent, module_name, MemoryInjectedLinearMoELora(original, self, site_key))


class _HierarchicalAdapterSite(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_shared_pref_experts: int,
        shared_pref_rank: int,
        user_pref_rank: int,
        user_profile_rank: int,
        pref_profile_mix: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_shared_pref_experts = num_shared_pref_experts
        self.shared_pref_rank = shared_pref_rank
        self.user_pref_rank = user_pref_rank
        self.user_profile_rank = user_profile_rank
        self.pref_profile_mix = float(max(0.0, min(1.0, pref_profile_mix)))
        self.profile_mix = 1.0 - self.pref_profile_mix
        gate_hidden_dim = max(32, min(128, hidden_size // 64))
        self.gate_hidden_dim = gate_hidden_dim
        ##pref_router
        self.pref_router = nn.Sequential(
            nn.Linear(gate_hidden_dim * 3, gate_hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, num_shared_pref_experts, bias=False),
        )
        self.user_pref_context = _ContextProjector(gate_hidden_dim * 3, user_pref_rank)
        self.user_pref_adapter = UserSpecificPreferenceAdapter(hidden_size, user_pref_rank)
        ##profile_router
        self.profile_router = nn.Sequential(
            nn.Linear(gate_hidden_dim * 3, gate_hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 2, bias=False),
        )
        self.image_context = _ContextProjector(gate_hidden_dim * 3, user_profile_rank)
        self.description_context = _ContextProjector(gate_hidden_dim * 3, user_profile_rank)
        self.image_adapter = ProfileImageAdapter(hidden_size, user_profile_rank)
        self.description_adapter = ProfileDescriptionAdapter(hidden_size, user_profile_rank)
        ##profile_router
        self.task_router = nn.Sequential(
            nn.Linear(gate_hidden_dim * 3, gate_hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 2, bias=False),
        )
        ###forward里偏好分支，外貌/描述分支做混合
    def forward(
        self,
        hidden_states: torch.Tensor,
        context: dict,
        shared_pref_experts: nn.ModuleList,
        site_kind: str = "",
        site_index: int = -1,
    ) -> torch.Tensor:
        if context is None:
            return hidden_states.new_zeros(hidden_states.shape)

        uid_gate = context["uid_gate"]
        image_gate = context["image_gate"]
        description_gate = context["description_gate"]
        factor_contexts = context["factor_contexts"]
        concept_embedding = context["concept_embedding"]
        concept_index = context["concept_index"]
        profile_summary = context["profile_summary"]
        pref_signature = context["pref_signature"]

        pref_router_input = torch.cat([pref_signature, concept_embedding, uid_gate], dim=-1)
        pref_route = torch.softmax(self.pref_router(pref_router_input), dim=-1)

        delta_shared = hidden_states.new_zeros(hidden_states.shape)
        for idx, adapter in enumerate(shared_pref_experts):
            delta_factor = adapter(hidden_states, factor_contexts[:, idx, :])
            delta_shared = delta_shared + pref_route[:, idx].view(-1, 1, 1) * delta_factor

        user_pref_context = self.user_pref_context(torch.cat([pref_signature, concept_embedding, uid_gate], dim=-1))
        delta_user_pref = self.user_pref_adapter(hidden_states, user_pref_context)
        pref_branch = delta_shared + delta_user_pref

        profile_router_input = torch.cat([image_gate, description_gate, uid_gate], dim=-1)
        profile_gate = torch.softmax(self.profile_router(profile_router_input), dim=-1)
        image_context = self.image_context(torch.cat([image_gate, profile_summary, uid_gate], dim=-1))
        desc_context = self.description_context(torch.cat([description_gate, profile_summary, uid_gate], dim=-1))
        profile_branch = (
            profile_gate[:, 0].view(-1, 1, 1) * self.image_adapter(hidden_states, image_context)
            + profile_gate[:, 1].view(-1, 1, 1) * self.description_adapter(hidden_states, desc_context)
        )

        task_router_input = torch.cat([profile_summary, pref_signature, uid_gate], dim=-1)
        task_gate_raw = torch.softmax(self.task_router(task_router_input), dim=-1)
        mix_prior = task_gate_raw.new_tensor([self.pref_profile_mix, self.profile_mix]).view(1, 2)
        task_gate = task_gate_raw * mix_prior
        task_gate = task_gate / task_gate.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        debug_collector = context.get("_debug_collector")
        if debug_collector is not None:
            debug_collector(
                site_kind=site_kind,
                site_index=site_index,
                concept_index=concept_index,
                pref_route=pref_route,
                profile_gate=profile_gate,
                task_gate_raw=task_gate_raw,
                task_gate=task_gate,
            )
        return (
            task_gate[:, 0].view(-1, 1, 1) * pref_branch
            + task_gate[:, 1].view(-1, 1, 1) * profile_branch
        )

   ##定义一堆 encoder，把 Module 1 给的 token/summaries 编成 gate 用的小向量
class Module2AdapterStack(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 8,
        num_factors: int = 5,
        num_shared_pref_experts: int = 5,
        shared_pref_rank: int = 64,
        user_pref_rank: int = 64,
        user_profile_rank: int = 32,
        pref_profile_mix: float = 0.7,
        routing_mode: str = "hierarchical",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_factors = num_factors
        self.num_shared_pref_experts = max(1, min(int(num_shared_pref_experts), int(num_factors)))
        self.shared_pref_rank = shared_pref_rank
        self.user_pref_rank = user_pref_rank
        self.user_profile_rank = user_profile_rank
        self.pref_profile_mix = float(max(0.0, min(1.0, pref_profile_mix)))
        self.routing_mode = str(routing_mode or "hierarchical").strip().lower()
        self.gate_hidden_dim = max(32, min(128, hidden_size // 64))

        self.uid_encoder = _ContextProjector(hidden_size, self.gate_hidden_dim)
        self.image_encoder = _ContextProjector(hidden_size, self.gate_hidden_dim)
        self.description_encoder = _ContextProjector(hidden_size, self.gate_hidden_dim)
        self.profile_summary_encoder = _ContextProjector(hidden_size, self.gate_hidden_dim)
        self.pref_signature_encoder = _ContextProjector(hidden_size * self.num_factors, self.gate_hidden_dim)
        self.concept_embedding = nn.Embedding(num_factors + 1, self.gate_hidden_dim)
        self.shared_factor_generators = nn.ModuleList(
            [
                _ContextProjector(hidden_size, shared_pref_rank)
                for _ in range(self.num_shared_pref_experts)
            ]
        )
        self.shared_pref_experts = nn.ModuleList(
            [SharedPreferenceLoRAExpert(hidden_size, shared_pref_rank) for _ in range(self.num_shared_pref_experts)]
        )
        self.attn_sites = nn.ModuleList(
            [
                _HierarchicalAdapterSite(
                    hidden_size=hidden_size,
                    num_shared_pref_experts=self.num_shared_pref_experts,
                    shared_pref_rank=shared_pref_rank,
                    user_pref_rank=user_pref_rank,
                    user_profile_rank=user_profile_rank,
                    pref_profile_mix=self.pref_profile_mix,
                )
                for _ in range(num_layers)
            ]
        )
        self.mlp_sites = nn.ModuleList(
            [
                _HierarchicalAdapterSite(
                    hidden_size=hidden_size,
                    num_shared_pref_experts=self.num_shared_pref_experts,
                    shared_pref_rank=shared_pref_rank,
                    user_pref_rank=user_pref_rank,
                    user_profile_rank=user_profile_rank,
                    pref_profile_mix=self.pref_profile_mix,
                )
                for _ in range(num_layers)
            ]
        )
        self._context = None
        self._debug_route_sums = {}
        self._debug_route_count = 0
        self._debug_route_by_concept = {}

    def reset_route_debug_stats(self) -> None:
        self._debug_route_sums = {}
        self._debug_route_count = 0
        self._debug_route_by_concept = {}

    def _accumulate_route_debug(self, key: str, value: torch.Tensor) -> None:
        self._debug_route_sums[key] = self._debug_route_sums.get(key, 0.0) + float(
            value.detach().float().mean().item()
        )

    def _record_route_debug(
        self,
        *,
        site_kind: str,
        site_index: int,
        concept_index: torch.Tensor,
        pref_route: torch.Tensor,
        profile_gate: torch.Tensor,
        task_gate_raw: torch.Tensor,
        task_gate: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            self._debug_route_count += 1
            self._accumulate_route_debug("task_pref", task_gate[:, 0])
            self._accumulate_route_debug("task_profile", task_gate[:, 1])
            self._accumulate_route_debug("task_raw_pref", task_gate_raw[:, 0])
            self._accumulate_route_debug("task_raw_profile", task_gate_raw[:, 1])
            self._accumulate_route_debug("profile_image", profile_gate[:, 0])
            self._accumulate_route_debug("profile_description", profile_gate[:, 1])
            for idx in range(self.num_shared_pref_experts):
                self._accumulate_route_debug(f"pref_factor_{idx}", pref_route[:, idx])
            pref_choice = pref_route.argmax(dim=-1)
            for row_idx in range(pref_route.shape[0]):
                concept_idx = int(concept_index[row_idx].item())
                concept_key = f"c{concept_idx}"
                stats = self._debug_route_by_concept.setdefault(
                    concept_key,
                    {
                        "count": 0,
                        "task_pref": 0.0,
                        "task_profile": 0.0,
                        "profile_image": 0.0,
                        "profile_description": 0.0,
                        "pref_choice": [0.0 for _ in range(self.num_shared_pref_experts)],
                        "pref_route": [0.0 for _ in range(self.num_shared_pref_experts)],
                    },
                )
                stats["count"] += 1
                stats["task_pref"] += float(task_gate[row_idx, 0].item())
                stats["task_profile"] += float(task_gate[row_idx, 1].item())
                stats["profile_image"] += float(profile_gate[row_idx, 0].item())
                stats["profile_description"] += float(profile_gate[row_idx, 1].item())
                chosen = int(pref_choice[row_idx].item())
                stats["pref_choice"][chosen] += 1.0
                for gate_idx in range(self.num_shared_pref_experts):
                    stats["pref_route"][gate_idx] += float(pref_route[row_idx, gate_idx].item())

    def route_debug_summary(self) -> dict:
        count = max(1, int(self._debug_route_count))
        summary = {key: value / count for key, value in self._debug_route_sums.items()}
        summary["site_calls"] = float(self._debug_route_count)
        summary["pref_profile_mix_prior"] = float(self.pref_profile_mix)
        summary["profile_mix_prior"] = float(1.0 - self.pref_profile_mix)
        summary["shared_pref_expert_count"] = float(self.num_shared_pref_experts)
        concept_chunks = []
        for concept_key in sorted(self._debug_route_by_concept.keys()):
            stats = self._debug_route_by_concept[concept_key]
            concept_count = max(1, int(stats["count"]))
            concept_idx = int(concept_key[1:])
            if 0 <= concept_idx < len(NAME_MEMORY_FACTORS):
                concept_label = NAME_MEMORY_FACTORS[concept_idx]
            else:
                concept_label = "neutral"
            pref_route_mean = [value / concept_count for value in stats["pref_route"]]
            pref_choice_counts = stats["pref_choice"]
            chosen_idx = max(range(len(pref_choice_counts)), key=lambda idx: pref_choice_counts[idx])
            concept_chunks.append(
                f"{concept_label}:E{chosen_idx}"
                f"/task={stats['task_pref']/concept_count:.2f}|{stats['task_profile']/concept_count:.2f}"
                f"/prof={stats['profile_image']/concept_count:.2f}|{stats['profile_description']/concept_count:.2f}"
                f"/route=[{','.join(f'{value:.2f}' for value in pref_route_mean)}]"
            )
        summary["by_concept"] = " ; ".join(concept_chunks)
        return summary

    def get_runtime_aux_losses(self) -> dict:
        zero = next(self.parameters()).new_zeros(())
        return {
            "loss_task_router": zero,
            "task_router_acc": zero,
            "task_router_entropy": zero,
        }

    def _build_shared_pref_inputs(self, pref_tokens: torch.Tensor):
        if pref_tokens.ndim != 3 or pref_tokens.shape[1] <= 0:
            return [pref_tokens.new_zeros((pref_tokens.shape[0], pref_tokens.shape[-1])) for _ in range(self.num_shared_pref_experts)]
        pref_tokens = pref_tokens[:, : self.num_factors, :]
        if self.num_shared_pref_experts >= pref_tokens.shape[1]:
            return [pref_tokens[:, idx, :] for idx in range(self.num_shared_pref_experts)]
        chunks = torch.chunk(pref_tokens, self.num_shared_pref_experts, dim=1)
        return [chunk.mean(dim=1) for chunk in chunks]

    def set_context(self, context: dict) -> None:
        debug_routes = str(os.environ.get("NAME_MEMORY_DEBUG_LOSS", "")).strip().lower() in {"1", "true", "yes", "on"}
        if debug_routes:
            self.reset_route_debug_stats()
        concept_id = context["concept_id"].long()
        neutral_index = self.num_factors
        concept_index = concept_id.clone()
        concept_index = concept_index.clamp(min=-1, max=self.num_factors - 1)
        concept_index = torch.where(concept_index.ge(0), concept_index, torch.full_like(concept_index, neutral_index))
        if "final_pref_tokens" in context:
            pref_tokens = context["final_pref_tokens"]
        else:
            pref_tokens = context["pref_tokens"]
        pref_signature = self.pref_signature_encoder(_flatten_pref_tokens(pref_tokens, self.num_factors))
        factor_contexts = []
        shared_pref_inputs = self._build_shared_pref_inputs(pref_tokens)
        if self.routing_mode == "coarse":
            coarse_pref = torch.stack(shared_pref_inputs, dim=1).mean(dim=1)
            for generator in self.shared_factor_generators:
                factor_contexts.append(generator(coarse_pref))
        else:
            for idx, generator in enumerate(self.shared_factor_generators):
                factor_contexts.append(generator(shared_pref_inputs[idx]))
        self._context = {
            "uid_gate": self.uid_encoder(context["uid_token"]),
            "image_gate": self.image_encoder(context["image_token"]),
            "description_gate": self.description_encoder(context["description_token"]),
            "profile_summary": self.profile_summary_encoder(context["profile_summary"]),
            "pref_signature": pref_signature,
            "factor_contexts": torch.stack(factor_contexts, dim=1),
            "concept_embedding": self.concept_embedding(concept_index),
            "concept_index": concept_index,
        }
        if debug_routes:
            self._context["_debug_collector"] = self._record_route_debug

    def clear_context(self) -> None:
        self._context = None

    def current_context(self):
        return self._context

    def compute_delta(self, site_kind: str, site_index: int, hidden_states: torch.Tensor) -> torch.Tensor:
        if site_kind == "attn":
            return self.attn_sites[site_index](
                hidden_states,
                self._context,
                self.shared_pref_experts,
                site_kind=site_kind,
                site_index=site_index,
            )
        return self.mlp_sites[site_index](
            hidden_states,
            self._context,
            self.shared_pref_experts,
            site_kind=site_kind,
            site_index=site_index,
        )
  ##怎么把它挂上去
    def attach(self, decoder_layers) -> None:
        target_layers = list(decoder_layers)[-self.num_layers :]
        for site_index, layer in enumerate(target_layers):
            layer.self_attn = MemoryInjectedSelfAttention(layer.self_attn, self, site_index)
            layer.mlp = MemoryInjectedMLP(layer.mlp, self, site_index)


class HMoEModule2AdapterStack(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 8,
        num_factors: int = 5,
        num_shared_pref_experts: int = 5,
        num_profile_experts: int = 2,
        shared_pref_rank: int = 64,
        user_profile_rank: int = 32,
        routing_mode: str = "hierarchical",
        task_router_supervision: str = "none",
        task_router_mode: str = "memory_only",
        task_router_fixed_pref_weight: float = 0.6,
        task_router_target_confidence: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = int(num_layers)
        self.num_factors = int(num_factors)
        self.num_shared_pref_experts = max(1, min(int(num_shared_pref_experts), int(num_factors)))
        self.num_profile_experts = max(1, int(num_profile_experts))
        self.shared_pref_rank = int(shared_pref_rank)
        self.user_profile_rank = int(user_profile_rank)
        self.routing_mode = str(routing_mode or "hierarchical").strip().lower()
        self.task_router_supervision = str(task_router_supervision or "none").strip().lower()
        self.task_router_mode = self._normalize_task_router_mode(task_router_mode)
        self.task_router_fixed_pref_weight = float(max(0.0, min(1.0, task_router_fixed_pref_weight)))
        self.task_router_target_confidence = float(max(0.5, min(1.0, task_router_target_confidence)))
        self.gate_hidden_dim = max(32, min(128, hidden_size // 64))

        self.image_encoder = _ContextProjector(hidden_size, self.gate_hidden_dim)
        self.description_encoder = _ContextProjector(hidden_size, self.gate_hidden_dim)
        self.query_encoder = _ContextProjector(hidden_size, self.gate_hidden_dim)
        self.profile_summary_encoder = _ContextProjector(hidden_size, self.gate_hidden_dim)
        self.pref_signature_encoder = _ContextProjector(hidden_size * self.num_factors, self.gate_hidden_dim)

        task_router_input_dim = self.gate_hidden_dim * (3 if self.task_router_mode == "query" else 2)
        self.task_router = nn.Sequential(
            nn.Linear(task_router_input_dim, self.gate_hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.gate_hidden_dim, 2, bias=False),
        )
        self.pref_router = nn.Sequential(
            nn.Linear(hidden_size, self.gate_hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.gate_hidden_dim, 1, bias=False),
        )
        self.profile_router = nn.Sequential(
            nn.Linear(self.gate_hidden_dim * 3, self.gate_hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.gate_hidden_dim, self.num_profile_experts, bias=False),
        )

        self.shared_factor_generators = nn.ModuleList(
            [_ContextProjector(hidden_size, self.shared_pref_rank) for _ in range(self.num_shared_pref_experts)]
        )
        self.shared_pref_experts = nn.ModuleList(
            [SharedPreferenceLoRAExpert(hidden_size, self.shared_pref_rank) for _ in range(self.num_shared_pref_experts)]
        )

        self.profile_context_projectors = nn.ModuleList(
            [_ContextProjector(self.gate_hidden_dim * 2, self.user_profile_rank) for _ in range(self.num_profile_experts)]
        )
        self.profile_experts = nn.ModuleList(
            [ProfileImageAdapter(hidden_size, self.user_profile_rank)]
            + [ProfileDescriptionAdapter(hidden_size, self.user_profile_rank) for _ in range(self.num_profile_experts - 1)]
        )

        self._context = None
        self._task_router_loss = None
        self._task_router_correct = 0.0
        self._task_router_entropy = 0.0
        self._task_router_count = 0
        self._debug_route_sums = {}
        self._debug_route_count = 0
        self._debug_route_by_concept = {}

    @staticmethod
    def _normalize_task_router_mode(mode: str) -> str:
        mode = str(mode or "memory_only").strip().lower()
        if mode in {"memory", "memory_only", "learned", "learned_memory"}:
            return "memory_only"
        if mode in {"query", "query_conditioned", "learned_query"}:
            return "query"
        if mode in {"fixed", "fixed_mix", "constant", "constant_mix"}:
            return "fixed"
        raise ValueError(f"Unsupported HMoE task_router_mode={mode!r}")

    def reset_route_debug_stats(self) -> None:
        self._debug_route_sums = {}
        self._debug_route_count = 0
        self._debug_route_by_concept = {}
        self._task_router_loss = None
        self._task_router_correct = 0.0
        self._task_router_entropy = 0.0
        self._task_router_count = 0

    def _accumulate_route_debug(self, key: str, value: torch.Tensor) -> None:
        self._debug_route_sums[key] = self._debug_route_sums.get(key, 0.0) + float(
            value.detach().float().mean().item()
        )

    def _record_route_debug(
        self,
        *,
        concept_index: torch.Tensor,
        task_gate: torch.Tensor,
        profile_gate: torch.Tensor,
        pref_route: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            self._debug_route_count += 1
            self._accumulate_route_debug("task_pref", task_gate[:, 0])
            self._accumulate_route_debug("task_profile", task_gate[:, 1])
            self._accumulate_route_debug("task_raw_pref", task_gate[:, 0])
            self._accumulate_route_debug("task_raw_profile", task_gate[:, 1])
            self._accumulate_route_debug("profile_image", profile_gate[:, 0])
            if profile_gate.shape[-1] > 1:
                self._accumulate_route_debug("profile_description", profile_gate[:, 1])
            for idx in range(self.num_shared_pref_experts):
                self._accumulate_route_debug(f"pref_factor_{idx}", pref_route[:, idx])
            pref_choice = pref_route.argmax(dim=-1)
            for row_idx in range(pref_route.shape[0]):
                concept_idx = int(concept_index[row_idx].item())
                concept_key = f"c{concept_idx}"
                stats = self._debug_route_by_concept.setdefault(
                    concept_key,
                    {
                        "count": 0,
                        "task_pref": 0.0,
                        "task_profile": 0.0,
                        "profile_image": 0.0,
                        "profile_description": 0.0,
                        "pref_choice": [0.0 for _ in range(self.num_shared_pref_experts)],
                        "pref_route": [0.0 for _ in range(self.num_shared_pref_experts)],
                    },
                )
                stats["count"] += 1
                stats["task_pref"] += float(task_gate[row_idx, 0].item())
                stats["task_profile"] += float(task_gate[row_idx, 1].item())
                stats["profile_image"] += float(profile_gate[row_idx, 0].item())
                stats["profile_description"] += float(profile_gate[row_idx, min(1, profile_gate.shape[-1] - 1)].item())
                chosen = int(pref_choice[row_idx].item())
                stats["pref_choice"][chosen] += 1.0
                for gate_idx in range(self.num_shared_pref_experts):
                    stats["pref_route"][gate_idx] += float(pref_route[row_idx, gate_idx].item())

    def route_debug_summary(self) -> dict:
        count = max(1, int(self._debug_route_count))
        summary = {key: value / count for key, value in self._debug_route_sums.items()}
        summary["site_calls"] = float(self._debug_route_count)
        summary["shared_pref_expert_count"] = float(self.num_shared_pref_experts)
        if self._task_router_count > 0:
            summary["task_router_acc"] = float(self._task_router_correct / self._task_router_count)
            summary["task_router_entropy"] = float(self._task_router_entropy / self._task_router_count)
        else:
            summary["task_router_acc"] = 0.0
            summary["task_router_entropy"] = 0.0
        concept_chunks = []
        for concept_key in sorted(self._debug_route_by_concept.keys()):
            stats = self._debug_route_by_concept[concept_key]
            concept_count = max(1, int(stats["count"]))
            concept_idx = int(concept_key[1:])
            if 0 <= concept_idx < len(NAME_MEMORY_FACTORS):
                concept_label = NAME_MEMORY_FACTORS[concept_idx]
            else:
                concept_label = "neutral"
            pref_route_mean = [value / concept_count for value in stats["pref_route"]]
            pref_choice_counts = stats["pref_choice"]
            chosen_idx = max(range(len(pref_choice_counts)), key=lambda idx: pref_choice_counts[idx])
            concept_chunks.append(
                f"{concept_label}:E{chosen_idx}"
                f"/task={stats['task_pref']/concept_count:.2f}|{stats['task_profile']/concept_count:.2f}"
                f"/prof={stats['profile_image']/concept_count:.2f}|{stats['profile_description']/concept_count:.2f}"
                f"/route=[{','.join(f'{value:.2f}' for value in pref_route_mean)}]"
            )
        summary["by_concept"] = " ; ".join(concept_chunks)
        return summary

    def _build_shared_pref_inputs(self, pref_tokens: torch.Tensor):
        if pref_tokens.ndim != 3 or pref_tokens.shape[1] <= 0:
            return [
                pref_tokens.new_zeros((pref_tokens.shape[0], pref_tokens.shape[-1]))
                for _ in range(self.num_shared_pref_experts)
            ]
        pref_tokens = pref_tokens[:, : self.num_factors, :]
        if self.num_shared_pref_experts >= pref_tokens.shape[1]:
            return [pref_tokens[:, idx, :] for idx in range(self.num_shared_pref_experts)]
        chunks = torch.chunk(pref_tokens, self.num_shared_pref_experts, dim=1)
        return [chunk.mean(dim=1) for chunk in chunks]

    def set_context(self, context: dict) -> None:
        debug_routes = str(os.environ.get("NAME_MEMORY_DEBUG_LOSS", "")).strip().lower() in {"1", "true", "yes", "on"}
        self.reset_route_debug_stats()
        concept_id = context["concept_id"].long()
        neutral_index = self.num_factors
        concept_index = concept_id.clone()
        concept_index = concept_index.clamp(min=-1, max=self.num_factors - 1)
        concept_index = torch.where(concept_index.ge(0), concept_index, torch.full_like(concept_index, neutral_index))

        if "final_pref_tokens" in context:
            pref_tokens = context["final_pref_tokens"]
        else:
            pref_tokens = context["pref_tokens"]
        pref_signature = self.pref_signature_encoder(_flatten_pref_tokens(pref_tokens, self.num_factors))
        factor_contexts = []
        shared_pref_inputs = self._build_shared_pref_inputs(pref_tokens)
        if self.routing_mode == "coarse":
            coarse_pref = torch.stack(shared_pref_inputs, dim=1).mean(dim=1)
            for generator in self.shared_factor_generators:
                factor_contexts.append(generator(coarse_pref))
        else:
            for idx, generator in enumerate(self.shared_factor_generators):
                factor_contexts.append(generator(shared_pref_inputs[idx]))

        self._context = {
            "profile_summary": self.profile_summary_encoder(context["profile_summary"]),
            "pref_signature": pref_signature,
            "image_summary": self.image_encoder(context["image_token"]),
            "description_summary": self.description_encoder(context["description_token"]),
            "query_summary": self.query_encoder(
                context.get("query_summary", context["profile_summary"].new_zeros(context["profile_summary"].shape))
            ),
            "pref_expert_contexts": torch.stack(factor_contexts, dim=1),
            "pref_router_tokens": torch.stack(shared_pref_inputs, dim=1),
            "concept_index": concept_index,
            "router_task_label": context.get("router_task_label"),
            "_debug_routes": debug_routes,
        }
        if self.task_router_mode == "fixed":
            fixed_pref = self._context["profile_summary"].new_full(
                (self._context["profile_summary"].shape[0], 1),
                self.task_router_fixed_pref_weight,
            )
            task_gate = torch.cat([fixed_pref, 1.0 - fixed_pref], dim=-1)
            self._record_fixed_task_router_stats(task_gate)
        else:
            task_router_parts = [self._context["profile_summary"], self._context["pref_signature"]]
            if self.task_router_mode == "query":
                task_router_parts.append(self._context["query_summary"])
            task_router_input = torch.cat(task_router_parts, dim=-1)
            task_logits = self.task_router(task_router_input)
            task_gate = torch.softmax(task_logits, dim=-1)
            self._record_task_router_supervision(task_logits, task_gate)
        # Keep the task-router CE trainable, but avoid carrying the router graph
        # into each checkpointed Module2 site via the cached gate tensor.
        self._context["task_gate"] = task_gate.detach()

    def clear_context(self) -> None:
        self._context = None

    def current_context(self):
        return self._context

    def _record_task_router_supervision(self, task_logits: torch.Tensor, task_gate: torch.Tensor) -> None:
        if self._context is None:
            return
        labels = self._context.get("router_task_label")
        if labels is None or self.task_router_supervision != "category":
            return
        labels = labels.to(device=task_logits.device, dtype=torch.long)
        valid = labels.ge(0)
        if not valid.any():
            return
        valid_logits = task_logits[valid]
        valid_labels = labels[valid]
        if self.task_router_target_confidence < 1.0:
            num_classes = valid_logits.shape[-1]
            off_value = (1.0 - self.task_router_target_confidence) / max(1, num_classes - 1)
            soft_targets = valid_logits.new_full(valid_logits.shape, off_value)
            soft_targets.scatter_(1, valid_labels.view(-1, 1), self.task_router_target_confidence)
            self._task_router_loss = -(soft_targets * F.log_softmax(valid_logits, dim=-1)).sum(dim=-1).mean()
        else:
            self._task_router_loss = F.cross_entropy(valid_logits, valid_labels, reduction="mean")
        with torch.no_grad():
            valid_gate = task_gate[valid]
            pred = valid_logits.argmax(dim=-1)
            self._task_router_correct += float(pred.eq(valid_labels).float().sum().item())
            entropy = -(valid_gate * valid_gate.clamp_min(1e-8).log()).sum(dim=-1).mean()
            self._task_router_entropy += float(entropy.item()) * float(valid_labels.shape[0])
            self._task_router_count += int(valid_labels.shape[0])

    def _record_fixed_task_router_stats(self, task_gate: torch.Tensor) -> None:
        if self._context is None:
            return
        labels = self._context.get("router_task_label")
        if labels is None:
            return
        labels = labels.to(device=task_gate.device, dtype=torch.long)
        valid = labels.ge(0)
        if not valid.any():
            return
        with torch.no_grad():
            valid_gate = task_gate[valid]
            pred = valid_gate.argmax(dim=-1)
            valid_labels = labels[valid]
            self._task_router_correct += float(pred.eq(valid_labels).float().sum().item())
            entropy = -(valid_gate * valid_gate.clamp_min(1e-8).log()).sum(dim=-1).mean()
            self._task_router_entropy += float(entropy.item()) * float(valid_labels.shape[0])
            self._task_router_count += int(valid_labels.shape[0])

    def get_runtime_aux_losses(self) -> dict:
        device = None
        if self._context is not None:
            device = self._context["profile_summary"].device
        else:
            device = next(self.parameters()).device
        zero = torch.zeros((), device=device)
        loss_task_router = self._task_router_loss if self._task_router_loss is not None else zero
        if self._task_router_count > 0:
            acc = torch.tensor(self._task_router_correct / self._task_router_count, device=device)
            entropy = torch.tensor(self._task_router_entropy / self._task_router_count, device=device)
        else:
            acc = zero
            entropy = zero
        return {
            "loss_task_router": loss_task_router,
            "task_router_acc": acc,
            "task_router_entropy": entropy,
        }

    def compute_delta(self, site_kind: str, site_index: int, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._context is None:
            return hidden_states.new_zeros(hidden_states.shape)

        task_gate = self._context["task_gate"]

        pref_route_logits = self.pref_router(self._context["pref_router_tokens"]).squeeze(-1)
        pref_route = torch.softmax(pref_route_logits, dim=-1)
        pref_branch = hidden_states.new_zeros(hidden_states.shape)
        for idx, expert in enumerate(self.shared_pref_experts):
            pref_delta = expert(hidden_states, self._context["pref_expert_contexts"][:, idx, :])
            pref_branch = pref_branch + pref_route[:, idx].view(-1, 1, 1) * pref_delta

        profile_router_input = torch.cat(
            [self._context["image_summary"], self._context["description_summary"], self._context["profile_summary"]],
            dim=-1,
        )
        profile_gate = torch.softmax(self.profile_router(profile_router_input), dim=-1)
        profile_context_inputs = [
            torch.cat([self._context["image_summary"], self._context["profile_summary"]], dim=-1),
            torch.cat([self._context["description_summary"], self._context["profile_summary"]], dim=-1),
        ]
        profile_branch = hidden_states.new_zeros(hidden_states.shape)
        for idx, expert in enumerate(self.profile_experts):
            context_index = min(idx, len(profile_context_inputs) - 1)
            profile_context = self.profile_context_projectors[idx](profile_context_inputs[context_index])
            profile_delta = expert(hidden_states, profile_context)
            profile_branch = profile_branch + profile_gate[:, idx].view(-1, 1, 1) * profile_delta

        if self._context.get("_debug_routes"):
            self._record_route_debug(
                concept_index=self._context["concept_index"],
                task_gate=task_gate,
                profile_gate=profile_gate,
                pref_route=pref_route,
            )
        return (
            task_gate[:, 0].view(-1, 1, 1) * pref_branch
            + task_gate[:, 1].view(-1, 1, 1) * profile_branch
        )

    def attach(self, decoder_layers) -> None:
        target_layers = list(decoder_layers)[-self.num_layers :]
        for site_index, layer in enumerate(target_layers):
            layer.self_attn = MemoryInjectedSelfAttention(layer.self_attn, self, site_index)
            layer.mlp = MemoryInjectedMLP(layer.mlp, self, site_index)


class MemoryInjectedSelfAttention(nn.Module):
    def __init__(self, original_module: nn.Module, adapter_stack: Module2AdapterStack, site_index: int):
        super().__init__()
        self.original_module = original_module
        object.__setattr__(self, "_adapter_stack_ref", weakref.proxy(adapter_stack))
        self.site_index = site_index

    def forward(self, *args, **kwargs):
        outputs = self.original_module(*args, **kwargs)
        attn_hidden = outputs[0]
        delta = self._adapter_stack_ref.compute_delta("attn", self.site_index, attn_hidden)
        attn_hidden = attn_hidden + delta
        return (attn_hidden,) + tuple(outputs[1:])


class MemoryInjectedMLP(nn.Module):
    def __init__(self, original_module: nn.Module, adapter_stack: Module2AdapterStack, site_index: int):
        super().__init__()
        self.original_module = original_module
        object.__setattr__(self, "_adapter_stack_ref", weakref.proxy(adapter_stack))
        self.site_index = site_index

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.original_module(hidden_states)
        delta = self._adapter_stack_ref.compute_delta("mlp", self.site_index, hidden_states)
        return hidden_states + delta


class MemoryInjectedLinearMoELora(nn.Module):
    def __init__(self, original_module: nn.Linear, adapter_stack: HierarchicalMoELoraAdapterStack, site_key: str):
        super().__init__()
        self.original_module = original_module
        object.__setattr__(self, "_adapter_stack_ref", weakref.proxy(adapter_stack))
        self.site_key = str(site_key)

    @property
    def in_features(self) -> int:
        return self.original_module.in_features

    @property
    def out_features(self) -> int:
        return self.original_module.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_module(x) + self._adapter_stack_ref.compute_delta(self.site_key, x)

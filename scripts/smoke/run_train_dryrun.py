import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.model.prefmllm import PrefMLLMWrapper


class FakeSubModule(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class FakeBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = FakeSubModule(hidden_size)
        self.mlp = FakeSubModule(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attn = self.self_attn.o_proj(torch.tanh(self.self_attn.q_proj(hidden_states) + self.self_attn.v_proj(hidden_states)))
        gate = torch.sigmoid(self.mlp.gate_proj(hidden_states))
        mlp = self.mlp.down_proj(F.gelu(self.mlp.up_proj(hidden_states)) * gate)
        return hidden_states + attn + mlp


class FakeLlavaBackbone(nn.Module):
    def __init__(self, hidden_size: int = 16, vocab_size: int = 32):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            eos_token_id=2,
            pad_token_id=0,
            bos_token_id=1,
            use_cache=False,
        )
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([FakeBlock(hidden_size), FakeBlock(hidden_size)])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_model(self):
        return self

    def get_input_embeddings(self):
        return self.embed_tokens

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
    ):
        inputs_embeds = self.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=False,
        return_dict=True,
        **kwargs,
    ):
        hidden_states = inputs_embeds
        all_hidden = [hidden_states]
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            all_hidden.append(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=tuple(all_hidden) if output_hidden_states else None,
        )


def main() -> None:
    torch.manual_seed(7)
    os.makedirs("./outputs/smoke", exist_ok=True)

    base = FakeLlavaBackbone()
    model = PrefMLLMWrapper(
        base,
        num_slots=3,
        num_pseudo_users=0,
        top_layers=1,
        shared_pref_rank=2,
        user_pref_rank=2,
        user_profile_rank=2,
        module2_arch="hierarchical_moe_lora",
        task_router_mode="query",
        task_router_supervision="none",
        pref_router_mode="learned",
        profile_router_mode="learned",
        enable_counterfactual=False,
        pref_residual_loss_mode="density_focal",
    )
    model.train()
    model.name_memory_module.resize_real_bank(3)
    with torch.no_grad():
        tokens = torch.randn(3, 8, base.config.hidden_size) * 0.05
        tokens[0].zero_()
        model.name_memory_module.real_proto_tokens.copy_(tokens)
        model.name_memory_module.shared_pref_points.copy_(tokens[1:, 3:, :].mean(dim=0))
        model.name_memory_module.slot_pref_annotation_ids = torch.tensor(
            [
                [-1, -1, -1, -1, -1],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ],
            dtype=torch.long,
        )

    input_ids = torch.tensor([[4, 5, 6, 7, 8], [4, 9, 10, 11, 8]], dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :2] = -100
    output = model(
        input_ids=input_ids,
        labels=labels,
        user_slot_id=torch.tensor([1, 2], dtype=torch.long),
        concept_id=torch.tensor([2, 1], dtype=torch.long),
        router_task_label=torch.tensor([0, 1], dtype=torch.long),
        return_dict=True,
    )
    if output.loss is None or not torch.isfinite(output.loss):
        raise SystemExit("dry-run loss is not finite")
    output.loss.backward()

    loss_items = {
        key: float(value.detach().cpu())
        for key, value in output.name_memory_losses.items()
        if torch.is_tensor(value) and value.numel() == 1
    }
    payload = {
        "train_dryrun_ok": True,
        "loss": float(output.loss.detach().cpu()),
        "loss_items": loss_items,
        "output_file": "./outputs/smoke/train_dryrun.json",
    }
    with open("./outputs/smoke/train_dryrun.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

# PrefMLLM Paper Alignment

This release is organized around the PrefMLLM paper narrative.  A few internal
configuration keys still use the `name_memory` prefix only for checkpoint
compatibility.

## Method Mapping

| Paper term | Release implementation |
| --- | --- |
| Factorized user state `S_i` | `llava.model.prefmllm.FactorizedUserStateMemory` |
| Image profile factor `z_img` | `image_token` / `slot_img` in `NameMemoryModule.build_prefix` |
| Description profile factor `z_desc` | `description_token` / `slot_desc` |
| Five preference facets | `PREFMLLM_FACTORS = entertainment, travel, lifestyle, shopping, fashion` |
| Shared preference prototype `z_bar_f` | `shared_pref_points` and `shared_pref_tokens` |
| Personalized residual `Delta_i,f` | `offset_pref_tokens` / `raw_slot_pref` |
| Counterfactual users | `mmpb_clean/pseudo_users.csv` loaded through `load_pseudo_user_bank` |
| Profile alignment/separation | `loss_consistency_img_desc`, `loss_contrast_profile_img`, `loss_contrast_description` |
| Preference factor contrast | `loss_pref_factor_contrast` |
| Imbalance-aware focal residual preservation | `loss_pref_focal` / `loss_pref_residual_density_focal` |
| Preference decorrelation | `loss_pref_decorrelation` |
| Factorized user-aware hierarchical MoE | `FactorizedUserAwareHierarchicalMoE` |
| Task-adaptive preference/profile fusion | `task_router_mode=query` |
| Preference-facet router | `pref_router_mode=learned` |
| Profile image/description router | `profile_router_mode=learned` |
| Preference collapse metric | `llava.eval.prefmllm.collapse_metrics` |

## Audit Checklist

- Model structure: PrefMLLM wraps the LLaVA backbone with a factorized user-state
  memory and a top-layer hierarchical LoRA-MoE adapter.
- Preference factors: the release uses the five paper facets and keeps
  factor-specific preference tokens instead of a single monolithic vector.
- Prototype/residual decomposition: `final_pref_tokens = shared_pref_tokens +
  offset_pref_tokens`.
- Profile losses: image-description consistency and profile contrastive losses are
  present.
- Preference losses: density-aware focal residual preservation and the two-term
  residual/prototype decorrelation objective are present and configurable.
- Counterfactual augmentation: pseudo users are loaded from `./data/mmpb_clean/pseudo_users.csv`
  and added to the residual contrastive candidate set.
- Routing: preference, profile, and task routers are separate; the public recipe
  uses query-conditioned task routing.
- Inference privacy path: evaluation with `--drop-profile-in-test` relies on the
  learned user state rather than raw profile text in the prompt.
- 0-turn/10-turn evaluation: 0-turn is default; 10-turn is enabled with
  `--generic-conversation-enable --generic-conversation-n-turn 10`.
- Private paths/artifacts: release defaults use `./data`, `./checkpoints`, and
  `./outputs`; checkpoints, logs, run folders, and pretrained weights are excluded.

## Known Compatibility Notes

- Some internal flags still begin with `name_memory_` because existing checkpoints
  store those config keys.  Use `--train_mode prefmllm` and the release scripts as
  the paper-facing entry points.
- Full training and full model evaluation require external LLaVA/Vicuna/CLIP
  checkpoints.  The included smoke tests use a tiny fake backbone to validate the
  PrefMLLM code path without shipping private checkpoints.

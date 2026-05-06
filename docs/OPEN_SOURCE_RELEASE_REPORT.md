# Open-Source Release Report

## Release Directory

This release is staged in `opensource_release/`. It is an independent copy for publication; the original research workspace was not edited.

## Main Changes

- Added paper-facing PrefMLLM entry points:
  - `llava/model/prefmllm/__init__.py`
  - `llava/eval/prefmllm/collapse_metrics.py`
  - aliases in `llava/model/memory/__init__.py`
- Renamed the publication-facing MoE/LoRA adapter package to `PrefMoE`.
- Kept checkpoint-compatible internal `name_memory` modules while exposing paper terminology for factorized user state and hierarchical MoE.
- Added publication configuration and runnable scripts:
  - `configs/prefmllm_default.json`
  - `scripts/train/train_prefmllm.sh`
  - `scripts/eval/eval_prefmllm_0turn.sh`
  - `scripts/eval/eval_prefmllm_10turn.sh`
  - `scripts/eval/compute_preference_collapse.sh`
- Added smoke-test scaffolding and an incomplete 200-row MMPB clean public subset:
  - `scripts/smoke/check_config.py`
  - `scripts/smoke/check_imports.py`
  - `scripts/smoke/run_train_dryrun.py`
  - `scripts/smoke/run_eval_dryrun.py`
  - `data/mmpb_clean/sample.csv`
  - `data/mmpb_clean/split.json`
  - `data/mmpb_clean/pseudo_users.csv`
  - `data/mmpb_clean/images/`
  - `data/mmpb_clean/injection/`
  - `data/mmpb_clean/manifest.json`
- Added release metadata:
  - `README.md`
  - `docs/PAPER_ALIGNMENT.md`
  - `requirements.txt`
  - `LICENSE`
  - `.gitignore`
- Removed private paths, old local scripts, checkpoints, logs, runs, generated evaluation dumps, and oversized artifacts.

## Paper-Alignment Checklist

- Factorized user state is represented by image profile, description profile, and five preference facets.
- Preference factors keep the paper decomposition into shared prototype plus personalized residual.
- Hierarchical MoE remains query-conditioned/task-adaptive, with preference experts, profile experts, and a task router.
- Training follows the paper loss form: VQA CE, profile consistency/contrast, density-aware focal residual preservation, two-term preference decorrelation, and counterfactual users in the residual contrastive candidate set.
- Evaluation supports both 0-turn and 10-turn personalized VQA modes.
- Preference-collapse reporting implements boundary-external false positive rate (`FPout`) with small/middle/large preference buckets.
- The bundled incomplete MMPB clean subset has 160 train and 40 test rows, covering preference yes/no, recognition yes/no, preference MCQ, and recognition MCQ across all five preference facets.

## Smoke Commands

Run from the release root:

```bash
python3 -m compileall -q llava scripts
python3 scripts/smoke/check_config.py
python3 scripts/smoke/check_imports.py
python3 scripts/smoke/run_train_dryrun.py
python3 scripts/smoke/run_eval_dryrun.py
```

## Smoke Results

- `compileall`: passed.
- `check_config.py`: passed; method is `PrefMLLM` with five preference facets.
- `check_imports.py`: passed; PrefMLLM aliases, metrics, and collapse metric import successfully.
- `run_train_dryrun.py`: passed; tiny fake backbone produced a finite training loss with `loss_pref_decorrelation` and `loss_pref_residual_density_focal`, then wrote `outputs/smoke/train_dryrun.json`.
- `run_eval_dryrun.py`: passed; generated 0-turn and 10-turn dry-run outputs plus `FPout` reports with a boundary-external smoke case under `outputs/smoke/`.
- `PrefMoE.peft` direct import: passed; `PrefMoEMOELoraConfig`, `TaskType.CAUSAL_LM_PrefMoE`, and `PeftType.MOE_LORA_PrefMoE` resolve successfully.

The local environment prints bitsandbytes CPU-build warnings during import. The smoke tests still exit successfully.

## Publication Notes

- Full training and full benchmark evaluation were not rerun in this release directory because real checkpoints and private datasets are intentionally excluded.
- Users must provide their own LLaVA/Vicuna backbone, projector/checkpoint paths, and full personalized VQA data before running the main train/eval scripts.
- The `outputs/`, `checkpoints/`, `logs/`, `runs/`, and `wandb/` directories are ignored by `.gitignore`.
- Generated smoke outputs were cleaned after validation; rerunning the smoke scripts will recreate them.
- Hidden copied `.git` metadata was removed from the release directory.

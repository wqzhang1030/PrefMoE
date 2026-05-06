import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from llava.eval.CVLMP.eval_demo.metrics import evaluate_mmpb_no_gpt, report_acc
from llava.eval.prefmllm import write_preference_collapse_report


def _split_names(value):
    names = []
    for chunk in str(value or "").replace(";", "|").replace(",", "|").split("|"):
        item = chunk.strip()
        if item:
            names.append(item)
    return set(names)


def _boolish(value):
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "invert", "inverted"}


def _semantic_positive(row) -> bool:
    surface_positive = int(float(row["answer"])) == 4
    return (not surface_positive) if _boolish(row.get("preference_semantic_invert", False)) else surface_positive


def _boundary_external_negative(row) -> bool:
    users = _split_names(row.get("preference_bucket_names", ""))
    return (not _semantic_positive(row)) and str(row.get("name", "")).strip() not in users


def _run_one(turns: int, predictions):
    out_dir = f"./outputs/smoke/eval_{turns}turn"
    os.makedirs(out_dir, exist_ok=True)
    frame = pd.read_csv("./data/mmpb_clean/sample.csv")
    pref = frame[
        frame.get("category", "").astype(str).str.lower().eq("preference")
        & pd.to_numeric(frame.get("answer"), errors="coerce").isin([4, 5])
    ]
    positive_row = pref[pref.apply(_semantic_positive, axis=1)].head(1)
    boundary_row = pref[pref.apply(_boundary_external_negative, axis=1)].head(1)
    frame = pd.concat([positive_row, boundary_row], ignore_index=True)
    if len(frame) != 2:
        raise RuntimeError("Eval dry-run requires one positive and one boundary-negative preference example.")
    frame["prediction"] = predictions
    scored = evaluate_mmpb_no_gpt(frame)
    raw_path = os.path.join(out_dir, "raw.csv")
    score_path = os.path.join(out_dir, "score.csv")
    scored.to_csv(raw_path, index=False)
    report_acc(scored).to_csv(score_path, index=False)
    collapse_payload = write_preference_collapse_report(raw_path, os.path.join(out_dir, "fpout"))
    return {
        "turns": turns,
        "raw": raw_path,
        "score": score_path,
        "collapse": collapse_payload,
    }


def main() -> None:
    os.makedirs("./outputs/smoke", exist_ok=True)
    zero_turn = _run_one(0, ["yes", "yes"])
    ten_turn = _run_one(10, ["no", "yes"])
    payload = {
        "eval_dryrun_ok": True,
        "zero_turn": zero_turn,
        "ten_turn": ten_turn,
    }
    with open("./outputs/smoke/eval_dryrun.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

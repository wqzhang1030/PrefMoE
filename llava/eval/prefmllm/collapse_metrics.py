import argparse
import json
import os
from typing import Iterable, List, Set, Tuple

import numpy as np
import pandas as pd

from llava.eval.CVLMP.eval_demo.metrics import answer_to_text, normalize_yes_no


def _split_user_set(value) -> Set[str]:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return set()
    parts = []
    for chunk in text.replace(";", "|").replace(",", "|").split("|"):
        item = chunk.strip()
        if item:
            parts.append(item)
    return set(parts)


def _boolish(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "invert", "inverted"}


def _semantic_label(surface: str, invert: bool) -> int:
    label = 1 if str(surface).strip().lower() == "yes" else 0
    return 1 - label if invert else label


def _bucket_name(size: int) -> str:
    if size <= 4:
        return "small_<=4"
    if size <= 8:
        return "middle_5-8"
    return "large_>=9"


def _first_existing(columns: Iterable[str], frame: pd.DataFrame):
    for column in columns:
        if column in frame.columns:
            return column
    return None


def compute_preference_collapse(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute the paper's FPout preference-collapse metric.

    Expected input columns:
    - `name`: target user id/name.
    - `answer` and `prediction`, or precomputed `gt_norm` and `pred_norm`.
    - `preference_bucket_names`/`positive_bucket`/`true_user_set`: users in S_i.
    - optional `preference_semantic_invert`/`semantic_invert`: whether yes/no semantics are inverted.
    """
    work = frame.copy().reset_index(drop=True)
    if "category" in work.columns:
        work = work[work["category"].astype(str).str.lower().eq("preference")].copy()

    bucket_col = _first_existing(
        ("preference_bucket_names", "positive_bucket", "true_user_set", "bucket_names"),
        work,
    )
    if bucket_col is None:
        raise ValueError("Preference collapse requires a true user set column such as preference_bucket_names.")
    if "name" not in work.columns:
        raise ValueError("Preference collapse requires a target user column named 'name'.")

    invert_col = _first_existing(("preference_semantic_invert", "semantic_invert", "is_inverted"), work)
    gt_col = "gt_norm" if "gt_norm" in work.columns else None
    pred_col = "pred_norm" if "pred_norm" in work.columns else None

    rows: List[dict] = []
    for row_idx, row in work.iterrows():
        users = _split_user_set(row.get(bucket_col, ""))
        target_user = str(row.get("name", "")).strip()
        gt_surface = str(row.get(gt_col, "")) if gt_col else answer_to_text(row.get("answer", ""))
        pred_surface = str(row.get(pred_col, "")) if pred_col else normalize_yes_no(row.get("prediction", ""))
        if gt_surface not in {"yes", "no"} or pred_surface not in {"yes", "no"}:
            continue

        invert = _boolish(row.get(invert_col, False)) if invert_col else False
        y_true = _semantic_label(gt_surface, invert)
        y_pred = _semantic_label(pred_surface, invert)
        outside = int(target_user not in users and y_true == 0)
        bucket_size = len(users)
        enriched = row.to_dict()
        enriched.update(
            {
                "collapse_target_user": target_user,
                "collapse_true_user_set": " | ".join(sorted(users)),
                "collapse_bucket_size": bucket_size,
                "collapse_bucket": _bucket_name(bucket_size),
                "collapse_y_true": y_true,
                "collapse_y_pred": y_pred,
                "collapse_boundary_external": outside,
                "collapse_fpout_error": int(outside and y_pred == 1),
            }
        )
        rows.append(enriched)

    enriched_df = pd.DataFrame(rows)
    if len(enriched_df) == 0:
        summary = pd.DataFrame([{"metric": "FPout", "value": np.nan, "n_boundary_external": 0}])
        bucket_details = pd.DataFrame(columns=["bucket", "FPout", "n_boundary_external"])
        return enriched_df, summary, bucket_details

    boundary = enriched_df[enriched_df["collapse_boundary_external"].eq(1)]
    fpout = float(boundary["collapse_fpout_error"].mean()) if len(boundary) else np.nan
    summary = pd.DataFrame(
        [
            {
                "metric": "FPout",
                "value": fpout,
                "n_boundary_external": int(len(boundary)),
                "n_preference_yesno": int(len(enriched_df)),
            }
        ]
    )

    bucket_rows = []
    for bucket in ("small_<=4", "middle_5-8", "large_>=9"):
        sub = boundary[boundary["collapse_bucket"].eq(bucket)]
        bucket_rows.append(
            {
                "bucket": bucket,
                "FPout": float(sub["collapse_fpout_error"].mean()) if len(sub) else np.nan,
                "n_boundary_external": int(len(sub)),
            }
        )
    bucket_details = pd.DataFrame(bucket_rows)
    return enriched_df, summary, bucket_details


def write_preference_collapse_report(input_csv: str, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    frame = pd.read_csv(input_csv)
    enriched, summary, bucket_details = compute_preference_collapse(frame)

    enriched_path = os.path.join(output_dir, "raw_enriched.csv")
    summary_path = os.path.join(output_dir, "main_metrics.csv")
    bucket_path = os.path.join(output_dir, "collapse_bucket_details.csv")
    json_path = os.path.join(output_dir, "summary.json")

    enriched.to_csv(enriched_path, index=False)
    summary.to_csv(summary_path, index=False)
    bucket_details.to_csv(bucket_path, index=False)
    payload = {
        "FPout": None if summary.empty or pd.isna(summary.iloc[0]["value"]) else float(summary.iloc[0]["value"]),
        "n_boundary_external": 0 if summary.empty else int(summary.iloc[0]["n_boundary_external"]),
        "files": {
            "raw_enriched": enriched_path,
            "main_metrics": summary_path,
            "collapse_bucket_details": bucket_path,
        },
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute PrefMLLM preference-collapse FPout metrics.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", default="./outputs/collapse_metrics")
    args = parser.parse_args()
    payload = write_preference_collapse_report(args.input_csv, args.output_dir)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

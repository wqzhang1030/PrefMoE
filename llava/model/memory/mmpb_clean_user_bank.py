from pathlib import Path
from typing import Iterable

import pandas as pd


PSEUDO_USER_CSV_COLUMNS = (
    "pseudo_id",
    "pseudo_name",
    "concept_family",
    "description_text",
    "pref_entertainment",
    "pref_travel",
    "pref_lifestyle",
    "pref_shopping",
    "pref_fashion",
    "uid_seed",
    "image_seed",
)


def _project_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").exists() and (parent / "llava").is_dir():
            return parent
    return path.parents[3]


def resolve_pseudo_user_csv_path(csv_path: str) -> Path:
    raw = Path(str(csv_path or "").strip())
    if raw.is_absolute():
        return raw

    direct = Path.cwd() / raw
    if direct.exists():
        return direct

    project_relative = _project_root() / raw
    if project_relative.exists():
        return project_relative

    return project_relative


def _require_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Pseudo user CSV is missing required columns: {missing}")


def load_pseudo_user_bank(csv_path: str, expected_rows: int = 50) -> pd.DataFrame:
    resolved = resolve_pseudo_user_csv_path(csv_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Pseudo user CSV not found: {resolved}")

    frame = pd.read_csv(resolved)
    _require_columns(frame, PSEUDO_USER_CSV_COLUMNS)

    if expected_rows > 0 and len(frame) != int(expected_rows):
        raise ValueError(
            f"Pseudo user CSV row count mismatch: expected {expected_rows}, got {len(frame)} ({resolved})"
        )

    frame = frame[list(PSEUDO_USER_CSV_COLUMNS)].copy()
    frame["pseudo_id"] = frame["pseudo_id"].astype(int)
    frame["pseudo_name"] = frame["pseudo_name"].astype(str)
    frame["concept_family"] = frame["concept_family"].astype(str).str.strip().str.lower()
    frame["uid_seed"] = frame["uid_seed"].astype(int)
    frame["image_seed"] = frame["image_seed"].astype(int)
    frame = frame.sort_values("pseudo_id").reset_index(drop=True)
    return frame

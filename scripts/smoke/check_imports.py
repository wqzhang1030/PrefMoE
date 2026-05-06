import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llava.eval.CVLMP.eval_demo.metrics import score_prediction
from llava.eval.prefmllm import compute_preference_collapse
from llava.model.prefmllm import (
    PREFMLLM_FACTORS,
    FactorizedUserAwareHierarchicalMoE,
    FactorizedUserStateMemory,
)


def main() -> None:
    os.makedirs("./outputs/smoke", exist_ok=True)
    scored = score_prediction("yes", "yes", "awareness", {})
    payload = {
        "imports_ok": True,
        "factors": list(PREFMLLM_FACTORS),
        "factorized_user_state_class": FactorizedUserStateMemory.__name__,
        "hierarchical_moe_class": FactorizedUserAwareHierarchicalMoE.__name__,
        "metric_hit": int(scored["hit"]),
        "collapse_callable": callable(compute_preference_collapse),
    }
    with open("./outputs/smoke/import_test.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

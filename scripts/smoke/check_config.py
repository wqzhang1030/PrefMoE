import json
import os


def main() -> None:
    with open("./configs/prefmllm_default.json", "r", encoding="utf-8") as handle:
        config = json.load(handle)

    required = [
        ("method",),
        ("paper_terms", "factorized_user_state"),
        ("paper_terms", "routing"),
        ("training", "train_mode"),
        ("evaluation", "zero_turn"),
        ("evaluation", "ten_turn"),
    ]
    missing = []
    for path in required:
        node = config
        for key in path:
            if key not in node:
                missing.append(".".join(path))
                break
            node = node[key]
    if missing:
        raise SystemExit(f"Missing config keys: {missing}")
    if config["training"]["train_mode"] != "prefmllm":
        raise SystemExit("configs/prefmllm_default.json must use train_mode=prefmllm")

    os.makedirs("./outputs/smoke", exist_ok=True)
    payload = {
        "config_ok": True,
        "method": config["method"],
        "num_preference_facets": len(config["paper_terms"]["preference_facets"]),
    }
    with open("./outputs/smoke/config_test.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

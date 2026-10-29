import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METRICS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "evaluation_metrics.json"
)

MIN_F1 = 0.60
MIN_ROC_AUC = 0.80


def main():

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    f1 = metrics["f1_score"]
    roc_auc = metrics["roc_auc"]

    print("\nModel Quality Gate")
    print("-----------------")
    print(f"F1-score: {f1:.4f} (minimum: {MIN_F1})")
    print(f"ROC-AUC:  {roc_auc:.4f} (minimum: {MIN_ROC_AUC})")

    if f1 < MIN_F1 or roc_auc < MIN_ROC_AUC:
        raise SystemExit(
            "\nQUALITY GATE FAILED: Model does not meet "
            "the minimum evaluation criteria."
        )

    print("\nQUALITY GATE PASSED: Model meets the criteria.")


if __name__ == "__main__":
    main()
import argparse
import sys
import time
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import torch

from src.config import SUBMISSIONS_DIR, TEST_PATH, get_device
from src.features import add_feats
from src.train_utils import LogisticClass, to_dense


def save_submission(out_path: Path, test_df: pd.DataFrame, preds) -> None:
    submission_df = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Survived": preds}
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(out_path, index=False)


def predict_logistic(artifact: dict, test_df: pd.DataFrame, device: str):
    X = test_df.copy()
    if artifact.get("feature_engineering") == "add_feats":
        X = add_feats(X)
    preprocess = artifact["preprocess"]
    X_proc = to_dense(preprocess.transform(X))
    model = LogisticClass(artifact["input_dim"]).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_proc, dtype=torch.float32).to(device))
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    return (probs >= 0.5).astype(int)


def predict_xgb(artifact: dict, test_df: pd.DataFrame):
    pipe = artifact.get("pipeline", artifact)
    probs = pipe.predict_proba(test_df)[:, 1]
    return (probs >= 0.5).astype(int)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Kaggle submissions.")
    parser.add_argument(
        "--model-path",
        required=True,
        type=Path,
        help="Path to a saved model artifact.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=TEST_PATH,
        help="Path to test.csv.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Custom submission filename (without extension).",
    )
    args = parser.parse_args()

    artifact = joblib.load(args.model_path)
    model_type = artifact.get("model_type", "xgb")

    test_df = pd.read_csv(args.test_path)
    device = get_device()

    if model_type in {"baseline_logistic", "improved_logistic"}:
        preds = predict_logistic(artifact, test_df, device)
    else:
        preds = predict_xgb(artifact, test_df)

    timestamp = time.strftime("%Y%m%d-%H-%M-%S")
    output_name = args.output_name or f"submission_{model_type}_{timestamp}"
    out_path = SUBMISSIONS_DIR / f"{output_name}.csv"
    save_submission(out_path, test_df, preds)
    print(f"Saved submission to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

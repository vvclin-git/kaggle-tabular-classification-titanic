import argparse
import json
import sys
import time
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from src.config import MODELS_DIR, PARAMS_PATH, SEED, TRAIN_PATH, get_device
from src.features import add_feats
from src.preprocess import (
    create_preprocess_baseline,
    create_preprocess_improved,
    create_preprocess_xgb,
)
from src.train_utils import LogisticClass, Trainer, to_dense


def evaluate_binary(y_true, probs) -> dict:
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, preds),
        "auc": roc_auc_score(y_true, probs),
    }


def load_params(params_path: Path, key: str) -> dict:
    if not params_path.exists():
        return {}
    try:
        data = json.loads(params_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data.get(key, {})


def resolve_params(model_key: str, defaults: dict, use_hpo: bool, params_path: Path) -> tuple:
    hpo_meta = load_params(params_path, model_key) if use_hpo else {}
    params = defaults.copy()
    return params, hpo_meta


def train_logistic(
    train_df: pd.DataFrame, preprocess, params: dict, device: str, use_feats: bool
):
    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"].to_numpy()
    if use_feats:
        X = add_feats(X)
    X_proc = to_dense(preprocess.fit_transform(X))

    model = LogisticClass(X_proc.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    trainer = Trainer(model, optimizer, torch.nn.BCEWithLogitsLoss(), device=device)
    trainer.train((X_proc, y), batch_size=params["batch_size"], epochs=params["epochs"])
    with torch.no_grad():
        logits = model(torch.tensor(X_proc, dtype=torch.float32).to(device))
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    metrics = evaluate_binary(y, probs)
    return model, preprocess, X_proc.shape[1], metrics


def train_xgb(train_df: pd.DataFrame, params: dict):
    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"]
    pipe = Pipeline(
        [
            ("fe", FunctionTransformer(add_feats, validate=False)),
            ("preprocess", create_preprocess_xgb()),
            (
                "model",
                XGBClassifier(
                    **params,
                    eval_metric="logloss",
                    random_state=SEED,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    probs = pipe.predict_proba(X)[:, 1]
    metrics = evaluate_binary(y.to_numpy(), probs)
    return pipe, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Titanic models.")
    parser.add_argument(
        "--model",
        required=True,
        choices=["baseline", "improved", "xgb"],
        help="Model type to train.",
    )
    parser.add_argument(
        "--use-hpo",
        action="store_true",
        help="Load best params from outputs/params/best_params.json if available.",
    )
    parser.add_argument(
        "--params-path",
        type=Path,
        default=PARAMS_PATH,
        help="Path to best params JSON.",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="Custom output name (without extension).",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(TRAIN_PATH)
    device = get_device()

    timestamp = time.strftime("%Y%m%d-%H-%M-%S")

    if args.model == "baseline":
        model_key = "baseline_logistic"
        defaults = {"lr": 1e-2, "weight_decay": 0.0, "batch_size": 64, "epochs": 5}
        params, hpo_meta = resolve_params(model_key, defaults, args.use_hpo, args.params_path)
        if hpo_meta:
            params.update({k: hpo_meta[k] for k in ["lr", "weight_decay", "batch_size", "epochs"] if k in hpo_meta})
        preprocess = create_preprocess_baseline()
        model, preprocess, in_dim, train_metrics = train_logistic(
            train_df, preprocess, params, device, use_feats=False
        )
        artifact = {
            "model_type": model_key,
            "state_dict": model.state_dict(),
            "input_dim": in_dim,
            "preprocess": preprocess,
            "hyperparams": params,
            "hpo_meta": hpo_meta,
            "feature_engineering": None,
            "train_metrics": train_metrics,
        }
    elif args.model == "improved":
        model_key = "improved_logistic"
        defaults = {"lr": 1e-2, "weight_decay": 0.0, "batch_size": 128, "epochs": 20}
        params, hpo_meta = resolve_params(model_key, defaults, args.use_hpo, args.params_path)
        if hpo_meta:
            params.update({k: hpo_meta[k] for k in ["lr", "weight_decay", "batch_size", "epochs"] if k in hpo_meta})
        preprocess = create_preprocess_improved()
        model, preprocess, in_dim, train_metrics = train_logistic(
            train_df, preprocess, params, device, use_feats=True
        )
        artifact = {
            "model_type": model_key,
            "state_dict": model.state_dict(),
            "input_dim": in_dim,
            "preprocess": preprocess,
            "hyperparams": params,
            "hpo_meta": hpo_meta,
            "feature_engineering": "add_feats",
            "train_metrics": train_metrics,
        }
    else:
        model_key = "xgb"
        defaults = {
            "n_estimators": 100,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 5,
            "gamma": 0.3,
            "subsample": 0.6,
            "colsample_bytree": 0.6,
            "reg_lambda": 8,
            "reg_alpha": 0.2,
        }
        params, hpo_meta = resolve_params(model_key, defaults, args.use_hpo, args.params_path)
        if hpo_meta:
            params.update(
                {
                    k: hpo_meta[k]
                    for k in [
                        "n_estimators",
                        "learning_rate",
                        "max_depth",
                        "min_child_weight",
                        "gamma",
                        "subsample",
                        "colsample_bytree",
                        "reg_lambda",
                        "reg_alpha",
                    ]
                    if k in hpo_meta
                }
            )
        pipe, train_metrics = train_xgb(train_df, params)
        artifact = {
            "model_type": model_key,
            "pipeline": pipe,
            "hyperparams": params,
            "hpo_meta": hpo_meta,
            "train_metrics": train_metrics,
        }

    artifact_name = args.save_name or f"{model_key}_{timestamp}"
    artifact_path = MODELS_DIR / f"{artifact_name}.joblib"
    joblib.dump(artifact, artifact_path)
    print(
        "Train metrics - accuracy: {:.4f}, auc: {:.4f}".format(
            artifact["train_metrics"]["accuracy"],
            artifact["train_metrics"]["auc"],
        )
    )
    print(f"Saved model artifact to {artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

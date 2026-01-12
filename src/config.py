from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
SUBMISSIONS_DIR = OUTPUT_DIR / "submissions"
PARAMS_PATH = OUTPUT_DIR / "params" / "best_params.json"

SEED = 37


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

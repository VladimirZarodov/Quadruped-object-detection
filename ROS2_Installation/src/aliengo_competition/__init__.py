import os
from pathlib import Path


for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")


COMPETITION_ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = COMPETITION_ROOT_DIR / "src"
CONFIGS_DIR = COMPETITION_ROOT_DIR / "configs"
MODELS_DIR = COMPETITION_ROOT_DIR / "models"
RESOURCES_DIR = COMPETITION_ROOT_DIR / "resources"

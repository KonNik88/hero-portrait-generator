from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

with open(DATA_DIR / "dataset_config.json", "r", encoding="utf-8") as f:
    DATASET_CONFIG = json.load(f)

CDC_GAN_CONFIG = {
    "image_size": DATASET_CONFIG["image_size"],
    "channels": DATASET_CONFIG["channels"],
    "cond_dim": DATASET_CONFIG["cond_dim"],
    "nz": 128,
    "ngf": 64,
    "ndf": 64,
}

import os
from pathlib import Path
from dotenv import load_dotenv

# Automatically load environment variables from .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

FLIR_ROOT = Path(os.getenv("ECHO_SARS_FLIR_ROOT", str(BASE_DIR / "Dataset" / "FLIR_ADAS_v2")))
FROG_TRAIN_VAL = Path(os.getenv("ECHO_SARS_FROG_TRAIN_VAL", str(BASE_DIR / "Dataset" / "frog_11-36_12-43_train_val.h5")))
FROG_TEST = Path(os.getenv("ECHO_SARS_FROG_TEST", str(BASE_DIR / "Dataset" / "frog_16-41_test.h5")))

YOLO_MODEL = os.getenv("ECHO_SARS_YOLO_MODEL", "yolov8n.pt")
CONF_THRESHOLD = float(os.getenv("ECHO_SARS_CONF_THRESHOLD", "0.35"))
OUTPUT_DIR = Path(os.getenv("ECHO_SARS_OUTPUT_DIR", str(BASE_DIR / "outputs")))


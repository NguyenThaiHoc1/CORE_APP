from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / 'core_makeup' / 'model'
MODEL_PATH_ARCHITECTURE = MODEL_PATH / 'model.meta'

TENSORBOARD_DIR = BASE_DIR / 'core_makeup' / 'tensorboard'

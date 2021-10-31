from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / 'core' / 'model'

SOURCE_FRAME = BASE_DIR / 'data'

CARTOON_IMAGE = BASE_DIR / 'cartoonized_images'

MODEL_FILE_TF2 = BASE_DIR / 'core' / 'model_tf2' / 'weights.h5'

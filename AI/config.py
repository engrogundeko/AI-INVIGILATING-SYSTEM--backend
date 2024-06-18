import dotenv
import os


dotenv.load_dotenv()

# Loading environment variables into constants
ROOT_DIR = os.environ.get("WORKING_DIR")

IMG_IN_DIR = os.path.join(ROOT_DIR, "media", "datasets", "converted")
IMG_OUT_DIR = os.path.join(ROOT_DIR, "media", "datasets", "frame_classifier", "image", "all")
CLASS_DIRS = [
    "back",
    "exchange",
    "normal",
    "peek",
]

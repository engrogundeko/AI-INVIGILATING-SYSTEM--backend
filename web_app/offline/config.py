import dotenv
import os
import numpy as np
from PIL import Image


def check_and_create_folder(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


dotenv.load_dotenv()

# Loading environment variables into constants

ALGORITHM = os.environ.get("ALGORITHM")
SECRET_KEY = os.environ.get("SECRET_KEY")
ACCESS_TOKEN_EXPIRE_MINUTES = os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES")
MONGODB_URL = "mongodb://localhost:27017/"

WKR_DIR = os.environ.get("WORKING_DIR")
model_version = os.environ.get("MODEL_VERSION")
MEDIA_DIR = os.environ.get("MEDIA")

if MEDIA_DIR is None:
    raise ValueError("No media directory was provided")

if WKR_DIR is None:
    raise ValueError("No working directory was provided")

if not os.path.exists(WKR_DIR) and not os.path.exists(MEDIA_DIR):
    raise ValueError(
        """Working directory or Media Directory has not been set:
        Working directory should be the location of the cloned directory
        Media directory should be the location of the saved files
        """
    )


if model_version is None:
    model_version = 6

DATABASE = os.path.join(MEDIA_DIR, "database")
check_and_create_folder(DATABASE)

IMAGE_RECORD_PATH = os.path.join(DATABASE, "suspicious_image")
check_and_create_folder(IMAGE_RECORD_PATH)

VIDEO_RECORD_PATH = os.path.join(DATABASE, "exam_video")
check_and_create_folder(VIDEO_RECORD_PATH)

IMAGE_SAVE_DIR = os.path.join(DATABASE, "student_face")
check_and_create_folder(IMAGE_SAVE_DIR)

REPORT_FOLDER = os.path.join(MEDIA_DIR, "reports")
check_and_create_folder(REPORT_FOLDER)

STATIC_FOLDER = os.path.join(MEDIA_DIR, "graph")
check_and_create_folder(STATIC_FOLDER)

MODELS_DIR = os.path.join(WKR_DIR, "models")
check_and_create_folder(MODELS_DIR)

STUDENT_DATA = os.path.join(DATABASE, "student_data")
check_and_create_folder(STUDENT_DATA)

AIS_HEAD_MODEL = os.path.join(MODELS_DIR, "aisv1n.pt")
AIS_YOLO = os.path.abspath(os.path.join(MODELS_DIR, "aisv4l.pt"))
AIS__MODEL = os.path.join(MODELS_DIR, f"ais_modelv{int(model_version)}.pkl")
# STATIC_FOLDER = os.path.join(MEDIA_DIR, "grap", "images")

class AIConstant:
    time_format = "%H:%M:%S"
    VALID_ROLE: list = ["ADMIN", "STUDENT", "LECTURER", "INVIGILATOR"]

    @staticmethod
    def save_image(image, image_name):
        if image.dtype != np.uint8:
            image = (255 * image).astype(np.uint8)
        if not os.path.exists(IMAGE_SAVE_DIR):
            os.makedirs(IMAGE_SAVE_DIR)
        image_path = os.path.join(IMAGE_SAVE_DIR, image_name)
        image = Image.fromarray(image)  # Convert NumPy array to PIL image
        image.save(image_path)

        return image_path

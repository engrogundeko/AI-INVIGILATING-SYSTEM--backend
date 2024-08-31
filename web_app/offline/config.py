import dotenv
import os
import numpy as np
from PIL import Image


dotenv.load_dotenv()

# Loading environment variables into constants
WKR_DIR = os.environ.get("WORKING_DIR")
MONGODB_URL = os.environ.get("MONGODB")
ALGORITHM = os.environ.get("ALGORITHM")
SECRET_KEY = os.environ.get("SECRET_KEY")
ACCESS_TOKEN_EXPIRE_MINUTES = os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES")

MEDIA_DIR = os.path.join(WKR_DIR, "media")
DATABASE = os.path.join(MEDIA_DIR, "database")
IMAGE_RECORD_PATH = os.path.join(DATABASE, "suspicious_image")
VIDEO_RECORD_PATH = os.path.join(DATABASE, "exam_video")
IMAGE_SAVE_DIR = os.path.join(DATABASE, "student_face")
UNRECOGNISED_FACE = os.path.join(DATABASE, "unrecognised_faces")
UNDETECTED_FACE = os.path.join(DATABASE, "undetected_face")
MODELS_DIR = os.path.join(MEDIA_DIR, "models")
AIS_YOLO = r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\aisv1n.pt"
AIS_HEAD_MODEL = os.path.join(MODELS_DIR, "aisv1n.pt")
AIS__MODEL = r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\ais_modelv5.pkl"

class AIConstant:
    PERSON_CLASS: int = 0
    VALID_ROLE: list = ["ADMIN", "STUDENT", "LECTURER", "INVIGILATOR"]
    PROB_ALLOWANCE: int = 0.6

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

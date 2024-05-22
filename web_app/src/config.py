import dotenv
import os


dotenv.load_dotenv()

WKR_DIR = os.environ.get("WORKING_DIR")
MONGODB_URL = os.environ.get("MONGODB")

MEDIA_DIR = os.path.join(WKR_DIR, "media")
DATABASE = os.path.join(MEDIA_DIR, "database")
YOLOV8_MODEL = os.path.join(MEDIA_DIR, "model", "yolov8n.pt")


class AIConstant:
    PERSON_CLASS: int = 0
    VALID_ROLE = ["ADMIN", "STUDENT", "LECTURER", "INVIGILATOR"]
    PROB_ALLOWANCE: int = 0.6

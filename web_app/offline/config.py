import dotenv
import os


dotenv.load_dotenv()

# Loading environment variables into constants
WKR_DIR = os.environ.get("WORKING_DIR")
MONGODB_URL = os.environ.get("MONGODB")
ALGORITHM = os.environ.get("ALGORITHM")
SECRET_KEY = os.environ.get("SECRET_KEY")
ACCESS_TOKEN_EXPIRE_MINUTES = os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES")

MEDIA_DIR = os.path.join(WKR_DIR, "media")
DATABASE = os.path.join(MEDIA_DIR, "database")


class AIConstant:
    PERSON_CLASS: int = 0
    VALID_ROLE: list = ["ADMIN", "STUDENT", "LECTURER", "INVIGILATOR"]
    PROB_ALLOWANCE: int = 0.6

import dotenv
import os


dotenv.load_dotenv()

WKR_DIR = os.environ.get("WORKING_DIR")
MONGODB_URL = os.environ.get("MONGODB")
SECRET_KEY = os.environ.get("SECRET_KEY")
ALGORITHM = os.environ.get("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES")

MEDIA_DIR = os.path.join(WKR_DIR, "media")
DATABASE = os.path.join(MEDIA_DIR, "database")

# Loading environment variables into constants

class AIConstant:
    PERSON_CLASS: int = 0
    VALID_ROLE = ["ADMIN", "STUDENT", "LECTURER", "INVIGILATOR"]
    PROB_ALLOWANCE: int = 0.6

import dotenv
import os

dotenv.load_dotenv()
MONGODB_URL = os.environ.get("MONGODB")

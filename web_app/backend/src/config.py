import dotenv
import os

from pymongo import MongoClient

dotenv.load_dotenv()
MONGODB_URL = os.environ.get("MONGODB")


from ..utils.algorithms.facial_recognition.facial import preprocess_and_embed
from ..utils.images import save_image
from .model import User
from ..repository import repository
from .schema import StudentInSchema


def hash_password(password) -> str:
    return


def create_student_service(payload: StudentInSchema):
    image = save_image(payload.image)
    embed = preprocess_and_embed(image)
    payload.image = image
    payload = payload.__dict__
    payload["embed"] = embed
    print(payload)
    repository.insert_one("user", payload)

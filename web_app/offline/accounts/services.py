from ..algorithms.facial import preprocess_and_embed
from ..utils.images import save_image, save_file
from .model import User
from ..repository import userRespository
from .schema import StudentInSchema


def hash_password(password) -> str:
    return


def create_student_service(payload: StudentInSchema):
    # image = save_file(payload.image)
    # embed = preprocess_and_embed(image)
    # payload.image = image
    payload = payload.__dict__
    # payload["embed"] = embed
    userRespository.insert_one(payload)

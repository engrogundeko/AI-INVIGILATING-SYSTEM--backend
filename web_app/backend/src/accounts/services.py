from typing import BinaryIO, List
from .model import Gender, UserRole, user
from .schema import StudentInSchema


def create_user(payload: StudentInSchema):
    first_name: str
    last_name: str
    matric_no: int
    email: str
    phone: str
    address: str
    gender: Gender
    role: UserRole
    photo_path: List[str]
    photo_embed: List[BinaryIO]

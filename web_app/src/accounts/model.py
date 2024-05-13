from pydantic import BaseModel
from enum import Enum
from typing import List, BinaryIO


class Gender(Enum):
    male = "MALE"
    female = "FEMALE"


class UserRole(Enum):
    student = "STUDENT"
    invigilator = "INVIGILATOR"
    admin = "ADMIN"
    lecturer = "LECTURER"


class User(BaseModel):
    id: int
    first_name: str
    last_name: str
    password: str | None = None
    matric_no: int | None = None
    email: str
    phone: str
    address: str
    gender: Gender
    role: UserRole
    img_path: List[str]
    img_embed: List[bytes]

    @property
    def full_name(self):
        return self.first_name + self.last_name

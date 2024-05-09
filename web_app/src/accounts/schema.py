from dataclasses import dataclass
from fastapi import UploadFile
from typing import BinaryIO, List
from pydantic import BaseModel

from .model import Gender, UserRole

class ImagePayload(BaseModel):
    image: str

@dataclass
class StudentInSchema:
    first_name: str
    last_name: str
    matric_no: int
    email: str
    phone: str
    address: str
    gender: Gender
    role: UserRole
    image: List[UploadFile]

    def validate_email(self):
        pass

    def validate_image(self):
        pass

    def validate_matric_no(self):
        pass

    def validate_phone(self):
        pass

    def __post_init__(self):
        # self.validate_image()
        self.validate_email()
        self.validate_matric_no()
        self.validate_phone()

from dataclasses import dataclass
from fastapi import Form, UploadFile, File
from typing import BinaryIO, List
from pydantic import BaseModel

from .model import Gender, UserRole


class ImagePayload(BaseModel):
    image: str


@dataclass
class StudentInSchema:
    first_name: str = Form(default=None)
    last_name: str = Form(...)
    matric_no: int = Form(...)
    email: str = Form(...)
    phone: str = Form(...)
    address: str = Form(...)
    gender: str = Form(...)
    role: str = Form(...)
    image_path: UploadFile = File(alias="image")
    
    @property
    def get_fullname(self):
        return self.first_name + self.last_name

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

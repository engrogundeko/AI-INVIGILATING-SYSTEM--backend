import io
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..config import AIConstant
from ..repository import userRespository
from ..base import Model

from fastapi import File, Form, UploadFile
from pydantic import BaseModel, field_validator, EmailStr
from fastapi.exceptions import RequestValidationError


class LoginSchema(BaseModel):
    email: EmailStr
    password: str

    @field_validator("email")
    @classmethod
    def validate_user(cls, email):
        user = userRespository.find_one({"email": email})
        if user is None:
            raise RequestValidationError("The user does not exist in our database")

    @field_validator("password")
    @classmethod
    def validate_user(cls, password):
        user = userRespository.find_one({"email": cls.email})
        if user["password"] != password:
            raise RequestValidationError("The password is incorrect")


class ImagePayload(BaseModel):
    image: str


class StudentOutSchema(Model):
    role: str
    email: str
    phone: int
    gender: str
    address: str
    matric_no: int
    last_name: str
    first_name: str
    image: str


@dataclass
class StudentInSchema:
    role: str = Form(...)
    email: str = Form(...)
    phone: str = Form(...)
    gender: str = Form(...)
    address: str = Form(...)
    last_name: str = Form(...)
    matric_no: int = Form(...)
    first_name: str = Form(default=None)
    image: UploadFile = File(alias="image")

    @property
    def get_fullname(self):
        return self.first_name + self.last_name

    def validate_role(self):
        if self.role.upper() not in AIConstant.VALID_ROLE:
            raise RequestValidationError(
                f"The role is not valid, valid roles include {AIConstant.VALID_ROLE}"
            )
        self.role = self.role.upper()

    def validate_email(self):
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(pattern, self.email):
            raise RequestValidationError("Email is invalid")
        user = userRespository.find_one({"email": self.email})
        if user is not None:
            raise RequestValidationError("The email already exist in our database")

    def validate_image(self):
        from ..utils.image_utils import load_image
        from deepface import DeepFace

        try:
            file_content = self.image.file.read()
            img = Image.open(io.BytesIO(file_content))
            arr = np.array(img)
            img, _ = load_image(arr)
        except Exception as e:
            raise RequestValidationError(str(e))
        try:
            faces = DeepFace.extract_faces(img, detector_backend="yolov8")
        except Exception as e:
            raise RequestValidationError("No faces found in the image")
        student_face = faces[0]["face"]
        student_face_path = f"{self.matric_no}.jpg"
        path = AIConstant.save_image(student_face, student_face_path)
        self.image = path

    def validate_matric_no(self):
        if len(str(self.matric_no)) < 9:
            raise RequestValidationError("The matric no is invalid")
        user = userRespository.find_one({"matric_no": self.matric_no})
        if user is not None:
            raise RequestValidationError("The matric no already exist in our database")

    def validate_phone(self):
        if len(str(self.phone)) < 11:
            raise RequestValidationError("The phone no is incorrect")

    def __post_init__(self):
        self.validate_role()
        self.validate_email()
        self.validate_matric_no()
        self.validate_image()
        self.validate_phone()

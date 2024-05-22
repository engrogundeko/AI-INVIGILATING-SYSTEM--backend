from pydantic import BaseModel
from enum import Enum
from typing import List, BinaryIO
from ..repository import userRespository


class Gender(Enum):
    male = "MALE"
    female = "FEMALE"


class UserRole(Enum):
    student = "STUDENT"
    invigilator = "INVIGILATOR"
    admin = "ADMIN"
    lecturer = "LECTURER"


class User:
    def __init__(
        self,
        first_name: str,
        last_name: str,
        email: str,
        phone: int,
        gender: str,
        address: str,
        role: str,
        password: str = None,
        matric_no: int = None,
        img_path: str = None,
    ) -> None:
        self.first_name = first_name
        self.last_name = last_name
        self.gender = gender
        self.role = role
        self.phone = phone
        self.address = address
        self.email = email
        self.password = password
        self.matric_no = matric_no
        self.img_path = img_path

    # @property
    # def _id(self):
    #     return str(self.id)

    def render(self):
        return {
            # "id": self._id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "email": self.email,
            "gender": self.gender,
            "phone": self.phone,
            "address": self.address,
            "role": self.role,
            "matric": self.matric_no,
        }
        
    def insert_one(self):
        return userRespository.insert_one(self.render())

    @property
    def full_name(self):
        return self.first_name + self.last_name

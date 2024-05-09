from typing import BinaryIO, List
from pydantic import BaseModel

from .model import Gender, UserRole

class StudentInSchema(BaseModel):
    first_name: str
    last_name: str
    matric_no: int
    email: str
    phone: str
    address: str
    gender: Gender
    role: UserRole

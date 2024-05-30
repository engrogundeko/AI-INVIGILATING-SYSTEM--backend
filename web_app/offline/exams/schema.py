from typing import List
from bson import ObjectId
from pydantic import BaseModel
from datetime import date, time, datetime


class VideoSchema(BaseModel):
    image: str


class CreateExamSchema(BaseModel):
    exam_id: int
    students: List[int]
    status: str


class GetExamAttendance(BaseModel):
    exam_id: int


class AttendanceSchema(BaseModel):
    image: str

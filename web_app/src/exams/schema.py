from typing import List
from bson import ObjectId
from pydantic import BaseModel
from datetime import date, time, datetime


class VideoSchema(BaseModel):
    image: str


class CreateExamSchema(BaseModel):
    exam_id: ObjectId
    students: List[ObjectId]
    status: str


class GetExamAttendance(BaseModel):
    exam_id: ObjectId


class AttendanceSchema(BaseModel):
    image: str

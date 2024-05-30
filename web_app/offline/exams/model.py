from datetime import datetime, date, time
from typing import List
from bson import ObjectId
from pydantic import BaseModel


class Exam(BaseModel):
    id: int
    name: str
    date: date
    start_time: time
    end_time: time
    room_id: int
    course_id: int
    invilgilator_id: int


# class VideoRecording(BaseModel):
#     id: int
#     exam_id: int
#     timestamp: datetime
#     file_path = str
#     duration: str
#     resolution: str


class SuspicionReport(BaseModel):
    id: int
    exam_id: int
    student_id: int
    timestamp: datetime


class ExamAttendance(BaseModel):
    status: str
    exam_id: str
    student_id: int
    status: str


class ExamRegistration(BaseModel):
    id: int
    exam_id: Exam
    student_id: int
    status: str


class Course(BaseModel):
    id: int
    name: str
    department: str
    faculty: str


class Room(BaseModel):
    id: int
    name: str
    capacity: int
    building: str

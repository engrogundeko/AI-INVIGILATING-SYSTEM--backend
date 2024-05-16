from datetime import datetime, date, time
from typing import List
from bson import ObjectId
from pydantic import BaseModel


class Exam(BaseModel):
    id: ObjectId
    name: str
    date: date
    start_time: time
    end_time: time
    room_id: ObjectId
    course_id: ObjectId
    invilgilator_id: ObjectId


class VideoRecording(BaseModel):
    id: ObjectId
    exam_id: ObjectId
    timestamp: datetime
    file_path = str
    duration: str
    resolution: str


class SuspicionReport(BaseModel):
    id: ObjectId
    exam_id: ObjectId
    student_id: ObjectId
    timestamp: datetime


class ExamAttendance(BaseModel):
    status: str
    exam_id: str
    student_id: ObjectId
    status: str


class ExamRegistration(BaseModel):
    id: ObjectId
    exam_id: Exam
    student_id: ObjectId
    status: str


class Course(BaseModel):
    id: ObjectId
    name: str
    department: str
    faculty: str


class Room(BaseModel):
    id: ObjectId
    name: str
    capacity: int
    building: str

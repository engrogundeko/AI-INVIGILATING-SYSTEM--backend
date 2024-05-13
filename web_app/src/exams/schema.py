from pydantic import BaseModel
from datetime import date, time, datetime
from bson import ObjectId


class VideoSchema(BaseModel):
    image: str


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
    id: ObjectId
    exam_id: ObjectId
    student_id: ObjectId
    attendace_status: str  # absent present


class ExamRegistration(BaseModel):
    id: ObjectId
    exam_id: ObjectId
    student_id: ObjectId
    status: str  # confirmed reconfirmed cancelled


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


class Exam(BaseModel):
    id: ObjectId
    name: str
    date: date
    start_time: time
    end_time: time
    room_id: ObjectId
    course_id: ObjectId
    invilgilator_id: ObjectId

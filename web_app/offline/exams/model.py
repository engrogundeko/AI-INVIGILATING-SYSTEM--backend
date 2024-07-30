from datetime import datetime, date, time
from typing import List, Tuple
from ..base import Model


class Exam(Model):
    name: str
    date: date
    start_time: time
    end_time: time
    room_id: int
    course_id: int
    invilgilator_id: int


# class VideoRecording(BaseModel):
#     exam_id: int
#     timestamp: datetime
#     file_path = str
#     duration: str
#     resolution: str


class ExamLocation(Model):
    exam_id: str
    locations: List[dict]


class SuspiciousReport(Model):
    frame_id: int
    exam_id: str
    student_id: int
    timestamp: datetime
    coordinates: Tuple[float, float]
    confidence_score: float
    pixel_changes: float
    image_path: str


class ExamAttendance(Model):
    status: str
    exam_id: str
    student_id: str
    coordinate: Tuple[float, float, float, float]


class ExamRegistration(Model):
    exam_id: str
    student_id: str
    # status: str


class Course(Model):
    name: str
    code: str
    department: str
    faculty: str


class Room(Model):
    name: str
    capacity: int
    building: str


class Exam(Model):
    name: str
    date: datetime
    start_time: str
    end_time: str
    room_id: str
    course_id: str
    invilgilator_id: List[str] | str

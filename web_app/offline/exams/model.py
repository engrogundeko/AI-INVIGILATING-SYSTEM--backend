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


class VideoRecording(Model):
    exam_id: int
    timestamp: datetime
    file_path: str
    duration: str
    resolution: str


class ExamLocation(Model):
    exam_id: str
    locations: List[dict]


class SuspiciousReport(Model):
    exam_id: str
    student_id: int
    timestamp: datetime
    all_cheating_scores: List[float]
    coordinates: Tuple[float, float]
    average_cheat: str
    image_paths: str


class ExamAttendance(Model):
    status: str
    exam_id: str
    student_id: str
    coordinate: Tuple[float, float, float, float]


class ExamRegistration(Model):
    exam_id: str
    student_id: str
    # status: str

class FacultyResponse(Model):
    name: str

class DepartmentResponse(Model):
    name: str

class Course(Model):
    name: str
    code: str
    department: str
    faculty: str
    session: str


class Room(Model):
    name: str
    capacity: int
    building: str


class Exam(Model):
    name: str
    date: str
    start_time: str
    end_time: str
    room_id: str
    course_id: str
    # invilgilator_id: List[str] | str

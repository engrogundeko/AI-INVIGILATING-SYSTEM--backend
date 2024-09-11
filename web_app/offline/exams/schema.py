from typing import List, Tuple
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, field_validator
from datetime import date, datetime, time

from ..repository import examRespository, courseRespository

class SuspiciousReportSchema(BaseModel):

    exam_id: str
    student_id: int
    timestamp: List[datetime]
    all_cheating_scores: List[float]
    coordinates: Tuple[float, float, float, float]
    average_cheat: str
    image_paths: List[str]

class VideoRecordingSchema(BaseModel):
    exam_id: str
    timestamp: datetime
    file_path: str
    duration: float
    resolution: Tuple[int, int]
class ExamAttendance(BaseModel):
    status: str
    exam_id: str
    student_id: str
    coordinate: Tuple[float, float, float, float]

class Location(BaseModel):
    student_id: str
    coordinate: Tuple[float, float, float, float]
    
class ExamLocation(BaseModel):
    exam_id: str
    locations: List[dict]
    undetected_faces: List[str]
    unrecognised_faces: List[str]
    
class CreateCourseShema(BaseModel):
    name: str
    code: str
    department: str
    faculty: str
    session: str


class VideoSchema(BaseModel):
    image: str


class RegisterExamSchema(BaseModel):
    exam_id: str
    student_id: str


class CreateExamSchema(BaseModel):
    name: str
    date: datetime
    start_time: str
    end_time: str
    room_id: str
    session: str
    course_code: str
    # invilgilator_id: List[str] | str

    @field_validator("course_code", mode="before")
    def vallidate_course_code(cls, v):
        course = courseRespository.find_one({"code": cls.course_code})
        if course is None:
            raise RequestValidationError("Course does not exist")
        
        
    @field_validator("date", mode="before")
    def convert_date(cls, v):
        if isinstance(v, date):
            return datetime.combine(v, datetime.min.time())
        return v

    @field_validator("start_time", mode="before")
    def convert_start_time(cls, v):
        if isinstance(v, time):
            return v.strftime("%H:%M:%S")
        return v

    @field_validator("end_time", mode="before")
    def convert_end_time(cls, v):
        if isinstance(v, time):
            return v.strftime("%H:%M:%S")
        return v


class CreateRoom(BaseModel):
    name: str
    capacity: int
    building: str


class GetExamAttendance(BaseModel):
    exam_id: int


class AttendanceSchema(BaseModel):
    image: str


class StreamSchema(BaseModel):
    exam_id: str

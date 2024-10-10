from typing import List, Tuple, Dict
from fastapi.exceptions import RequestValidationError
import numpy as np
from pydantic import BaseModel, field_validator
from datetime import date, datetime, time

from ..repository import examRespository, courseRespository


class ImageGraphs:
    average_cheat: str
    rated_cheat: str
    confusion_matrix: str
    performance: str
    anomalies: List[str]
    activities: List[str]


class ExamMetrics(BaseModel):
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    accuracy: float
    precision: float
    f1_score: float
    recall: float
    sample_images: List[str]


class SuspiciousReportSchema(BaseModel):

    exam_id: str
    student_id: int
    timestamp: List[datetime]
    all_cheating_scores: List[float]
    coordinates: Tuple[float, float, float, float]
    average_cheat: str
    image_paths: List[str]


class ReportAnalysisSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    student_id: int
    average_cheating: float
    all_cheating_score: List[float]
    mean_outliers: float
    contamination_level: str
    n_iqr_anomalies: int
    n_anomalies: int
    score: str
    magnitude: List[float]
    cheat_timestamp: List[datetime]
    anomalies_idx: List[int]
    comment: str  
    confidence: float

class ReportSchema(BaseModel):
    exam_id: str
    anomaly_analysis: List[dict]
    start_time: datetime
    end_time: datetime
    n_suspicious: int
    exam_metrics: ExamMetrics | None


# class OpticalFlowSchema(Base)


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

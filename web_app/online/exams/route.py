from .schema import CreateExamSchema, GetExamAttendance
from ..repository import (
    examRespository,
    examRegistrationRespository,
    courseRespository,
    userRespository,
    examAttedanceRespository,
)


from fastapi import APIRouter, Path, Query
from fastapi.responses import JSONResponse

router = APIRouter(tags=["Exams"], prefix="/exams")


@router.post("")
def create_exam(payload: CreateExamSchema):
    return examRespository.insert_one(payload.__dict__)


@router.get("/")
def get_exam(exam_name: str = Query(...)):
    if exam_name:
        return examRespository.find_one({"name": exam_name})
    return examRespository.find_many()


@router.post("/course")
def create_course(payload: CreateExamSchema):
    return courseRespository.insert_one(payload.__dict__)


@router.post("/register")
def create_exam(payload: CreateExamSchema):
    return examRegistrationRespository.insert_one(payload.__dict__)


@router.get("/attendance")
def get_attendance(payload: GetExamAttendance):
    exam = examRespository.find_one({"exam_id": payload.exam_id})
    attendance = examAttedanceRespository.find_one({"_id": exam["attendace_id"]})
    return JSONResponse(attendance)

from datetime import date, datetime, time
from typing import List


from fastapi.exceptions import RequestValidationError
from .utils import session_validator
from .model import (
    Course,
    Room,
    Exam,
    FacultyResponse,
    DepartmentResponse,
)
from .schema import (
    CreateRoom,
)
from ..config import AIConstant

from ..repository import (
    examRespository,
    courseRespository,
    roomRespository,
    sessionRepository,
    facultyRepository,
    departmentRepository,
)
from .sessions import (
    validate_faculty,
    validate_rooms,
    validate_session,
    validate_department,
    get_all_departments,
    get_all_faculties,
    get_all_rooms,
    # get_all_sessions
)
from fastapi import APIRouter, Depends, Query

router = APIRouter(tags=["Exams"], prefix="/exams")


@router.post("/session", response_model=FacultyResponse)
def create_session(session: str):
    valid_session = session_validator(session)
    if not valid_session:
        raise RequestValidationError("Invalid Session")
    return sessionRepository.insert_one(dict(name=session))


@router.post("/department", response_model=DepartmentResponse)
def create_department(department: str):
    return departmentRepository.insert_one(dict(name=department.upper()))


@router.post("/faculty", response_model=FacultyResponse)
def create_faculty(faculty: str):
    return facultyRepository.insert_one(dict(name=faculty.upper()))


@router.post("", response_model=Exam)
def create_exam(
    name: str,
    date: date,
    start_time: time,
    end_time: time,
    course_code: str,
    venue: str = Depends(validate_rooms),
    session: str = Depends(validate_session),
):
    exam = examRespository.find_one({"name": name, "session": str(session)})
    if exam is not None:
        raise RequestValidationError("Exam already exist")
    
    course = courseRespository.find_one({"code": course_code.upper()})
    if course is None:
        raise RequestValidationError("Course does not exist")

    venue = roomRespository.find_one({"name": venue.upper()})
    session = sessionRepository.find_one({"name": session})
    return examRespository.insert_one(
        dict(
            name=name.upper(),
            date=date.strftime("%d/%m/%Y"),
            start_time=start_time.strftime(AIConstant.time_format),
            end_time=end_time.strftime(AIConstant.time_format),
            room_id=str(venue["_id"]),
            session=str(session["_id"]),
            course_id=str(course["_id"]),
        )
    )


@router.get("", response_model=List[Exam])
def get_exam():
    return examRespository.find_many()


@router.get("/")
def get_exam(exam_name: str = Query(...)):
    if exam_name:
        return examRespository.find_one({"name": exam_name})
    return examRespository.find_many()


@router.post("/room", response_model=Room)
def create_room(payload: CreateRoom):
    return roomRespository.insert_one(payload.__dict__)


@router.post("/course", response_model=Course)
def create_course(
    name: str,
    code: str,
    faculty: str = Depends(validate_faculty),
    session: str = Depends(validate_session),
    department: str = Depends(validate_department),
):
    session = sessionRepository.find_one({"name": session})["_id"]
    department_ = departmentRepository.find_one({"name": department})
    faculty_ = facultyRepository.find_one({"name": faculty})
    
    course = courseRespository.find_one({"code": code, "session": str(session)})
    if course:
        raise RequestValidationError("Course already exist")
    
    return courseRespository.insert_one(
        dict(
            name=name.upper(),
            code=code,
            department=str(department_["_id"]),
            faculty=str(faculty_["_id"]),
            session=str(session),
        )
    )


@router.get("/course", response_model=List[Course])
def get_course():
    return courseRespository.find_many()

@router.get("/room", response_model=List[DepartmentResponse])
def get_rooms():
    return get_all_rooms()

@router.get("/department", response_model=List[DepartmentResponse])
def get_department():
    return get_all_departments()

@router.get("/faculty", response_model=List[FacultyResponse])
def get_faculty():
    return get_all_faculties()


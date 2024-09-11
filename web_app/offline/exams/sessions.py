from typing import List
from ..repository import (
    roomRespository,
    sessionRepository,
    facultyRepository,
    departmentRepository,
)

from fastapi import Depends, HTTPException


def get_all_sessions():
    # Query the database to fetch available session names
    return sessionRepository.find({}, {"name": 1})


def validate_session(session: str, sessions: List[str] = Depends(get_all_sessions)):
    """
    Dependency function to validate if the session exists in the list of sessions.
    """
    available_sessions = [s["name"] for s in sessions]
    if session not in available_sessions:
        raise HTTPException(status_code=400, detail="Invalid session.")
    return session


def get_all_faculties():
    # Query the database to fetch available session names
    return facultyRepository.find({}, {"name": 1})


def validate_faculty(faculty: str, faculties: List[str] = Depends(get_all_faculties)):
    """
    Dependency function to validate if the session exists in the list of sessions.
    """
    available_faculties = [s["name"] for s in faculties]
    if faculty not in available_faculties:
        print("---------------------------")
        raise HTTPException(status_code=400, detail="Invalid faculty.")
    return faculty


def get_all_departments():
    # Query the database to fetch available session names
    return departmentRepository.find({}, {"name": 1})


def validate_department(department: str, departments: List[str] = Depends(get_all_departments)):
    """
    Dependency function to validate if the session exists in the list of sessions.
    """
    avaliable_departments = [s["name"] for s in departments]
    if department not in avaliable_departments:
        raise HTTPException(status_code=400, detail="Invalid department.")
    return department


# def validate_session(
#     department: str, departments: List[str] = Depends(get_all_departments)
# ):
#     """
#     Dependency function to validate if the session exists in the list of sessions.
#     """
#     avalable_departments = [d["name"] for d in departments]
#     if department not in avalable_departments:
#         raise HTTPException(status_code=400, detail="Invalid Department.")
#     return department


def get_all_rooms():
    # Query the database to fetch available session names
    return roomRespository.find({}, {"name": 1})


def validate_rooms(room: str, rooms: List[str] = Depends(get_all_rooms)):
    """
    Dependency function to validate if the session exists in the list of sessions.
    """
    avalable_rooms = [r["name"] for r in rooms]
    if room not in avalable_rooms:
        raise HTTPException(status_code=400, detail="Invalid Department.")
    return room

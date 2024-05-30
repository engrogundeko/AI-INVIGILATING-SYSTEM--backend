import base64
from bson import ObjectId
import cv2
import numpy as np

from ..algorithms.data import DataStorage
from ..repository import (
    userRespository,
    examRespository,
    examRegistrationRespository,
    examAttedanceRespository,
)
from .model import ExamRegistration, Exam
from deepface import DeepFace
from pathlib import Path


class AttendanceSystem:
    def __init__(self):
        self.records = DataStorage()


def take_attendance(image: str, exam_id: int):
    base64_data = image.split(",")[1]
    image_bytes = base64.b64decode(base64_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    faces = DeepFace.extract_faces(frame)
    for face in faces:
        exam_reg = examRegistrationRespository.find_many({"exam_id": exam_id})
        for exam in exam_reg:
            student = userRespository.find_one({"_id": exam["student_id"]})
            img_path = Path(student["img_path"])
            result = DeepFace.verify(face["face"], img_path)
            if result["verified"]:
                att = {
                    "exam_id": exam_id,
                    "student_id": student["_id"],
                    "status": "PRESENT",
                }
                attendance = examAttedanceRespository.insert_one(att)
                return attendance.inserted_id


class ExamService:
    pass


# return generate_frame()

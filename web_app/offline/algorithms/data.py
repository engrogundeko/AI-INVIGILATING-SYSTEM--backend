from ..repository import examRespository, userRespository, suspicionReportRespository
from pathlib import Path
from deepface import DeepFace
from datetime import datetime


class DataStorage:
    def __init__(self):
        self.data = []

    def verify_student_identity(self, face, course):
        exam = examRespository.find_one({"exam_id": course})
        students = userRespository.find_many({"_id": exam["students"]})
        for student in students:
            img_path = student["img_path"]
            path = Path(img_path)
            result = DeepFace.verify(path, face)
            if result["verified"]:
                return True

    def log_unrecognized_face(self, face, course):
        return suspicionReportRespository.insert_one(
            {
                "exam_id": course,
                "description": "Could not identified face during exams",
                "image": face,
                "date": datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
                "student": None,
            }
        )

    def log_gesture(self, gesture):
        # Log detected gestures for further analysis
        pass

    def log_anomaly(self, anomaly):
        # Log detected anomalies for further analysis
        pass

    def log_object(self, obj):
        # Log recognized objects for further analysis
        pass

    def analyze_data(self):
        # Analyze stored data to identify suspicious activities
        pass

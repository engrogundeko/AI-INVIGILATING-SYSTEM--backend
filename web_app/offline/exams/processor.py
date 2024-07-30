import os
import cv2

import httpx
from deepface import DeepFace
import multiprocessing as mp

from ..repository import MongoDBRepository
from ..repository import userRespository, examRespository, suspiciousReportRespository
from ..config import WKR_DIR, VIDEO_RECORD_PATH, IMAGE_RECORD_PATH
from .model import SuspiciousReport, ExamAttendance, Exam


input_queue = mp.Queue()
result_queue = mp.Queue()
event = mp.Event()



class MultiProcessor:
    def __init__(self, exam_id):
        self.process_queue = mp.Queue()
        self.facial_recognition = mp.Process(
            target=detect_face,
            args=(exam_id, input_queue, self.process_queue),
        )
        # self.detect_cheating = mp.Process(target=ai)
        self.db_process = mp.Process(target=save_to_db, args=(self.process_queue,))

    def start(self):
        print("Starting processes...")
        # self.detect_cheating.start()
        self.facial_recognition.start()
        self.db_process.start()

    def join(self):
        print("Joining processes...")
        # self.detect_cheating.join()
        self.facial_recognition.join()
        self.db_process.join()

    def stop(self):
        input_queue.put(None)
        result_queue.put(None)
        self.process_queue.put(None)


def detect_face(exam_id, facial_queue: mp.Queue, db_queue: mp.Queue):
    print("Starting detect_face process...")
    users = userRespository.find_many()
    while True:
        frame = facial_queue.get()
        if frame is None:  # Stop signal
            break
        print(f"Processing frame in detect_face: {frame}")

        try:
            faces = DeepFace.extract_faces(frame)
            for face in faces:
                for user in users:
                    recognize = DeepFace.verify(face, user["image_path"])
                    if recognize["verified"]:
                        db_queue.put(
                            {
                                "frame": frame,
                                # "timestamp": time.time(),
                                "user": user,
                                "exam_id": exam_id,
                            }
                        )
                        event.set()
                        break
        except Exception as e:
            print(f"Error in face detection: {e}")


def save_to_db(process_queue: mp.Queue):
    print("Starting save_to_db process...")
    while True:
        db = process_queue.get()
        if db is None:
            break
        print(f"Saving to DB: {db}")
        frame, timestamp, user, exam_id = db.values()
        path = os.path.join(str(exam_id), f"{user['matric_no']}_{timestamp}.jpg")
        cv2.imwrite(path, frame)
        data = {
            "image_path": path,
            "exam_id": exam_id,
            "timestamp": timestamp,
            "student_id": user["_id"],
        }
        suspicious = SuspiciousReport(data)
        suspiciousReportRespository.insert_one(suspicious.__dict__)

        url = "https://mssn-mailer.onrender.com/api{api_secret}/send_mail{api_key}"

        with httpx.Client() as client:
            print(f"Sending data to external service: {db}")
            response = client.post(url, data=db)
            print(response.json())


import json
from pathlib import Path
from bson import ObjectId

from ..config import AIConstant, YOLOV8_MODEL
from ..repository import (
    examRegistrationRespository,
    userRespository,
    examAttedanceRespository,
)

import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from apscheduler.schedulers.background import BackgroundScheduler


# model = YOLO(YOLOV8_MODEL)
img_db = Path("media/profilephotos")
scheduler = BackgroundScheduler()


def label_box(frame, detection):
    bounding_box = detection.boxes
    bounding_box = bounding_box.xyxy.tolist()[0]
    x1, x2, y1, y2 = bounding_box
    return cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


def generate_frame(exam_id):
    camera = cv2.VideoCapture(0)
    try:
        while True:
            # Read frame from camera
            success, frame = camera.read()
            if not success:
                break

            # Perform object detection
            detections = model.track(frame, persist=True)

            frame = detections[0].plot()

            # schedule_live_recognition(results, exam_id)
            # scheduler.add_job(
            #     schedule_live_recognition,
            #     "interval",
            #     id=1,
            #     minutes=10,
            #     args=[results, exam_id],
            # )

            ret, buffer = cv2.imencode(".webp", frame)
            if not ret:
                break

            # Yield encoded frame as bytes
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
    finally:

        # Release camera when done
        scheduler.remove_job(job_id=1)
        camera.release()


def anomaly_detection():
    pass


def gesture_detection():
    pass


def schedule_live_recognition(detections, exam_id: ObjectId):
    for detection in detections:
        boxes = detection.tojson(detection)
        boxes_json = json.loads(boxes)
        for obj in boxes_json:
            if obj["class"] == AIConstant.PERSON_CLASS:
                if obj["confidence"] > AIConstant.PROB_ALLOWANCE:
                    try:
                        result = live_recognition(detection.orig_img, exam_id)
                    except ValueError as e:
                        pass


def live_recognition(nparr, exam_id: ObjectId) -> list:
    results = []
    exam = examRegistrationRespository.find_one({"id": exam_id})
    for student_id in exam["students"]:
        student = userRespository.find_one({"id": student_id})
        path = Path(student["image_path"])
        if path.exists:
            result = DeepFace.verify(nparr, path, detector_backend="mtcnn")
            results.append({"student": student["matric_no"], "result": result})
    return results


class AIInvigilatingSystem: ...

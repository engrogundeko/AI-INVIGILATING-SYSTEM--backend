import json
import numpy as np
from pathlib import Path
from ..repository import examRespository
from apscheduler.schedulers.background import BackgroundScheduler
import cv2
from ultralytics import YOLO
from deepface import DeepFace

img_db = Path("media/profilephotos")
model = YOLO("yolov8n.pt")
scheduler = BackgroundScheduler()


def label_box(frame, detection):
    bounding_box = detection.boxes
    bounding_box = bounding_box.xyxy.tolist()[0]
    x1, x2, y1, y2 = bounding_box
    return cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


def generate_frame():
    camera = cv2.VideoCapture(0)
    try:
        while True:
            # Read frame from camera
            success, frame = camera.read()
            if not success:
                break

            # Perform object detection
            results = model.track(frame, persist=True)

            frame = results[0].plot()
            schedule_live_recognition(results)
            # job = scheduler.add_job(
            #     schedule_live_recognition, "interval", minutes=10, args=[results]
            # )

            ret, buffer = cv2.imencode(".webp", frame)
            if not ret:
                break

            # Yield encoded frame as bytes
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
    finally:

        # Release camera when done
        # job.remove()
        camera.release()


def schedule_live_recognition(detections):
    for detection in detections:
        boxes = detection.tojson(detection)
        boxes_json = json.loads(boxes)
        for obj in boxes_json:
            if obj["class"] == 0:
                if obj["confidence"] > 0.6:
                    try:
                        result = live_recognition(detection.orig_img)
                        print(result)
                    except ValueError as e:
                        pass


def live_recognition(nparr) -> bool:
    results = []
    students = examRespository.find_one({"course": ...})
    for student in students:
        path = Path(student["image_path"])
        result = DeepFace.verify(nparr, path)
        results.append({"student": student["matric_no"], "result": result})
    return results

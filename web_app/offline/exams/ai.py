import base64
from datetime import datetime
import os
import time
import math
from pathlib import Path
from threading import Thread
from queue import Empty, Queue

from bson import ObjectId
# from fastapi.encoders import jsonable_encoder

from .preprocessor import PreProcessor
from ..config import (
    IMAGE_RECORD_PATH, 
    VIDEO_RECORD_PATH, 
    AIConstant, 
    AIS_MODEL, 
    UNRECOGNISED_FACE, 
    UNDETECTED_FACE)

from ..repository import (
    suspiciousReportRespository,
    userRespository,
    examRegistrationRespository,
    examAttedanceRespository,
    examLocationRepo,
)
from .schema import ExamLocation, ExamAttendance, Location, SuspiciousReportSchema
from ..utils import image_utils
from .model import SuspiciousReport
from .email import EmailHandler

import cv2
import anyio
import anyio.to_process
import anyio.to_thread
from deepface import DeepFace
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io

def save_temp_image(image, filename="temp_image.jpg"):
    return AIConstant.save_image(image, filename)

def take_attendance(image, exam_id):
    file_content = image.file.read()

    # Convert file content to an image
    img = Image.open(io.BytesIO(file_content))
    arr = np.array(img)
    student_locations = []
    undetected_faces = []
    unrecognised_faces = []

    model = YOLO(AIS_MODEL)
    img, _ = image_utils.load_image(arr)
    results = model.predict(img)
    for re in results:
        for bbox in re.boxes.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            cropped_frame = img[y1:y2, x1:x2]
            exam_reg = examRegistrationRespository.find({"exam_id": exam_id})
            for exam in exam_reg:
                student_id = exam["student_id"]
                student = userRespository.find_one({"_id": ObjectId(student_id)})
                img_path = rf"{student["image"]}"
                is_present = examAttedanceRespository.find_one({"student_id": student_id, "exam_id": exam_id})
    
                if is_present is not None and is_present["status"] == "PRESENT":
                    continue
                try:
                    result = DeepFace.verify(img1_path=cropped_frame, img2_path=img_path, detector_backend="yolov8")
                except ValueError:
                    filename = os.path.join(UNRECOGNISED_FACE, datetime.now().strftime("%M:%S"))
                    img_path = AIConstant.save_image(cropped_frame, filename)
                    unrecognised_faces.append(img_path)
                    continue
                except Exception:
                    filename = os.path.join(UNDETECTED_FACE, datetime.now().strftime("%M:%S"))
                    img_path = AIConstant.save_image(cropped_frame, filename)
                    undetected_faces.append(img_path)
                    continue
                
                if result["verified"]:
                    student_locations.append(
                        Location(student_id=student_id, coordinate=(x1, y1, x2, y2)).__dict__)
                    attendance = ExamAttendance(
                        status="PRESENT",
                        exam_id=exam_id,
                        student_id=student_id,
                        coordinate=(x1, y1, x2, y2),
                    )
                    examAttedanceRespository.insert_one(attendance.__dict__)
                    
    exam = ExamLocation(
        exam_id=exam_id, 
        locations=student_locations, 
        unrecognised_faces=unrecognised_faces, 
        undetected_faces=undetected_faces
        )
    
    examLocationRepo.insert_one(exam.__dict__)
    return list(examAttedanceRespository.find({"exam_id": exam_id}))


class AIInvigilatingSystem:
    IMAGE_SAVE_DIR: str = IMAGE_RECORD_PATH
    VIDEO_SAVE_DIR: str = VIDEO_RECORD_PATH

    def __init__(
        self,
        video_path: str,
        model_path: str,
        output_path: str,
        camera: int | str = None,
        tracker_config: str = None,
        conf_score: float | int = 0.6,
        max_magnitude: float | int = 3.2,
    ):
        self.cheating_log = []
        self.student_locations = {}
        self.threshold = 30
        self.new_width = 1168
        self.new_height = 656
        self.frame_counter = 0
        self.video_path = video_path
        self.output_path = output_path
        self.tracker_config = tracker_config
        self.conf_score = float(conf_score)
        self.max_magnitude = float(max_magnitude)

        self.email = EmailHandler()
        self.model = YOLO(model_path)
        self.preprocesor = PreProcessor()
        self.frame_queue = Queue(maxsize=10)
        self.result_queue = Queue()
        self.cap = cv2.VideoCapture(camera if camera else video_path)

    @property
    def out(self):
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        return cv2.VideoWriter(
            output_path, fourcc, 20.0, (self.new_width, self.new_height)
        )

    @property
    def farneback_params(self) -> dict:
        return dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

    @property
    def save_to_db(self):
        for log in self.cheating_log:
            suspicious = SuspiciousReportSchema(**log)
            suspiciousReportRespository.insert_one(suspicious.__dict__)
            
    def __load_image(self, image: str | np.ndarray):
        return image_utils.load_image(image)

    def _initialize_location(self, exam_id):
        self.student_locations = examLocationRepo.find_one({"exam_id": exam_id})
        # print(self.student_locations)
        
    async def send_email(self, data: dict):
        return await self.email.send(data)

    def _match(self, x1, y1, x2, y2):
        distance = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
        result = distance < self.threshold
        return result

    def _get_center(self, x1, y1, x2, y2):
        Xc = (x1 + x2) / 2
        Yc = (y1 + y2) / 2
        return Xc, Yc

    def _get_student_id(self, coordinate):
        # [{"student_id", "coordinate"}]
        student_locations = self.student_locations.get("locations")
        for location in student_locations:
            # print(coordinate)
            # print("real",location["coordinate"])
            student_id = location["student_id"]
            # c1, z1, c2, z2 = coordinates
            Xc1, Yc1 = self._get_center(*coordinate)
            # c11, z11, c22, z22 = student["coordinate"]
            Xc2, Yc2 = self._get_center(*location["coordinate"])
            is_match = self._match(Xc1, Yc1, Xc2, Yc2)
            if is_match:
                return student_id
            # raise an exception indicating no match which can be
            # caught and handled or logged to show that

    def _get_unique_filename(self, base_filename: str, extension: str) -> str:
        counter = 1
        unique_filename = f"{base_filename}.{extension}"
        while os.path.exists(os.path.join(self.IMAGE_SAVE_DIR, unique_filename)):
            unique_filename = f"{base_filename}_{counter:04d}.{extension}"
            counter += 1
        return unique_filename

    def save_image(self, frame, student_id) -> str:
        file_name = self._get_unique_filename(student_id, "jpg")
        image_path = os.path.join(self.IMAGE_SAVE_DIR, file_name)
        cv2.imwrite(image_path, frame)
        return image_path

    async def process_frame(self, prev_gray, frame, frame_time, exam_id):
        frame = cv2.resize(frame, (self.new_width, self.new_height))
        noise = self.preprocesor.apply_noise_reduction(frame)
        edge = self.preprocesor.apply_edge_detection(noise)
        his = self.preprocesor.apply_histogram_equalization(edge)

        clax_frame = self.preprocesor.apply_clahe(his)
        thres_frame = self.preprocesor.apply_adaptive_threshold(clax_frame)
        gaus = self.preprocesor.apply_gaussian_filter(thres_frame)
        gray = cv2.cvtColor(gaus, cv2.COLOR_BGR2GRAY)

        results = self.model.predict(frame, conf=0.15, iou=0.4)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, **self.farneback_params
        )

        for result in results:
            for bbox in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, bbox)
                roi_flow = flow[y1:y2, x1:x2]

                magnitude, _ = cv2.cartToPolar(roi_flow[..., 0], roi_flow[..., 1])
                avg_magnitude = np.mean(magnitude)

                confidence_score = min(max(avg_magnitude / self.max_magnitude, 0), 1)

                if confidence_score > self.conf_score:
                    coordinates = (x1, y1, x2, y2)
                    cropped_image = frame[y1:y2, x1:x2]
                    label = "Cheating"
                    colour = (0, 0, 255)
                    student_id = self._get_student_id(coordinates)
                    if student_id is not None:
                        path = self.save_image(cropped_image, student_id)
                    
                    #     await self.send_email(student_id, cropped_image, exam_id)

                        log_entry = {
                            "exam_id": exam_id,
                            "student_id": student_id,
                            "frame_id": self.frame_counter,
                            "timestamp": frame_time,
                            "coordinates": coordinates,
                            # "match_threshold": match_threshold,
                            "confidence_score": float(confidence_score),
                            "pixel_changes": float(avg_magnitude),
                            "image_path": path,
                        }
                        self.cheating_log.append(log_entry)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(
                        frame,
                        f"{label}: {confidence_score:.2f}%",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        colour,
                        2,
                    )

        return gray, frame

    async def frame_worker(self):
        prev_gray = None
        frame_counter = 0
        while True:
            frame, frame_time, exam_id = self.frame_queue.get()
            if frame is None:
                break
            frame_counter += 1
            if frame_counter % 5 == 0:  # Process every 5th frame for better tracking
                if prev_gray is None:
                    frame_resized = cv2.resize(frame, (self.new_width, self.new_height))
                    _frame = self.preprocesor.apply_edge_detection(frame_resized)
                    _frame = self.preprocesor.apply_noise_reduction(_frame)
                    _frame = self.preprocesor.apply_histogram_equalization(_frame)

                    frame_resized = self.preprocesor.apply_clahe(_frame)
                    thres_frame = self.preprocesor.apply_adaptive_threshold(
                        frame_resized
                    )
                    gaus = self.preprocesor.apply_gaussian_filter(thres_frame)
                    prev_gray = cv2.cvtColor(gaus, cv2.COLOR_BGR2GRAY)
                    processed_frame = frame_resized

                else:
                    prev_gray, processed_frame = await self.process_frame(
                        prev_gray, frame, frame_time, exam_id
                    )
                self.result_queue.put(processed_frame)

    async def __call__(self, exam_id):
        self._initialize_location(exam_id)
        num_workers = 16
        threads = []
        for _ in range(num_workers):
            t = Thread(target=anyio.run, args=(self.frame_worker,))
            t.start()
            threads.append(t)

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    self.save_to_db
                    break
                frame_time = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.gmtime(time.time())
                )
                self.frame_queue.put((frame, frame_time, exam_id))

                try:
                    processed_frame = self.result_queue.get_nowait()
                    # out.write(processed_frame)

                    cv2.imshow("Movement Detection", processed_frame)
                    self.result_queue.task_done()
                    yield processed_frame
                except Empty:
                    pass

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            self.frame_queue.join()

            for _ in range(num_workers):
                self.frame_queue.put((None, None))
            for t in threads:
                t.join()
            # out.release()
            self.cap.release()
            cv2.destroyAllWindows()


# if __name__ == "__main__":
# def frame_generator():
video_path = (
    r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\media\datasets\frame_classifier\images\test\exchange.mp4"
)
model_path = AIS_MODEL
output_path = "cheating_spoof.avi"
tracker_config = r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\bytetrack.yaml"

ai_invigilating_system = AIInvigilatingSystem(
    video_path=video_path,
    model_path=model_path,
    output_path=output_path,
    tracker_config=tracker_config,
)
# for frame in ai_invigilating_system("dsbkbdskbhdsbhsdbhbhbdsh"):
#     continue

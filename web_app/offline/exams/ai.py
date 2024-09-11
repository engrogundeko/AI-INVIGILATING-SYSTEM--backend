import os
import time
import math
import winsound
from threading import Thread
from queue import Queue
from datetime import datetime
from bson import ObjectId

from .recorder import VideoRecorder
from .preprocessor import PreProcessor
from ..config import (
    IMAGE_RECORD_PATH,
    VIDEO_RECORD_PATH,
    AIS__MODEL,
    AIS_YOLO,
)

from ..repository import (
    suspiciousReportRespository,
    examLocationRepo,
    examRespository,
    studentDataRepo,
)
from .timer import ExamTimer
from .schema import SuspiciousReportSchema
from ..utils import image_utils
from .email import EmailHandler

import cv2
import anyio
import joblib
import anyio.to_process
import anyio.to_thread
from ultralytics import YOLO
import numpy as np


class AIInvigilatingSystem:
    IMAGE_SAVE_DIR: str = IMAGE_RECORD_PATH
    VIDEO_SAVE_DIR: str = VIDEO_RECORD_PATH

    def __init__(
        self,
        video_path: str,
        camera: int = 1,
        record_video: bool = False,
        use_timer: bool = False
    ):
        self.flows = []
        self.cheating_list = []
        self.detected_bboxes = []
        self.student_locations = {}
        self.record_video = record_video

        self.expansion = 50
        self.threshold: int = 0.2
        self.new_width = 1920
        self.new_height = 1080
        self.dis_width = 840
        self.dis_height = 680
        self.alarm_duration = 500
        self.alarm_frequency = 2500
        self.expected_feature_length = 2
        self.high_brightness_factor = 0.5
        self.use_timer = use_timer
        self.stop_thread = False
        # self.low_brightness_factor = 0.1

        self.svm_model = self.load_ais_model()
        self.model = self.load_yolo()

        self.email = EmailHandler()
        self.time = ExamTimer()
        self.preprocesor = PreProcessor()
        self.frame_queue = Queue(maxsize=10)
        self.result_queue = Queue()
        self.cap = cv2.VideoCapture(camera if camera else video_path)

    @property
    def out(self):
        return VideoRecorder(self.VIDEO_SAVE_DIR, self.new_width, self.new_height)

    @property
    def lk_params(self) -> dict:
        return dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
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
    async def _ring_alarm(self):
        await anyio.to_thread.run_sync(
            winsound.Beep, self.alarm_frequency, self.alarm_duration
        )

    @property
    def save_to_db(self):
        self.evaluate_total_result()
        for log in self.cheating_list:
            suspicious = SuspiciousReportSchema(**log)
            suspiciousReportRespository.insert_one(suspicious.__dict__)

    def load_yolo(self):
        return YOLO(AIS_YOLO)

    def load_ais_model(self):
        try:
            model = joblib.load(AIS__MODEL)
            return model
        except Exception as e:
            raise

    def __load_image(self, image: str | np.ndarray):
        return image_utils.load_image(image)

    def _initialize_location(self, exam_id):
        self.student_locations = examLocationRepo.find_one({"exam_id": exam_id})

    async def send_email(self, data: dict):
        return await self.email.send(data)

    def evaluate_total_result(self):
        for cheat in self.cheating_list:
            cheating_score = cheat["all_cheating_scores"]
            score = np.mean(cheating_score) if cheating_score else 0
            score = score * 100
            cheat["average_cheat"] = f"{score:.4f}"

        # with open("cheating_list.json", "w") as json_file:
        #     json.dump(self.cheating_list, json_file, indent=4, cls=NumpyEncoder)

    def evaluate_cheating(self, x1, y1, x2, y2, confidence_score, frame):

        X2c, Y2c = self._get_center(x1, y1, x2, y2)
        for cheat in self.cheating_list:

            X1, Y1, X2, Y2 = map(int, cheat["coordinates"])

            X1c, Y1c = self._get_center(X1, Y1, X2, Y2)
            if self._match(X1c, Y1c, X2c, Y2c):

                id = cheat["student_id"]

                cv2.putText(
                    frame,
                    f"ID:{id}",
                    (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                y1 = max(0, y1 - self.expansion)
                y2 = min(frame.shape[0], y2 + self.expansion)
                x1 = max(0, x1 - self.expansion)
                x2 = min(frame.shape[1], x2 + self.expansion)
                crp_frame = frame[y1:y2, x1:x2]

                score = self.calculate_conf(crp_frame, confidence_score)

                # if score > 0.5:
                cheat["all_cheating_scores"].append(score)
                cheat["timestamp"].append(datetime.now())

                dt = datetime.now()
                dg = dt.strftime("%H_%M_%S")
                filename = f"{id}__{dg}"
                path = self.save_image(crp_frame, filename)
                cheat["image_paths"].append(path)
                # cheat["timestamp"] = dt

                return score
                # return crp_frame

    def _match(self, x1, y1, x2, y2):
        distance = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
        result = distance < self.threshold
        return result

    def _get_center(self, x1, y1, x2, y2):
        Xc = (x1 + x2) / 2
        Yc = (y1 + y2) / 2
        return Xc, Yc

    def _get_student_id(self, coordinate):
        student_locations = self.student_locations.get("locations")
        for location in student_locations:
            student_id = location["student_id"]
            Xc1, Yc1 = self._get_center(*coordinate)
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

    def calculate_conf(self, frame, conf):
        brightness = np.mean(frame)

        normalized_brightness = brightness / 255
        print(normalized_brightness)

        if normalized_brightness > 0.7:
            adjusted_score = conf - (
                self.high_brightness_factor * normalized_brightness
            )
        # elif normalized_brightness < 0.3:
        #     adjusted_score = conf + (
        #         self.low_brightness_factor * (1 - normalized_brightness)
        #     )
        else:
            adjusted_score = conf

        adjusted_score = max(0, min(adjusted_score, 1))

        return adjusted_score

    def save_image(self, frame, student_id) -> str:
        file_name = self._get_unique_filename(student_id, "jpg")
        image_path = os.path.join(self.IMAGE_SAVE_DIR, file_name)
        cv2.imwrite(image_path, frame)
        return image_path

    def extract_features(self, frame, good_new, good_old):
        for bbox in self.detected_bboxes:
            x1, y1, x2, y2 = map(int, bbox)

            within_box = (
                (good_new[:, 0] >= x1)
                & (good_new[:, 0] <= x2)
                & (good_new[:, 1] >= y1)
                & (good_new[:, 1] <= y2)
            )
            moving_points = good_new[within_box]
            old_points = good_old[within_box]

            if len(moving_points) > 0:
                magnitudes = np.linalg.norm(moving_points - old_points, axis=1)
                avg_magnitude = np.mean(magnitudes)
            else:
                avg_magnitude = 0

            object_size = (x2 - x1) * (y2 - y1)

            # Append individual feature values to the list
            X2c, Y2c = self._get_center(x1, y1, x2, y2)
            for cheat in self.cheating_list:

                X1, Y1, X2, Y2 = map(int, cheat["coordinates"])

                X1c, Y1c = self._get_center(X1, Y1, X2, Y2)
                if self._match(X1c, Y1c, X2c, Y2c):

                    cheat["flows"].append(avg_magnitude)
                    if len(cheat["flows"]) == 6:
                        mean_flow = float(np.mean(cheat["flows"]))
                        cheat["flows"] = []

                        features = [
                            mean_flow,
                            object_size,
                        ]

                        if not len(features) == self.expected_feature_length:
                            print(
                                f"Skipping prediction: Expected {self.expected_feature_length} features, but got {len(features)}"
                            )
                        else:

                            f_val = np.array(features).reshape(
                                1, -1
                            )  # Reshape to (1, n_features)
                            label = self.svm_model.predict(f_val)[0]
                            probabilities = self.svm_model.predict_proba(f_val)

                            if label == 1:
                                confidence_score = probabilities[0, 1]
                                score = self.evaluate_cheating(
                                    x1, y1, x2, y2, confidence_score, frame
                                )
                                print("score", confidence_score, score)
                                # score = self.calculate_conf(crop_frame, confidence_score)

                                code = "Suspicious"
                                colour = (0, 0, 255)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                                cv2.putText(
                                    frame,
                                    f"{code}: {score:.2f}%",
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    colour,
                                    2,
                                )

                    break

    def pre_process_frame(
        self,
        frame,
        clahe=True,
        gaussian_filter=True,
        adaptive_threshold=True,
        edge_detection=True,
        noise_reduction=True,
        histogram_equalization=True,
        kernel_size=(5, 5),
        sigma=1,
    ):
        # Convert to grayscale once, to avoid redundant conversion
        processed_frame = frame.copy()  # Start with the original frame

        # Convert to grayscale for certain operations
        if clahe or adaptive_threshold or edge_detection or histogram_equalization:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if clahe:
            clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe_obj.apply(gray)
            processed_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if (
            gaussian_filter or noise_reduction
        ):  # Combine both as they apply Gaussian Blur
            processed_frame = cv2.GaussianBlur(processed_frame, kernel_size, sigma)

        if adaptive_threshold:
            adaptive_thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            processed_frame = cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2BGR)

        if edge_detection:
            edges = cv2.Canny(gray, 100, 200)
            processed_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        if histogram_equalization:
            hist_eq = cv2.equalizeHist(gray)
            processed_frame = cv2.cvtColor(hist_eq, cv2.COLOR_GRAY2BGR)

        return cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)

    def process_frame(self, prev_gray, prev_points, gray, frame):

        # Calculate optical flow using Lucas-Kanade
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_points, None, **self.lk_params
        )

        # Select good points
        good_new = next_points[status == 1]
        good_old = prev_points[status == 1]

        self.extract_features(frame, good_new, good_old)

        return gray, good_new.reshape(-1, 1, 2), frame

    def frame_producer(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if self.stop_thread:
                break
            self.frame_queue.put(frame)
        self.frame_queue.put(None)
        print("Producer thread ended and resources released.")

    def frame_consumer(self, exam_id):
        try:
            prev_gray = None
            prev_points = None
            frame_counter = 0
            fps = 0
            timestamp = datetime.now()
            start_time = time.time()

            exam_name = examRespository.find_one({"_id": ObjectId(exam_id)})["name"]

            if self.record_video:
                recorder = self.out
                record = recorder.out(exam_id)
                
            while True:
                frame = self.frame_queue.get()  
                if frame is None:  # End of the stream
                    self.save_to_db
                    break

                frame_counter += 1

                # Calculate FPS
                current_time = time.time()
                elapsed_time = current_time - start_time
                fps = (frame_counter / elapsed_time) if elapsed_time != 0 else 0

                if self.record_video:
                    record.write(frame)

                frame = cv2.resize(frame, (self.new_width, self.new_height))
                if frame_counter == 1:
                    # start timer
                    if self.use_timer:
                        self.time.start_exam_timer(exam_id)

                    results = studentDataRepo.find_one({"exam_id": exam_id})
                    if results is None:
                        raise ValueError("You have not yet capture students")
                    detections = results["students_data"]

                    for result in detections:
                        bbox = result["coordinates"]
                        box = list(map(int, bbox))
                        self.detected_bboxes.append(box)
                        self.cheating_list.append(
                            {
                                "exam_id": exam_id,
                                "student_id": result["id"],
                                "coordinates": box,
                                "all_cheating_scores": [],
                                "image_paths": [],
                                "flows": [],
                                "timestamp": [],
                            }
                        )
                if frame_counter % 5 == 0:
                    gray_frame = self.pre_process_frame(frame)

                    if prev_gray is None:
                        prev_gray = gray_frame
                        prev_points = cv2.goodFeaturesToTrack(
                            prev_gray,
                            maxCorners=100,
                            qualityLevel=0.3,
                            minDistance=7,
                            blockSize=7,
                        )
           
                    else:
                        prev_gray, prev_points, _ = self.process_frame(
                            prev_gray, prev_points, gray_frame, frame
                        )

                    # Display FPS on the frame
                    show_frame = cv2.resize(frame, (self.dis_width, self.dis_height))
                    cv2.putText(
                        show_frame,
                        f"FPS: {fps:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
   
                    cv2.imshow(exam_name, show_frame)
                if self.use_timer:
                    if self.time.evaluate_time():
                        self.stop_thread = True


                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.stop_thread = True
        finally:
            self.cap.release()
            if self.record_video:
                record.release()
                recorder.save_to_db(exam_id, timestamp)
            cv2.destroyAllWindows()
            print("Consumer thread ended and resources released.")

    def main(self, exam_id):
        producer_thread = Thread(target=self.frame_producer)
        consumer_thread = Thread(target=self.frame_consumer, args=(exam_id,))

        producer_thread.start()
        consumer_thread.start()

        producer_thread.join()
        consumer_thread.join()
        print("Both threads have completed.")

    def __call__(self, exam_id):
        self.main(exam_id)

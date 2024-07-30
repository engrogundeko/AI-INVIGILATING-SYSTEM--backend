import os
import time
import json

import anyio.to_thread
import cv2
import anyio
import numpy as np
from ultralytics import YOLO

from .preprocessor import PreProcessor
from ..config import WKR_DIR, VIDEO_RECORD_PATH, IMAGE_RECORD_PATH


class CheatingDetection:
    image_record_path: str = IMAGE_RECORD_PATH

    def __init__(
        self,
        camera: int = 0,
        log: int = False,
        video_path=None,
        width: int = 840,
        height: int = 680,
        max_magnitude: float = 3.2,
        iou_threshold: float = 0.4,
        confidence_score: float = 0.6,
        resize_percentage: float = None,
        confidence_threshold: float = 0.25,
        no_frame_process: int = 7,
        record_frame: bool = False,
        record_video: bool = True,
        show: bool = False,
        model_name: str = "aisv1n.pt"
    ) -> None:
        self.width = width
        self.hieght = height
        self.preprocessor = PreProcessor()
        self.show = show
        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.model = YOLO
        self.model_name = model_name
        self.record_video = record_video
        self.record_frame = record_frame
        self.video_path = video_path
        self.camera = camera
        self.log = log
        self.cheating_log = []
        self.no_frame = no_frame_process
        self.max_magnitude = max_magnitude
        self.iou_threshold = iou_threshold
        self.confidence_score = confidence_score
        self.resize_percentage = resize_percentage
        self.confidence_threshold = confidence_threshold

    def load_model(self):
        return self.model(self.model_name)

    def _save_to_json(self):
        with open("cheating_log.json", "w") as f:
            json.dump(self.cheating_log, f, indent=4)

    @property
    def farneback_params(self):
        return dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        
    def inference_cheating(self, img):
        pass


    def preprocess(self, prev_gray, frame, frame_counter, frame_time):
        gray, _ = self.start(frame)
        model = self.load_model()
        results = model(frame, conf=self.confidence_threshold, iou=self.iou_threshold)

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

                if confidence_score < 0.60:
                    label = "Not Cheating"
                    colour = (0, 255, 0)
                    confidence_score = 1 - confidence_score
                else:
                    cropped_image = frame[y1:y2, x1:x2]
                    anyio.to_thread.run_sync(self.inference_cheating, cropped_image)

                    label = "Cheating"
                    colour = (0, 0, 255)
                    log_entry = {
                        "frame_id": frame_counter,
                        "time": frame_time,
                        "coordinates": (x1, y1, x2, y2),
                        "confidence_score": float(confidence_score),
                        "pixel_changes": float(avg_magnitude),
                    }
                    self.cheating_log.append(log_entry)
                    if self.record_video:
                        # Handle video recording logic here
                        pass
                    if self.record_frame:
                        pass
                        # self.record(frame, "image", frame_time, confidence_score)

                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(
                    frame,
                    f"{label}: {confidence_score:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    colour,
                    2,
                )
        return gray, frame

    def record(self, frame, prefix, frame_time, confidence_score):
        if not os.path.exists(self.image_record_path):
            os.makedirs(self.image_record_path)
        filename = f"{prefix}_{int(frame_time)}.jpg"
        path = os.path.join(self.image_record_path, filename)
        cv2.imwrite(path, frame)
        data = {
            "image_path": path,
            "timestamp": frame_time,
            "confidence_score": confidence_score,
        }
        # self._save_to_db("cheating_logs", data)

    def start(self, frame):

        frame = self.preprocessor.resize(frame, self.width, self.hieght)
        _frame = self.preprocessor.apply_edge_detection(frame)
        _frame = self.preprocessor.apply_noise_reduction(_frame)
        _frame = self.preprocessor.apply_histogram_equalization(_frame)

        frame = self.preprocessor.apply_clahe(_frame)
        thres_frame = self.preprocessor.apply_adaptive_threshold(frame)
        gaus = self.preprocessor.apply_gaussian_filter(thres_frame)
        prev_gray = cv2.cvtColor(gaus, cv2.COLOR_BGR2GRAY)

        frame_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))

        return prev_gray, frame_time

    def __call__(self):
        frame_counter = 0

        cap = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            raise ValueError("Error opening video stream or file")

        ret, frame = cap.read()

        prev_gray, frame_time = self.start(frame)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_time = time.time()
            frame_counter += 1

            if frame_counter % self.no_frame == 3:
                prev_gray, frame = self.preprocess(
                    prev_gray, frame, frame_counter, frame_time
                )
                if self.show:
                    cv2.imshow("Movement Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            yield frame

        cap.release()
        cv2.destroyAllWindows()
        # self._save_to_json()

    def stop(self):
        cv2.destroyAllWindows()
        
frame_generator = CheatingDetection()



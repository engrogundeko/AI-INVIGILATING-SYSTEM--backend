from .facial import FacialRecogntion
from .anomaly import AnomalyRecogntion
from .data import DataStorage
from .gesture import GestureDetection
from .object import ObjectRecognition


class AIInvigilationSystem:
    def __init__(self) -> None:
        self.data_storage = DataStorage()
        self.gesture_detection_model = GestureDetection()  # detect head movements
        self.facial_recognition_model = FacialRecogntion()
        self.anomaly_detection_model = AnomalyRecogntion()
        self.objection_recognition_model = ObjectRecognition()

    def monitor_exam(self, video_feed, course):
        # Process each frame of the video feed
        for frame in video_feed:
            # Perform facial recognition and detection
            faces = self.facial_recognition_model.detect_faces(frame)
            for face in faces:
                if face["confidence"] >= 0.6:
                    if not self.data_storage.verify_student_identity(
                        face["face"], course
                    ):
                        self.data_storage.log_unrecognized_face(face["face"], course)

            # Perform gesture detection
            gestures = self.gesture_detection_model.detect_gestures(frame)
            for gesture in gestures:
                self.data_storage.log_gesture(gesture)

            # Perform anomaly detection
            anomalies = self.anomaly_detection_model.detect_anomalies(frame)
            for anomaly in anomalies:
                self.data_storage.log_anomaly(anomaly)

            # Perform objection recognition
            objects = self.objection_recognition_model.recognize_objects(frame)
            for obj in objects:
                self.data_storage.log_object(obj)

    def analyze_exam_data(self):
        pass

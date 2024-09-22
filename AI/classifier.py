import math
import os

# import time
# import queue

# import threading
import json
from datetime import datetime

import cv2
import joblib
import numpy as np
from ultralytics import YOLO

# import matplotlib.pyplot as plt

# Load YOLOv8 model

# expansion = 100
expected_feature_length = 2
model = YOLO(r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\models\aisv4l.pt")
svm_model = joblib.load(
    r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\models\ais_modelv6.pkl"
)


new_width = 1920
new_height = 1080

dis_new_width = 840
dis_new_height = 680
threshold = 0.2


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)


# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

farneback_params = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)

# frame_queue = queue.Queue(maxsize=10)
detected_bboxes = []
cheating_list = []
# flows = []


def evaluate_total_result():
    unique_name = datetime.now().strftime("%M_%S")

    with open(f"cheating_list-{unique_name}.json", "w") as json_file:
        json.dump(cheating_list, json_file, indent=4, cls=NumpyEncoder)


def evaluate_cheating(x1, y1, x2, y2, flow, moving_points):

    X2c, Y2c = get_center(x1, y1, x2, y2)
    for cheat in cheating_list:

        X1, Y1, X2, Y2 = map(int, cheat["coordinate"])

        X1c, Y1c = get_center(X1, Y1, X2, Y2)
        if match(X1c, Y1c, X2c, Y2c):
            cheat["flows"].append(flow)
            cheat["moving_points"].append(len(moving_points))


def match(x1, y1, x2, y2):
    distance = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
    result = distance < threshold
    return result


def get_center(x1, y1, x2, y2):
    Xc = (x1 + x2) / 2
    Yc = (y1 + y2) / 2
    return Xc, Yc


def compute_average_cheating(good_new, good_old, bbox):
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

    return avg_magnitude, moving_points

def compute_fa(flow):
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(magnitude)

def extract_features(good_new, good_old):
    for bbox in detected_bboxes:
        x1, y1, x2, y2 = map(int, bbox)

        avg_magnitude, moving_points = compute_average_cheating(good_new, good_old, bbox)
        # fa_avg_magnitude = compute_fa(flow)

        object_size = (x2 - x1) * (y2 - y1)

        # Append individual feature values to the list

        X2c, Y2c = get_center(x1, y1, x2, y2)
        for cheat in cheating_list:

            X1, Y1, X2, Y2 = map(int, cheat["coordinate"])

            X1c, Y1c = get_center(X1, Y1, X2, Y2)
            if match(X1c, Y1c, X2c, Y2c):

                features = [
                    avg_magnitude,
                    object_size,
                ]

                if not len(features) == expected_feature_length:
                    print(
                        f"Skipping prediction: Expected {expected_feature_length} features, but got {len(features)}"
                    )
                else:
                    f_val = np.array(features).reshape(
                        1, -1
                    )  # Reshape to (1, n_features)
                    label = svm_model.predict(f_val)[0]
                    # probabilities = svm_model.predict_proba(f_val)

                    if label == 0:
                        evaluate_cheating(x1, y1, x2, y2, avg_magnitude, moving_points)

                break


def pre_process_frame(
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

    if gaussian_filter or noise_reduction:  # Combine both as they apply Gaussian Blur
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


def process_frame(prev_gray, gray, prev_points):

    # Calculate optical flow using Lucas-Kanade
    # flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **farneback_params)
    
    next_points, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, prev_points, None, **lk_params
    )

    # Select good points
    good_new = next_points[status == 1]
    good_old = prev_points[status == 1]
    
    extract_features(good_new, good_old)

    return gray, good_new.reshape(-1, 1, 2),


def frame_producer():
    path = r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\media\datasets\converted"
    video_files = os.listdir(path)
    video_paths = [os.path.join(path, file) for file in video_files]
    for path in video_paths:
        print(path)
        cap = cv2.VideoCapture(path)
        prev_gray = None
        frame_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1

            frame = cv2.resize(frame, (new_width, new_height))
            if frame_counter == 1:
                # print(frame_counter)
                global cheating_list
                cheating_list = []
                results = model(frame, conf=0.15, iou=0.3)

                for result in results:
                    count = 1

                    for bbox in result.boxes.xyxy:
                        detected_bboxes.append(list(map(int, bbox)))
                        cheating_list.append(
                            {
                                "student_id": count,
                                "coordinate": list(map(int, bbox)),
                                "flows": [],
                                "moving_points": []
                            }
                        )
                        count += 1

            if frame_counter % 5 == 0:

                gray_frame = pre_process_frame(frame)

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
                    prev_gray, prev_points = process_frame(prev_gray, gray_frame, prev_points)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        evaluate_total_result()


if __name__ == "__main__":
    frame_producer()

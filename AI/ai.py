import winsound
import json
import time
import anyio
import anyio.to_process
import anyio.to_thread
import cv2
import numpy as np
from ultralytics import YOLO
import joblib

# Load YOLOv8 model
model = YOLO("aisv1n.pt")  # Load YOLO model with pre-trained weights
pth = r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\media\datasets\converted\talking.mp4"
expected_feature_length = 5
# Initialize video capture

cap = cv2.VideoCapture(pth)
svm_model = joblib.load("ais_modelv3.pkl")
# Frame dimensions
new_width = 840
new_height = 680

# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("cheating_spoof.avi", fourcc, 20.0, (new_width, new_height))
cheating_log = []


def extract_features(frame, good_new, good_old, results):

    for result in results:
        for bbox in result.boxes.xyxy:
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
                std_magnitude = np.std(magnitudes)
                direction = np.arctan2(
                    moving_points[:, 1] - old_points[:, 1],
                    moving_points[:, 0] - old_points[:, 0],
                )
                avg_direction = np.mean(direction)
                num_moving_points = len(moving_points)
            else:
                avg_magnitude = 0
                std_magnitude = 0
                avg_direction = 0
                num_moving_points = 0

            object_size = (x2 - x1) * (y2 - y1)

            # Append individual feature values to the list
            features = [
                avg_magnitude,
                std_magnitude,
                avg_direction,
                num_moving_points,
                object_size,
            ]

            if len(features) == expected_feature_length:
                f_val = np.array(features).reshape(1, -1)  # Reshape to (1, n_features)
                label = svm_model.predict(f_val)[0]
                # print(label)
                probabilities = svm_model.predict_proba(f_val)

                if label == 0:
                    pass
                    # code = "Not Cheating"
                    # colour = (0, 255, 0)
                    # confidence_score = probabilities[0, 0]
                else:
                    code = "Cheating"
                    colour = (0, 0, 255)
                    confidence_score = probabilities[0, 1]

                    # for bbox in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(
                        frame,
                        f"{code}: {confidence_score:.2f}%",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        colour,
                        2,
                    )
            else:
                print(
                    f"Skipping prediction: Expected {expected_feature_length} features, but got {len(features)}"
                )


def apply_clahe(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)


def apply_gaussian_filter(frame, kernel_size=(5, 5), sigma=1):
    return cv2.GaussianBlur(frame, kernel_size, sigma)


def apply_adaptive_threshold(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2BGR)


def apply_edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def apply_noise_reduction(frame, kernel_size=(5, 5), sigma=1):
    return cv2.GaussianBlur(frame, kernel_size, sigma)


def apply_histogram_equalization(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)


def process_frame(prev_gray, prev_points, frame, frame_time, frame_counter):
    # Apply advanced pre-processing techniques
    frame = cv2.resize(frame, (new_width, new_height))
    _frame = apply_noise_reduction(frame)
    _frame = apply_edge_detection(_frame)
    _frame = apply_histogram_equalization(_frame)

    clax_frame = apply_clahe(frame)
    thres_frame = apply_adaptive_threshold(clax_frame)
    gaus = apply_gaussian_filter(thres_frame)
    gray = cv2.cvtColor(gaus, cv2.COLOR_BGR2GRAY)

    # Detect objects using YOLOv8
    results = model(frame, conf=0.25, iou=0.4)

    # Calculate optical flow using Lucas-Kanade
    next_points, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, prev_points, None, **lk_params
    )

    # Select good points
    good_new = next_points[status == 1]
    good_old = prev_points[status == 1]

    features = extract_features(frame, good_new, good_old, results)

    return gray, good_new.reshape(-1, 1, 2), frame


def main():
    prev_gray = None
    prev_points = None
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))
        frame_counter += 1

        if frame_counter % 5 == 0:  # Process every 5th frame for better tracking
            frame_resized = cv2.resize(frame, (new_width, new_height))
            _frame = apply_edge_detection(frame_resized)
            _frame = apply_noise_reduction(_frame)
            _frame = apply_histogram_equalization(_frame)

            frame_resized = apply_clahe(_frame)
            thres_frame = apply_adaptive_threshold(frame_resized)
            gaus = apply_gaussian_filter(thres_frame)
            gray = cv2.cvtColor(gaus, cv2.COLOR_BGR2GRAY)

            if prev_gray is None:
                prev_gray = gray
                prev_points = cv2.goodFeaturesToTrack(
                    prev_gray,
                    maxCorners=100,
                    qualityLevel=0.3,
                    minDistance=7,
                    blockSize=7,
                )
                processed_frame = frame_resized
            else:
                prev_gray, prev_points, processed_frame = process_frame(
                    prev_gray, prev_points, frame, frame_time, frame_counter
                )

            # out.write(processed_frame)
            cv2.imshow("Automatic Invigilation System", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

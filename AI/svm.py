import csv
import cv2
import numpy as np
import os
from ultralytics import YOLO

data = []

path = r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\media\datasets\converted"
new_width = 1920
new_height = 1080

video_files = os.listdir(path)
video_paths = [os.path.join(path, file) for file in video_files]
model = YOLO("aisv1n.pt")  # Specify the YOLO model or path here
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
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


def extract_features(frame, good_new, good_old, results):
    features = []
    max_magnitude = 9.142
    label = 0
    
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
                
                confidence_score = min(max(avg_magnitude / max_magnitude, 0), 1)
                if confidence_score > 0.65:
                    label = 1
            else:
                avg_magnitude = 0
                std_magnitude = 0
                avg_direction = 0
                num_moving_points = 0

            object_size = (x2 - x1) * (y2 - y1)
            
            features.extend([
                avg_magnitude,
                std_magnitude,
                avg_direction,
                num_moving_points,
                object_size,
                label
            ])
            
        # features.append(label)
    data.append(features)


def process_frame(prev_gray, prev_points, frame):
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

    extract_features(frame, good_new, good_old, results)


    return gray, next_points, frame


def main():
    frame_counter = 0

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        prev_gray = None
        prev_points = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

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
                else:
                    prev_gray, prev_points, processed_frame = process_frame(
                        prev_gray, prev_points, frame
                    )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Done ----- ", video_path)
        
    save_features_to_csv(data)


def save_features_to_csv(features, filename="datasetv5.csv"):
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow([
                "Avg Magnitude", "Std Magnitude", "Avg Direction", 
                "Num Moving Points", "Object Size", "Label"
            ])
        for feature_set in features:
            writer.writerow(feature_set)



if __name__ == "__main__":
    main()

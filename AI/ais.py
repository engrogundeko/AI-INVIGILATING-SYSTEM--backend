import math
import os
import time
import queue
import threading
import json
from datetime import datetime

import cv2
import joblib
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load YOLOv8 model
pth = r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\media\datasets\exams\normal_1.mp4"
expansion = 100
expected_feature_length = 2
model = YOLO(r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\models\aisv4l.pt")
svm_model = joblib.load(r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\models\ais_modelv6.pkl")
lstm_model = load_model(r'C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\lstm.h5')

# Initialize video capture
cap = cv2.VideoCapture(pth)
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

frame_queue = queue.Queue(maxsize=10)
detected_bboxes = []
cheating_list = []
flows = []

window_size = 10
num_features = 1  # Update this if you have more features per sample
buffer = np.zeros((window_size, num_features))

def predict_real_time(new_sample):
    # Append the new sample to the buffer
    buffer = np.append(buffer[1:], [new_sample], axis=0)  # Remove the oldest sample and add the new one

    # Check if the buffer is full (contains 10 samples)
    if len(buffer) == window_size:
        # Reshape buffer to match the LSTM input shape: (1, timesteps, num_features)
        input_data = np.expand_dims(buffer, axis=0)
        
        # Make predictions
        prediction = lstm_model.predict(input_data)
        return prediction
    else:
        return None

def plot():
    student_ids = [entry["student_id"] for entry in cheating_list]
    cheating_percentages = [
        float(entry["average_cheat"].strip("%")) for entry in cheating_list
    ]

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(student_ids, cheating_percentages, color="skyblue", edgecolor="black")
    plt.xlabel("Student ID")
    plt.ylabel("Cheating Percentage")
    plt.title("Cheating Percentage vs Student ID")
    plt.xticks(student_ids)
    plt.show()


def evaluate_total_result():
    for cheat in cheating_list:
        cheating_score = cheat["all_cheating_scores"]
        score = np.mean(cheating_score) if cheating_score else 0
        score = score * 100
        cheat["average_cheat"] = f"{score:.4f}"

    with open("cheating_list.json", "w") as json_file:
        json.dump(cheating_list, json_file, indent=4, cls=NumpyEncoder)


def evaluate_cheating(x1, y1, x2, y2, confidence_score, frame):

    X2c, Y2c = get_center(x1, y1, x2, y2)
    for cheat in cheating_list:

        X1, Y1, X2, Y2 = map(int, cheat["coordinate"])

        X1c, Y1c = get_center(X1, Y1, X2, Y2)
        if match(X1c, Y1c, X2c, Y2c):
            cheat["all_cheating_scores"].append(confidence_score)

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

            y1 = max(0, y1 - expansion)
            y2 = min(frame.shape[0], y2 + expansion)
            x1 = max(0, x1 - expansion)
            x2 = min(frame.shape[1], x2 + expansion)
            crp_frame = frame[y1:y2, x1:x2]
            
            dt = datetime.now().strftime("%H_%M_%S")
            filename = f"{id}__{dt}"
            path = save_image(crp_frame, filename)
            cheat["image_paths"].append(path)


def match(x1, y1, x2, y2):
    distance = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
    result = distance < threshold
    return result


def get_center(x1, y1, x2, y2):
    Xc = (x1 + x2) / 2
    Yc = (y1 + y2) / 2
    return Xc, Yc


def compute_average_cheating(): ...


def extract_features(frame, good_new, good_old):
    for bbox in detected_bboxes:
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
            # std_magnitude = np.std(magnitudes)
            # direction = np.arctan2(
            #     moving_points[:, 1] - old_points[:, 1],
            #     moving_points[:, 0] - old_points[:, 0],
            # )
            # avg_direction = np.mean(direction)
            # num_moving_points = len(moving_points)
        else:
            avg_magnitude = 0
            # std_magnitude = 0
            # avg_direction = 0
            # num_moving_points = 0

        object_size = (x2 - x1) * (y2 - y1)
        
        norm = avg_magnitude / object_size

        # Append individual feature values to the list

        X2c, Y2c = get_center(x1, y1, x2, y2)
        for cheat in cheating_list:

            X1, Y1, X2, Y2 = map(int, cheat["coordinate"])

            X1c, Y1c = get_center(X1, Y1, X2, Y2)
            if match(X1c, Y1c, X2c, Y2c):

                cheat["flows"].append(avg_magnitude)
                pre = predict_real_time(norm)
                print(norm, pre)
                if len(cheat["flows"]) == 6:
                    mean_flow = float(np.mean(cheat["flows"]))
                    cheat["flows"] = []

                    features = [
                        mean_flow,
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
                        probabilities = svm_model.predict_proba(f_val)

                        if label == 1:

                            code = "Suspicious"
                            colour = (0, 0, 255)
                            confidence_score = probabilities[0, 1]
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

                            evaluate_cheating(x1, y1, x2, y2, confidence_score, frame)
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


def process_frame(prev_gray, prev_points, gray, frame):

    # Calculate optical flow using Lucas-Kanade
    next_points, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, prev_points, None, **lk_params
    )

    # Select good points
    good_new = next_points[status == 1]
    good_old = prev_points[status == 1]

    extract_features(frame, good_new, good_old)

    return gray, good_new.reshape(-1, 1, 2), frame


def frame_producer():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)

    frame_queue.put(None)  # Signal the end of the stream


def frame_consumer():
    global detected_bboxes
    prev_gray = None
    prev_points = None
    frame_counter = 0
    fps = 0
    start_time = time.time()

    while True:
        frame = frame_queue.get()
        if frame is None:  # End of the stream
            break

        frame_counter += 1

        # Calculate FPS
        current_time = time.time()
        elapsed_time = current_time - start_time
        fps = (frame_counter / elapsed_time) if elapsed_time != 0 else 0

        frame = cv2.resize(frame, (new_width, new_height))
        if frame_counter == 1:  # Run YOLO detection only once on the first frame
            results = model(frame, conf=0.25, iou=0.4)

            for result in results:
                count = 1

                for bbox in result.boxes.xyxy:
                    detected_bboxes.append(list(map(int, bbox)))
                    cheating_list.append(
                        {
                            "student_id": count,
                            "coordinate": list(map(int, bbox)),
                            "all_cheating_scores": [],
                            "image_paths": [],
                            "flows": [],
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
                processed_frame = gray_frame
            else:
                prev_gray, prev_points, processed_frame = process_frame(
                    prev_gray, prev_points, gray_frame, frame
                )

            # Display FPS on the frame
            show_frame = cv2.resize(frame, (dis_new_width, dis_new_height))
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
            cv2.imshow("Automatic Invigilation System", show_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    evaluate_total_result()
    # plot()


def main():
    producer_thread = threading.Thread(target=frame_producer)
    consumer_thread = threading.Thread(target=frame_consumer)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()


def save_image(image, image_name):
    from PIL import Image

    IMAGE_SAVE_DIR = r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\AI\img"
    if image.dtype != np.uint8:
        image = (255 * image).astype(np.uint8)
    if not os.path.exists(IMAGE_SAVE_DIR):
        os.makedirs(IMAGE_SAVE_DIR)
    file_name = f"{image_name}.jpg"
    image_path = os.path.join(IMAGE_SAVE_DIR, file_name)
    image = Image.fromarray(image)
    image.save(image_path)

    return image_path


if __name__ == "__main__":
    main()

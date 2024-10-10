import io
import os
from bson import ObjectId
from datetime import datetime

from ..repository import (
    userRespository,
    examRegistrationRespository,
    examAttedanceRespository,
    examLocationRepo,
    suspiciousReportRespository,
    detectionRespository,
    studentDataRepo,
    examRespository,
    reportRespository,
    analysedRepo
)
from ..config import (
    AIConstant, 
    # UNRECOGNISED_FACE, 
    # UNDETECTED_FACE,
    AIS_YOLO,
    STUDENT_DATA,
    STATIC_FOLDER
    )

from ..utils import image_utils
from .schema import ExamLocation, ExamAttendance, Location

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from deepface import DeepFace
from matplotlib import patches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def take_attendance(image, exam_id):
    file_content = image.file.read()

    # Convert file content to an image
    img = Image.open(io.BytesIO(file_content))
    arr = np.array(img)
    student_locations = []
    undetected_faces = []
    unrecognised_faces = []

    model = YOLO(AIS_YOLO)
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
                    # filename = os.path.join(UNRECOGNISED_FACE, datetime.now().strftime("%M:%S"))
                    # img_path = AIConstant.save_image(cropped_frame, filename)
                    # unrecognised_faces.append(img_path)
                    continue
                except Exception:
                    # filename = os.path.join(UNDETECTED_FACE, datetime.now().strftime("%M:%S"))
                    # img_path = AIConstant.save_image(cropped_frame, filename)
                    # undetected_faces.append(img_path)
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

def time_to_datetime(t, base_date):
    """Combine a time object with a base date to create a datetime object."""
    return datetime.combine(base_date, t)

def plot(exam_id, save=False):
    student_ids = []
    all_average_cheating = []
    
    exam = examRespository.find_one({"_id": ObjectId(exam_id)})
    # Extract start and end time from the database
    start_time_str = exam["start_time"]
    end_time_str = exam["end_time"]

    # Define the base date (current date)
    base_date = datetime.now().date()

    # Parse the start and end times into datetime.time objects
    start_time = datetime.strptime(start_time_str, AIConstant.time_format).time()
    end_time = datetime.strptime(end_time_str, AIConstant.time_format).time()

    # Combine base date with time objects to create datetime objects
    start_datetime = time_to_datetime(start_time, base_date)
    end_datetime = time_to_datetime(end_time, base_date)

    # Set the duration and start time
    exam_duration = (end_datetime - start_datetime).total_seconds() / 60
    
    cheating_list = list(suspiciousReportRespository.find({"exam_id": exam_id}))
    
    for entry in cheating_list:
        student_id = entry["student_id"]
        cheating_score = [float(ch * 100) for ch in entry["all_cheating_scores"] ]
        
        total_cheating_score = sum(cheating_score)
        if exam_duration < 10:
            exam_duration = 10

        # Calculate the average cheating score over the exam duration
        cheating_rate = total_cheating_score / exam_duration
        

        normalized_score = min(max(0, cheating_rate), 100)
        
        student_ids.append(student_id)
        all_average_cheating.append(normalized_score)

        
        
    bar_colors = [
        'red' if percentage > 50 else 'purple' if percentage > 30 else 'skyblue'
        for percentage in all_average_cheating
    ]
        
        

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(student_ids, all_average_cheating, color=bar_colors, edgecolor="black")
    plt.xlabel("Student ID")
    plt.ylabel("Cheating Rating Per Minute")
    plt.title("Cheating Rating Percentage Per Minute vs Student ID")
    plt.xticks(student_ids)
    plt.ylim(0, 100)
    if save:
        path = os.path.join(STATIC_FOLDER, f"cheat_rate_{exam_id}.png")
        plt.savefig(path)
        plt.close()
        return path
    else:
        plt.show()
        plt.close()
    
def plot_average(exam_id, save=False):
    student_ids = []
    all_average_cheating = []
    
    cheating_list = list(suspiciousReportRespository.find({"exam_id": exam_id}))
    
    for entry in cheating_list:
        student_id = entry["student_id"]
        cheating_score = [float(ch) for ch in entry["all_cheating_scores"] ]
        
        student_ids.append(student_id)
        average_cheating = np.mean(cheating_score)
        all_average_cheating.append(average_cheating * 100)
        
    bar_colors = [
        'red' if percentage > 70 else 'purple' if percentage > 50 else 'skyblue'
        for percentage in all_average_cheating
    ]

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(student_ids, all_average_cheating, color=bar_colors, edgecolor="black")
    plt.xlabel("Student ID")
    plt.ylabel("Average Cheating")
    plt.title("Average Cheating vs Student ID")
    plt.xticks(student_ids)
    plt.ylim(0, 100)
    
    if save:
        path = os.path.join(STATIC_FOLDER, f"average_cheat_{exam_id}.png")
        plt.savefig(path)
        plt.close()  # Close the plot after saving
        return path
    else:
        plt.show()
        plt.close() 
    
def see_detections(exam_id):
    
    image = detectionRespository.find_one({"exam_id": exam_id})
    image = image["image_path"]
    
    # file_content = image.file.read()

    # # Convert file content to an image
    # img = Image.open(io.BytesIO(file_content))
    # arr = np.array(img)
    img, _ = image_utils.load_image(image)
    
    detections = list(suspiciousReportRespository.find({"exam_id": exam_id}))
    
    if not detections:
        print("No detections found")
        return
    
    results = [det["coordinates"] for det in detections]

    colour = (0, 255, 0)

    for re in results:
        count = 1
        x1, y1, x2, y2 = map(int, re)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
                img,
                f"id: {count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                colour,
                2,
                cv2.LINE_AA,
            )
        count += 1
    show_frame = cv2.resize(img, (840, 680))
    cv2.imshow("CLASS", show_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.destroyAllWindows()
            

def save_detections(exam_id, video_source, camera_id):
    model = YOLO(AIS_YOLO)
    
    if camera_id:
        video_source = None

    cap = cv2.VideoCapture(video_source if video_source else camera_id)

    # Check if the video source is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    # Resize the frame (if needed)
    frame = cv2.resize(frame, (1920, 1080))

    # Run model inference on the first frame
    results = model(frame, conf=0.15, iou=0.30)

    count = 1
    colour = (0, 255, 0)
    students_data = []

    for re in results:
        for bbox in re.boxes.xyxy:
            box = list(map(int, bbox))
            x1, y1, x2, y2 = box

            # Append student data
            students_data.append(
                {
                    "id": count,
                    "coordinates": box
                }
            )

            # Draw rectangle and put text
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(
                frame,
                f"id: {count}",
                (x1, y1 - 10),  # Adjusted text position
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                colour,
                2,
                cv2.LINE_AA,
            )
            count += 1

    # Display the frame (optional)
    show_frame = cv2.resize(frame, (840, 680))
    cv2.imshow("CLASS", show_frame)

    key = cv2.waitKey(0) & 0xFF
    print(key)
    
    return_data = None
    print(ord('s'))

    if key == ord('s'):
        print("-----------------")
        # Save the first frame to an image file
        path = os.path.join(STUDENT_DATA, f"{exam_id}.jpg")
        success = cv2.imwrite(path, frame)
        if success:
            print(f"First frame saved successfully at {path}")
        else:
            print("Error: Failed to save the frame.")

        # Save the exam data to the database
        exam = {
            "exam_id": exam_id,
            "path": path,
            "no_of_students": len(students_data),
            "students_data": students_data,
        }
        studentDataRepo.insert_one(exam)
        print("Data saved to the database.")
        return_data = exam

    elif key == ord('q'):
        print("Operation cancelled. Data not saved.")

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    if return_data is not None:
        return_data.pop("_id")
    return return_data

def plot_all_images(exam_id, save=False):
    cheating_list = list(suspiciousReportRespository.find({"exam_id": exam_id}))
    
    for cheat in cheating_list:
        student_id = cheat["student_id"]
        scores = cheat["all_cheating_scores"]
        scores = [score * 100 for score in scores]
        average_score = float(cheat["average_cheat"].strip("%"))
        coordinates = cheat["coordinates"]
        no_of_suspicious_activity = len(scores)
        image_paths = cheat["image_paths"]
        num_images = len(image_paths)
        cols = 3  # Number of columns in the grid
        rows = (num_images + cols - 1) // cols  # Calculate rows needed

        # Create a figure with subplots
        plt.figure(figsize=(15, 5 * rows))
        plt.suptitle(f'Student {student_id} - Cheating Detection', fontsize=16)

        for i, image_path in enumerate(image_paths):
            # Load the sample image using OpenCV
            sample_image = cv2.imread(image_path)
            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting

            # Add a subplot for each image
            ax = plt.subplot(rows, cols, i + 1)
            ax.imshow(sample_image)
            
            # Draw bounding boxes on the image
            x1,y1,x2,y2 = list(map(int, coordinates))
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

            ax.axis('off')
            ax.set_title(f'Image {i + 1}')

        # Add a text box with summary information below the plots
        summary_text = (
            f'Average Cheating Score: {average_score}%\n'
            f'Number of Suspicious Activities: {no_of_suspicious_activity}'
        )
        plt.figtext(0.5, 0.01, summary_text, ha="center", fontsize=12, bbox={"facecolor": "lightgrey", "alpha": 0.5, "pad": 5})

        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save:
            path = ...
            plt.savefig(path)
            return path
        else:
            plt.show()


def plot_student_data(exam_id):
    cheating_list = list(suspiciousReportRespository.find({"exam_id": exam_id}))
    
    for cheat in cheating_list:
        student_id = cheat["student_id"]
        scores = cheat["all_cheating_scores"]
        scores = [score * 100 for score in scores]
        average_score = float(cheat["average_cheat"].strip("%"))
        coordinates = cheat["coordinates"]
        no_of_suspicious_activity = len(scores)
        sample_image_path = cheat["image_paths"]

        # Create a figure with subplots
        plt.figure(figsize=(12, 8))
        grid = GridSpec(2, 2, height_ratios=[1, 2])

        # Bar plot for cheating scores
        plt.subplot(grid[0, 0])
        plt.bar(range(len(scores)), scores, color='skyblue', edgecolor='black')
        plt.xlabel('Activity Index')
        plt.ylabel('Cheating Score')
        plt.title(f'Cheating Scores for Student {student_id}')
        plt.ylim(0, 100)  # Assuming scores are percentages

        if sample_image_path:
            sample_image_path = sample_image_path[0]

            # Load the sample image using OpenCV
            sample_image = cv2.imread(sample_image_path)
            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting
            ax = plt.subplot(grid[1, :])
            ax.imshow(sample_image)
            
        # Scatter plot for coordinates if meaningful (assuming coordinates are (x, y) pairs)
        x1,y1,x2,y2 = list(map(int, coordinates))
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        plt.axis('off')
        plt.title(f'Sample Image with Bounding Boxes for Student {student_id}')


        # Add a text box with summary information
        summary_text = (
            f"student ID: {student_id}\n"
            f'Average Cheating Score: {average_score}%\n'
            f'Number of Suspicious Activities: {no_of_suspicious_activity}'
        )
        plt.figtext(0.5, 0.02, summary_text, ha="center", fontsize=12, bbox={"facecolor": "lightgrey", "alpha": 0.5, "pad": 5})

        # Show the plots
        plt.tight_layout()
        plt.show()
        
def evaluate_cheating(exam_id):
    cheating_list = list(suspiciousReportRespository.find({"exam_id": exam_id}))
    
    y_true = []
    y_pred = []
    sample = []
    
    for cheat in cheating_list:
        count = 0
        actual_cheating = []
        non_actual_cheating = []
        
        # id = cheat["_id"]
        image_paths = cheat.get("image_paths", [])
        
        for path in image_paths:
            count += 1
            cap = cv2.VideoCapture(path)

            # Check if the video source is opened successfully
            if not cap.isOpened():
                print(f"Error: Could not open video source {path}.")
                continue  # Skip to the next video instead of returning early

            # Read the first frame
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read the first frame of {path}.")
                cap.release()
                continue  # Skip to the next video instead of returning early
            
            show_frame = cv2.resize(frame, (840, 680))
            cv2.imshow("Cheating Classifier", show_frame)

            key = cv2.waitKey(0) & 0xFF
            print(key)
            
            print(ord('s'))

            if key == ord('s'):
                print(path)
                # actual_cheating.append(path)
                y_true.append(1)
                y_pred.append(1)
                
                if count == 1:
                    sample.append(path)
            else:
                # non_actual_cheating.append(path)
                y_true.append(0)
                y_pred.append(1)
        
            cap.release()  # Close the video capture

        # cheat_paths = {
        #     "cheating": actual_cheating,
        #     "non_cheating": non_actual_cheating
        # }

        # Update the exam record with cheat_paths
        # examRespository.update_one({"_id": id}, {"$set": {"all_paths": cheat_paths}})

    # Convert lists to numpy arrays for element-wise operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute confusion matrix elements
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    # Calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Prepare the data for update
    data = dict(
        y_true=y_true.tolist(),
        y_pred=y_pred.tolist(),
        true_positive=int(TP),
        true_negative=int(TN),
        false_positive=int(FP),
        false_negative=int(FN),
        accuracy=accuracy,
        precision=precision,
        f1_score=f1_score,
        recall=recall,
        sample_images=sample
    )

    # Update the report with the calculated metrics
    reportRespository.update_one({"exam_id": exam_id}, {"exam_metrics": data})

    # Cleanup: Close any open windows
    cv2.destroyAllWindows()
    
    analysedRepo.insert_one({
        "exam_id": exam_id,
        "is_printed": False,
        "pdf_path": None,
        "paths": None
    })
    
    return data
    
                
def plot_confusion_matrix(exam_id, metrics):
    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]
    
    cm = confusion_matrix(y_true, y_pred)

    # Create a confusion matrix display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Plot the confusion matrix
    disp.plot(cmap=plt.cm.Blues)

    # Save the plot to a file
    path = os.path.join(STATIC_FOLDER, f"conf_matrix_{exam_id}.png")
    plt.savefig(path)
    
    return path
    
def plot_activity(report):
    paths = {
        "activities": [],
        "anomalies": []
    }
    
    student_ids = []
    anomalies_score = []

    # Sample data
    start_time = report["start_time"]
    end_time = report["end_time"]

    activities = report["anomaly_analysis"]

    for count, activity in enumerate(activities, start=1):
        student_id = activity["student_id"]
        optical_flows = activity["magnitude"]
        scaler = StandardScaler()
        flow_array = np.array(optical_flows).reshape(-1, 1)
        
        normalized_flow = scaler.fit_transform(flow_array)
        
        cheat_timestamps = activity["cheat_timestamp"]
        
        # Create timestamps based on start and end times
        if count == 1:
            timestamps = pd.date_range(start=start_time, end=end_time, periods=len(normalized_flow))
        
        # Create a DataFrame to hold the timestamps and activities
        df = pd.DataFrame({'timestamp': timestamps, 'activity': normalized_flow})

        # Convert cheating timestamps to datetime
        cheating_timestamps = pd.to_datetime(cheat_timestamps)

        # Plotting the activity over time
        plt.figure(figsize=(10, 6))
        plt.plot(df['timestamp'], df['activity'], marker='o', linestyle='-', color='blue', label='Activity')

        # Mark cheating timestamps on the plot
        if cheat_timestamps:
            for cheating_time in cheating_timestamps:
                plt.axvline(cheating_time, color='red', linestyle='--', label='Cheating Event' if cheating_time == cheating_timestamps[0] else '')

        # Customizing the plot
        plt.title(f'Student {student_id} Activity with Cheating Timestamps', fontsize=16)
        plt.xlabel('Timestamp', fontsize=12)
        plt.ylabel('Activity Level', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()

        # Save the activity plot
        activity_path = os.path.join(STATIC_FOLDER, f"activity_{student_id}.png")
        # os.makedirs(os.path.dirname(activity_path), exist_ok=True)
        plt.savefig(activity_path)
        paths["activities"].append(activity_path)
        plt.close()

        # Plot anomalies in optical flows
        plt.figure(figsize=(10, 6))
        plt.plot(normalized_flow, label='Optical Flow', color='blue')

        # Highlight anomalies
        anomalies_idx = activity["anomalies_idx"]
        plt.scatter(anomalies_idx, [normalized_flow[i] for i in anomalies_idx], color='red', label='Anomalies', zorder=5)

        plt.xlabel('Data Point Index')
        plt.ylabel('Value')
        plt.title(f'Student {student_id} - Anomalies Detection')
        plt.legend()
        plt.grid(True)

        # Save the anomaly plot
        anomaly_path = os.path.join(STATIC_FOLDER, f"anomalies_{student_id}.png")
        # os.makedirs(os.path.dirname(anomaly_path), exist_ok=True)
        plt.savefig(anomaly_path)
        paths["anomalies"].append(anomaly_path)
        plt.close()

        student_ids.append(f"Student {student_id}")
        anomalies_score.append(float(activity["score"].strip("%")))  # Convert to float

    # Plot the contamination level per student
    plt.figure(figsize=(10, 6))
    plt.bar(student_ids, anomalies_score, color='skyblue')

    plt.xlabel('Student ID')
    plt.ylabel('Contamination Level (%)')
    plt.title('Contamination Level per Student')

    # Show the exact contamination levels on top of each bar
    for i, v in enumerate(anomalies_score):
        plt.text(i, v + 0.1, f"{v}%", ha='center', fontweight='bold')

    plt.tight_layout()

    # Save the contamination level plot
    contamination_path = os.path.join(STATIC_FOLDER, f"total_anomaly_{report['exam_id']}.png")
    # os.makedirs(os.path.dirname(contamination_path), exist_ok=True)
    plt.savefig(contamination_path)
    plt.close()

    return paths, contamination_path


def plot_model_score(exam_id, metric_values):
    # Define the metric names
    metric_names = ['Accuracy', 'Precision', 'F1 Score', "Recall"]
    
    # Get the corresponding values from the passed metric_values dictionary
    values = [metric_values["accuracy"], metric_values["precision"], metric_values["f1_score"], metric_values["recall"]]

    # Plot the metrics
    plt.figure(figsize=(8, 5))
    plt.bar(metric_names, values, color=['blue', 'green', 'orange', 'red'])
    plt.ylim(0, 1)  # Set y-axis limit for percentages
    plt.title('Performance Metrics')
    plt.ylabel('Score')
    plt.xlabel('Metrics')

    # Add text labels on top of bars
    for i, value in enumerate(values):
        plt.text(i, value + 0.02, f'{value:.2f}', ha='center', fontsize=12)

    # Save the plot to a file
    path = os.path.join(STATIC_FOLDER, "model_score", f"{exam_id}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure directory exists
    plt.savefig(path)
    plt.close()  
    
    return path


def plot_required_graphs(exam_id):
    report = reportRespository.find_one({"exam_id": exam_id})
    metrics = report["exam_metrics"]
    
    avg_path = plot_average(exam_id=exam_id, save=True)
    rate_path = plot(exam_id=exam_id, save=True)
    confusion_matrix_path = plot_confusion_matrix(exam_id, metrics)
    model_path = plot_model_score(exam_id, metrics)
    activity_paths, anomaly_path = plot_activity(report)
    
    paths =  dict(
        average_cheat=avg_path,
        rated_cheat=rate_path,
        confusion_matrix=confusion_matrix_path,
        performance=model_path,
        anomalies=activity_paths["anomalies"],
        activities=activity_paths["activities"],
        anomaly=anomaly_path
    )
    
    analysedRepo.update_one(
       { "exam_id": exam_id}, {"paths": paths}
    )

    return paths
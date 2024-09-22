import tempfile
from fastapi.exceptions import RequestValidationError
from .ai import AIInvigilatingSystem
from . import services
from ..repository import examRespository, courseRespository, sessionRepository

import cv2
from fastapi import APIRouter, File, UploadFile


router = APIRouter(tags=["Artificial Intelligience"], prefix="/ai")


def search_exam(session: str, course_code: str):
    course = courseRespository.find_one({"code": course_code.upper()})
    if course is None:
        raise RequestValidationError("Course does not exist")

    session_ = sessionRepository.find_one({"name": session})
    if session_ is None:
        raise RequestValidationError("Session does not exist")

    exam = examRespository.find_one(
        {"course_id": str(course["_id"]), "session": str(session_["_id"])}
    )

    return str(exam["_id"])


# @role_required([Role.admin])
@router.get("/plot_rate")
def plot_cheating_rate(
    course_code: str,
    session: str,
):
    """
    Generates a bar chart showing the cheating per minute for students in a specified exam.

    Parameters:
    exam_id (str): The unique identifier for the exam whose cheating data will be visualized.

    The function retrieves data from the suspiciousReportRespository based on the exam_id,
    extracts student IDs and their corresponding cheating percentages, and then plots a bar chart.
    The bars are color-coded based on the cheating percentage:
    - Red for percentages > 70%
    - Purple for percentages > 50% and <= 70%
    - Sky blue for percentages <= 50%
    """
    exam_id = search_exam(session, course_code)
    services.plot(exam_id)


@router.get("/plot")
def plot(
    course_code: str,
    session: str,
):
    """
    Generates a bar chart showing the cheating per minute for students in a specified exam.

    Parameters:
    exam_id (str): The unique identifier for the exam whose cheating data will be visualized.

    The function retrieves data from the suspiciousReportRespository based on the exam_id,
    extracts student IDs and their corresponding cheating percentages, and then plots a bar chart.
    The bars are color-coded based on the cheating percentage:
    - Red for percentages > 70%
    - Purple for percentages > 50% and <= 70%
    - Sky blue for percentages <= 50%
    """
    exam_id = search_exam(session, course_code)
    services.plot_average(exam_id)


@router.get("/plot_all_images/{exam_id}")
def plot_images(
    course_code: str,
    session: str,
):
    """
    Plots all images of detected cheating activities for each student in a given exam.

    Parameters:
    - exam_id (str): The unique identifier for the exam to retrieve cheating data.

    The function performs the following steps:
    1. Retrieves a list of cheating reports from the suspiciousReportRespository using the exam_id.
    2. For each student in the report, it processes the images associated with detected cheating activities.
    3. Displays images with bounding boxes indicating suspicious activities.
    4. Displays summary information including average cheating score and number of suspicious activities.
    """
    exam_id = search_exam(session, course_code)
    services.plot_all_images(exam_id)


@router.get("/plot_individual_exam")
def plot_individual_exam(
    course_code: str,
    session: str,
):
    """
    Plots cheating data for each student in a given exam, including cheating scores
    and images with bounding boxes indicating suspicious activities.

    Parameters:
    - exam_id (str): The unique identifier for the exam to retrieve and display student cheating data.

    The function performs the following steps:
    1. Retrieves a list of cheating reports for the specified exam from the suspiciousReportRespository.
    2. For each student, it plots their cheating scores as a bar graph and displays a sample image with bounding boxes.
    3. Displays a summary with the student's ID, average cheating score, and the number of suspicious activities detected.
    """
    exam_id = search_exam(session, course_code)
    services.plot_student_data(exam_id)


@router.get("/available_cameras")
def available_cameras():
    available_cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(f"Camera {i} is available.")
            cap.release()
    return available_cameras


@router.post("/video")
async def video(
    course_code: str,
    session: str,
    camera_id: int | None = None,
    use_timer: bool = False,
    record_video: bool = False,
    video_source: UploadFile = File(None),
):
    if video_source is None and camera_id is None:
        raise RequestValidationError("Either camera and video_source cannot be none")

    temp_video_path = None

    try:
        if camera_id:
            video_source = None
        if video_source:

            file_bytes = await video_source.read()

            # Save the bytes to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(file_bytes)
                temp_video_path = temp_video.name

        exam_id = search_exam(session, course_code.upper())

        # video_path = temp_video_path if temp_video_path else None

        ais = AIInvigilatingSystem(temp_video_path, camera_id, record_video, use_timer)
        ais(exam_id)
        return "Success"
    except ValueError as e:
        print({"error": str(e)})
        return {"error": str(e)}

    except Exception as e:
        print({"error": str(e)})
        return {"error": str(e)}


@router.post("/attendance/save_detections/")
async def save_detection(
    course_code: str,
    session: str,
    video_source: UploadFile = File(None),
    camera_id: int | None = None,
):
    """
    Processes a video source to detect objects (students) using a YOLO model and saves the detections.

    Parameters:
    - exam_id (str): A unique identifier for the exam, used to label saved data.
    - video_source (str or None): Path to the video file to process. Set to None if using a camera feed.
    - camera_id (int or None): ID of the camera to use if video_source is None.

    Returns:
    - return_data (dict or None): A dictionary containing exam data including the exam ID,
      path to the saved first frame, number of detected students, and their coordinates.
      Returns None if the operation is canceled or no data is saved.

    The function performs the following steps:
    1. Initializes a YOLO model for object detection.
    2. Opens the video source (file or camera) and checks for successful access.
    3. Reads the first frame from the video source and performs error checking.
    4. Resizes the frame to a standard resolution (1920x1080).
    5. Runs the YOLO model on the first frame to detect objects (students).
    6. Draws bounding boxes and labels on the detected objects.
    7. Displays the processed frame in a window and waits for a user input:
       - Press 's' to save the frame and data to a file and database.
       - Press 'q' to cancel and discard the data.
    8. Releases video resources and closes all OpenCV windows.

    The function saves the data and returns it if 's' is pressed, otherwise, it cancels the operation.
    """
    if video_source is None and camera_id is None:
        raise RequestValidationError("Either camera and video_source cannot be none")

    temp_video_path = None
    exam_id = search_exam(session, course_code)
    if camera_id:
        video_source = None

    if video_source:

        file_bytes = await video_source.read()

        # Save the bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(file_bytes)
            temp_video_path = temp_video.name

    return services.save_detections(exam_id, temp_video_path, camera_id)

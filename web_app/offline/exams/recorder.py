import os
import cv2
from .schema import VideoRecordingSchema
# from datetime import datetime
from ..repository import videoRecordingRespository


class VideoRecorder:
    def __init__(self, video_save_dir, new_width, new_height):
        self.VIDEO_SAVE_DIR = video_save_dir
        self.new_width = new_width
        self.new_height = new_height
        self.file_name = None

    def save_to_db(self, exam_id, date):
        # print(self.file_name)
        vid_dur, width, height = self.get_video_duration()
        vid = VideoRecordingSchema(
            exam_id=exam_id,
            timestamp=date,
            file_path=self.file_name,
            duration=vid_dur,
            resolution=(width, height),
        )
        videoRecordingRespository.insert_one(vid.__dict__)

    def out(self, exam_id):
        # Ensure the directory exists
        os.makedirs(self.VIDEO_SAVE_DIR, exist_ok=True)

        # Append .mp4 to exam_id to ensure correct file extension
        self.file_name = os.path.join(self.VIDEO_SAVE_DIR, f"{exam_id}.mp4")

        # Define the codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Create the VideoWriter object
        return cv2.VideoWriter(
            self.file_name, fourcc, 20.0, (self.new_width, self.new_height)
        )

    def get_video_duration(self):
        # Open the video file
        cap = cv2.VideoCapture(self.file_name)

        # Check if the video file opened successfully
        if not cap.isOpened():
            print("Error: Could not open the video file.")
            return None

        # Get the total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get the frames per second (fps)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate duration in seconds
        duration = total_frames / fps

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Release the video capture object
        cap.release()

        return duration, width, height

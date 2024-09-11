from pathlib import Path

import cv2
from config import IMG_IN_DIR, IMG_OUT_DIR, CLASS_DIRS

import cv2
import os

width = 224
height = 224


def get_unique_filename(directory, base_filename, extension):
    counter = 1
    unique_filename = f"{base_filename}.{extension}"
    while os.path.exists(os.path.join(directory, unique_filename)):
        unique_filename = f"{base_filename}_{counter:04d}.{extension}"
        counter += 1
    return unique_filename


def save_frames(video_path, file_name, interval=1):
    """
    Extracts and saves frames from a video at a specified interval.

    Parameters:
    - video_path: str, path to the video file.
    - output_dir: str, directory to save the frames.
    - interval: int, interval between frames to be saved. Default is 1 (save every frame).

    Returns:
    - None
    """

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame at specified intervals
        if frame_count % interval == 0:
            path = os.path.join(IMG_OUT_DIR, file_name)
            frame_filename = get_unique_filename(
                path, f"{saved_frame_count:04d}", "jpg"
            )
            full_path = os.path.join(IMG_OUT_DIR, file_name, frame_filename)
            print(frame_filename)
            _pad_image = pad_image(frame, (640, 640))
            cv2.imwrite(full_path, _pad_image)
            # break
            print(f"Image ----------- {frame_filename} saved")
            saved_frame_count += 1

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Total frames saved: {saved_frame_count}")
    return saved_frame_count


def pad_image(frame, target_size):
    old_size = frame.shape[:2]

    ratio = float(target_size[0]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    image = cv2.resize(frame, (new_size[1], new_size[0]))

    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return new_image


def resize_frame(frame, percent):
    # Calculate the new dimensions
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dimensions = (width, height)

    # Resize the frame
    resized_frame = cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)
    return resized_frame

def run():
    ...
    
def main():

    file_names = []
    file_path = Path(r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\media\datasets\frame_classifier\images\train")
    paths = [file.as_posix() for file in file_path.iterdir() if file.is_file()]
    for path in paths:
        file_name = os.path.basename(path).removesuffix(".mp4")
        if file_name[:3] == "exc":

            frame = save_frames(path, "cheating", 10)
        elif file_name[:3] == "bac":

            frame = save_frames(path, "cheating", 10)
        elif file_name[:3] == "pee":

            frame = save_frames(path, "cheating", 10)
        elif file_name[:3] == "nor":

            frame = save_frames(path, "non_cheating", 10)
        elif file_name[:3] == "mov":

            frame = save_frames(path, "cheating", 10)
        else:
            continue

    print(file_names)


if __name__ == "__main__":
    main()

import os
import cv2
from deepface import DeepFace
from config import IMG_OUT_DIR
import numpy as np
import matplotlib.pyplot as plt


def detect_faces(path):
    try:
        detect = DeepFace.extract_faces(path, detector_backend="retinaface")
        print(detect)

        # Read the original image
        original_image = cv2.imread(path)

        # Convert the original image to RGB for plotting with matplotlib
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        for face in detect:
            # Get facial area coordinates
            x, y, w, h = (
                face["facial_area"]["x"],
                face["facial_area"]["y"],
                face["facial_area"]["w"],
                face["facial_area"]["h"],
            )

            # Draw rectangle on the original image
            cv2.rectangle(original_image_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Plot the image with all bounding boxes
        plt.imshow(original_image_rgb)
        plt.axis("off")  # Hide axes
        plt.show()

    except Exception as e:
        print(f"Error detecting face: {e}")


file = os.path.join(IMG_OUT_DIR, "exchange_2_0008.jpg")
print(file)
detect_faces(file)

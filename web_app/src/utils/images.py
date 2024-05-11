import os
import io
from typing import List
import uuid
from PIL import Image
from datetime import datetime
import numpy as np
import cv2
from deepface import DeepFace
from fastapi import UploadFile
from ..repository import repository

SAVE_DIRECTORY = "media/profilephotos/"


def compare_images(image: Image.Image):
    users = repository.find_many("user")
    for user in users:
        embeddings = user["photo_embed"]
        result = DeepFace.verify(embeddings, image)
        print(result)


def save_file(file: UploadFile):
    # paths = []
    # for file in files:
    filename = file.filename
    file_path = os.path.join(SAVE_DIRECTORY, filename)
    with open(file_path, "wb") as file_data:
        file_data.write(file.file.read())
        # paths.append(file_path)
    return file_path


def save_image(images: List[Image.Image]) -> str:
    upload_paths = []
    for image in images:
        if not os.path.exists(SAVE_DIRECTORY):
            os.makedirs(SAVE_DIRECTORY)
        filename = f"media/profilephotos/{str(uuid.uuid4())}"
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_filename = f"{filename}_{timestamp}.jpg"
        os.makedirs(SAVE_DIRECTORY, exist_ok=True)
        upload_path = os.path.join(SAVE_DIRECTORY, unique_filename)
        image.save(upload_path)
        upload_paths.append(upload_path)
    return upload_paths


def preprocess_and_embed(image_data):
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Display the image
    cv2.imshow("Input Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Proceed with embedding extraction
    embeddings = DeepFace.represent(image, model_name="Facenet")
    return embeddings

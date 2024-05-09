import os
from typing import List
import uuid
from datetime import datetime
import numpy as np
import cv2
from deepface import DeepFace
from fastapi import UploadFile

UPLOAD_DIRECTORY = "..../media/facial/database"


def save_image(upload_files: List[UploadFile]) -> str:
    upload_paths = []
    for upload_file in upload_files:
        filename, file_extension = os.path.splitext(upload_file.filename)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_filename = f"{filename}_{uuid.uuid4().hex}_{timestamp}{file_extension}_"
        upload_path = os.path.join(UPLOAD_DIRECTORY, unique_filename)

        with open(upload_path, "wb") as buffer:
            buffer.write(upload_file.file.read())

        upload_paths.append(upload_path)

    return upload_paths


def preprocess_and_embed(image_data):
    nparr = np.frombuffer(image_data, np.uint8)

    image = cv2.imdecode(
        nparr, cv2.IMREAD_COLOR
    ) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, (96, 96))

    embeddings = DeepFace.represent(resized, model_name="Facenet")

    return embeddings

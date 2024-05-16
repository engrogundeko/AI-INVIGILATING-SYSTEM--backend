from typing import List

import cv2
from deepface import DeepFace

image_path = ".../media/datasets/facial/database"


class FacialRecogntion:
    def detect_faces(self, frame):
        return DeepFace.extract_faces(frame)

    def preprocess_and_embed(image_path: str):
        image = cv2.imread(image_path)
        embeddings = DeepFace.represent(image, model_name="Facenet")
        return embeddings


if __name__ == "__main__":
    pass

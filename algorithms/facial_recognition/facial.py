from typing import List

import cv2
from deepface import DeepFace

image_path = ".../media/datasets/facial/database"


def preprocess_and_embed(image_path: str):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (96, 96))

    embeddings = DeepFace.represent(resized, model_name="Facenet")

    return embeddings


cv2.destroyAllWindows()

if __name__ == "__main__":
    pass

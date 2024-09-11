import os
from uvicorn import run

from fastapi import FastAPI
from deepface import DeepFace


app = FastAPI()

@app.post(status_code=200)
async def detect_student(data: dict):
    print(data)
    face = DeepFace.extract_faces(data.get("cropped_image"))
    if face:
        ...
    else:
        ...


if __name__ == "__main__":
    run("main:app", reload=True, host=8001)

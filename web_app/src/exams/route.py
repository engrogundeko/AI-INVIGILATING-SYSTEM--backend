import base64
from pathlib import Path
from fastapi.responses import StreamingResponse

from ..utils.images import preprocess_and_embed, compare_images
from .schema import VideoSchema

import cv2
import numpy as np
from ultralytics import YOLO
from fastapi import APIRouter, Header, Response, WebSocket

model = YOLO("yolov8n.pt")
router = APIRouter(tags=["Exams"], prefix="/exams")


def generate_frame():
    camera = cv2.VideoCapture(
        0
    )  # Initialize camera (assuming camera is available at index 0)
    try:
        while True:
            # Read frame from camera
            success, frame = camera.read()
            if not success:
                break

            # Perform object detection
            results = model.track(frame, persist=True)

            ret, buffer = cv2.imencode(".webp", frame)
            if not ret:
                break

            # Yield encoded frame as bytes
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
    finally:
        # Release camera when done
        camera.release()


@router.get("/video")
def video():
    return StreamingResponse(
        generate_frame(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


# @router.websocket("/video-stream")
# async def video_stream(websocket: WebSocket):
#     await websocket.accept()

#     while True:
#         data = await websocket.receive_text()
#         if data is not None:
#             img_bytes = base64.b64decode(data.split(",")[1])
#             # embeddings = preprocess_and_embed(img_bytes)
#             # image_path = compare_images(embeddings)

#             nparr = np.frombuffer(img_bytes, np.uint8)
#             if len(nparr) >= 1:
#                 img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#                 results = model.track(img, persist=True)
#                 bounding_boxes = [result.orig_img for result in results if result]
#                 print(bounding_boxes)
#                 # await websocket.send_text(bounding_boxes)
#                 # break

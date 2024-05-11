import base64
import json
from .schema import VideoSchema

import cv2
import numpy as np
from ultralytics import YOLO
from fastapi import APIRouter, Request, WebSocket

model = YOLO("yolov8n.pt")
router = APIRouter(tags=["Exams"], prefix="/exams")


@router.post("/stream")
async def create_student(payload: VideoSchema):
    try:
        base64_data = payload.image.split(",")[1]
        # Convert image data from string to bytes
        image_bytes = base64.b64decode(base64_data)

        # Convert bytes to numpy array

    except Exception as e:
        print(f"Error: {e}")


@router.websocket("/video-stream")
async def video_stream(websocket: WebSocket):
    await websocket.accept()

    while True:
        data = await websocket.receive_text()
        if data is not None:
            # Decode base64 image data
            img_bytes = base64.b64decode(data.split(",")[1])

            nparr = np.frombuffer(img_bytes, np.uint8)
            if len(nparr) >= 1:
                # # Decode the numpy array into an image
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                results = model.track(img, persist=True)
                bounding_boxes = []
                for result in results:
                    # Assuming each detection result has 'xyxy' attribute
                    bounding_boxes.append(result.tojson(normalize=False))

                # Convert bounding box coordinates to JSON
                bounding_boxes_json = json.dumps(bounding_boxes)

                # Send bounding box coordinates to frontend
                await websocket.send_text(bounding_boxes_json)

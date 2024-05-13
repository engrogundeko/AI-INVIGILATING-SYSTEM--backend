from .algorithms import generate_frame

from ultralytics import YOLO
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

model = YOLO("yolov8n.pt")
router = APIRouter(tags=["Exams"], prefix="/exams")


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

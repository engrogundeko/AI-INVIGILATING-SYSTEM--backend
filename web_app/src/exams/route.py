from bson import ObjectId
from .schema import AttendanceSchema, CreateExamSchema, GetExamAttendance
from .algorithms import generate_frame
from .services import take_attendance
from ..repository import (
    examRespository,
    examAttedanceRespository,
    examRegistrationRespository,
    courseRespository,
    userRespository,
)

from ultralytics import YOLO
from fastapi import APIRouter, Path
from fastapi.responses import StreamingResponse, JSONResponse

model = YOLO("yolov8n.pt")
router = APIRouter(tags=["Exams"], prefix="/exams")


@router.get("/video")
def video():
    return StreamingResponse(
        generate_frame(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.post("")
def create_exam(payload: CreateExamSchema):
    exam_insert = examRespository.insert_one(payload.__dict__)
    exam = examRespository.find_one({"_id": exam_insert.inserted_id})
    return JSONResponse(exam)


@router.get("/{exam_name}")
def get_exam(exam_name: str = None):
    if exam_name:
        return examRespository.find_one({"name": exam_name})
    return examRespository.find_many()


@router.post("/course")
def create_course(payload: CreateExamSchema):
    exam_insert = courseRespository.insert_one(payload.__dict__)
    return JSONResponse(exam_insert)


@router.post("/register")
def create_exam(payload: CreateExamSchema):
    exam_insert = examRegistrationRespository.insert_one(payload.__dict__)
    return JSONResponse(exam_insert)


@router.get("/attendance")
def get_attendance(payload: GetExamAttendance):
    exam = examRespository.find_one({"exam_id": payload.exam_id})
    attendance = examAttedanceRespository.find_one({"_id": exam["attendace_id"]})
    return JSONResponse(attendance)


@router.post("/attendance/{exam_id}/verify_students/")
async def verify_students(exam_id: ObjectId, payload: AttendanceSchema):
    attendance_id = take_attendance(payload.image, exam_id)
    attendance = examAttedanceRespository.find_one({"_id": attendance_id})
    student = userRespository.find_one({"_id": attendance["student_id"]})
    confirmation = {
        "id": attendance_id,
        "student": {
            "id": student["_id"],
            "matric_no": student["matric_no"],
            "image_path": student["image_path"],
        },
        "status": "PRESENT",
    }
    return JSONResponse(confirmation)


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

from typing import List
from .model import Course, Room, ExamRegistration, Exam
from .schema import (
    AttendanceSchema,
    CreateRoom,
    RegisterExamSchema,
    CreateCourseShema,
    ExamAttendance,
    CreateExamSchema,
    GetExamAttendance,
)
from .ai import AIInvigilatingSystem

# from .services import take_attendance
from ..repository import (
    examRespository,
    examAttedanceRespository,
    examRegistrationRespository,
    courseRespository,
    userRespository,
)

# from .ais import frame_generator
from ..authentication.permission import role_required
from ..accounts.model import Role
from fastapi import APIRouter, Path, Query, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import Request

# model = YOLO("yolov8n.pt")
router = APIRouter(tags=["Exams"], prefix="/exams")

vid = r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\media\datasets\converted\exchange_6.mp4"


# @role_required([Role.admin])
@router.get("/video/{exam_id}")
async def video(exam_id: str):
    try:

        # def generate():
        ais = AIInvigilatingSystem(vid, None, False)
        ais(exam_id)
        # continue
        # print(frame)
        # ret, buffer = cv2.imencode(".webp", frame)
        # if ret:
        #     yield (
        #         b"--frame\r\n"
        #         b"Content-Type: image/webp\r\n\r\n" + buffer.tobytes() + b"\r\n"
        #     )
        return "Success"
        # return StreamingResponse(
        #     generate(), media_type="multipart/x-mixed-replace; boundary=frame"
        # )
    except Exception as e:
        print({"error": str(e)})
        return {"error": str(e)}


@router.post("", response_model=Exam)
def create_exam(payload: CreateExamSchema):
    return examRespository.insert_one(payload.__dict__)


@router.get("", response_model=List[Exam])
def get_exam():
    return examRespository.find_many()


# @router.post("")
# def create_exam(payload: CreateExamSchema):
#     return examRespository.insert_one(payload.__dict__)


@router.get("/")
def get_exam(exam_name: str = Query(...)):
    if exam_name:
        return examRespository.find_one({"name": exam_name})
    return examRespository.find_many()


@router.post("/room", response_model=Room)
def create_room(payload: CreateRoom):
    return courseRespository.insert_one(payload.__dict__)


@router.get("/course", response_model=List[Course])
def get_course():
    return courseRespository.find_many()


@router.post("/register", response_model=ExamRegistration)
def register_exam(payload: RegisterExamSchema):
    return examRegistrationRespository.insert_one(payload.__dict__)


@router.get("/attendance")
def get_attendance(payload: GetExamAttendance):
    exam = examRespository.find_one({"exam_id": payload.exam_id})
    attendance = examAttedanceRespository.find_one({"_id": exam["attendace_id"]})
    return JSONResponse(attendance)


@router.post(
    "/attendance/{exam_id}/verify_students/", response_model=List[ExamAttendance]
)
async def verify_students(exam_id: str, file: UploadFile):

    attendance = take_attendance(file, exam_id)
    return attendance


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

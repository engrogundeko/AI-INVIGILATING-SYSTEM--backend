import base64
from ..utils.images import preprocess_and_embed
from .services import create_student_service
from .schema import StudentInSchema, ImagePayload

from fastapi import APIRouter, File, UploadFile

router = APIRouter(tags=["Accounts"], prefix="/accounts")


@router.post("/")
async def create_student(payload: StudentInSchema):
    return create_student_service(payload)


@router.post("/verify-student")
async def verify_student(payload: ImagePayload):
    base64_data = payload.image.split(",")[1]
    image_bytes = base64.b64decode(base64_data)
    # print(image_bytes)
    embeddings = preprocess_and_embed(image_bytes)
    print(embeddings)
    # return {"user": data}

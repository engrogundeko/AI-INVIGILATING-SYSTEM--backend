import io
import base64
from PIL import Image
from ..utils.images import preprocess_and_embed
from ..utils.images import save_image, compare_images
from .services import create_student_service
from .schema import StudentInSchema, ImagePayload

from fastapi import APIRouter, File, UploadFile, Depends

router = APIRouter(tags=["Accounts"], prefix="/accounts")


@router.post("/")
async def create_student(payload: StudentInSchema = Depends()):
    return create_student_service(payload)


@router.post("/verify-student")
async def verify_student(payload: ImagePayload):
    base64_data = payload.image.split(",")[1]
    image_bytes = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(image_bytes))
    embeddings = preprocess_and_embed(image_bytes)
    image_path = compare_images(embeddings)

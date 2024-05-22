import io
import base64
from PIL import Image
from ..repository import userRespository
from ..utils.images import preprocess_and_embed
from ..utils.images import compare_images
from .schema import StudentInSchema, ImagePayload, StudentOutSchema

from fastapi import APIRouter,  Depends
from fastapi.responses import JSONResponse

router = APIRouter(tags=["Accounts"], prefix="/accounts")


@router.post("/", response_model=StudentOutSchema)
async def create_student(payload: StudentInSchema = Depends()):
    return userRespository.insert_one(payload.__dict__)


@router.post("/verify-student")
async def verify_student(payload: ImagePayload):
    base64_data = payload.image.split(",")[1]
    image_bytes = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(image_bytes))
    embeddings = preprocess_and_embed(image_bytes)
    image_path = compare_images(embeddings)

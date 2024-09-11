import io
import base64
from typing import List
from PIL import Image
from ..repository import userRespository
from ..utils.images import compare_images
from ..utils.images import preprocess_and_embed
from ..authentication.auth import get_access_token
from ..utils import image_utils
from .schema import StudentInSchema, ImagePayload, StudentOutSchema, LoginSchema

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

router = APIRouter(tags=["Accounts"], prefix="/accounts")


@router.post("/", response_model=StudentOutSchema)
async def create_student(payload: StudentInSchema = Depends()):
    return userRespository.insert_one(payload.__dict__)


@router.get("/", response_model=List[StudentOutSchema])
def get_all_students():
    return userRespository.find_many()

@router.delete("")
def delete_student():
    students: List[dict] = userRespository.find_many()
    for student in students:
        img = student.get("image")
        print(img)
        if img.split(".")[1] == "jfif" :
            userRespository.delete_one({"_id": student["_id"]})
            print(student)
            
    


@router.post("/login")
async def login(payload: LoginSchema):
    return get_access_token(payload.email)


@router.post("/logout")
async def logout(payload: LoginSchema): ...


@router.post("/verify-email")
async def verify_email(payload): ...


@router.post("/sign_up")
async def sign_up(): ...


@router.post("/verify-student")
async def verify_student(payload: ImagePayload):
    base64_data = payload.image.split(",")[1]
    image_bytes = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(image_bytes))
    embeddings = preprocess_and_embed(image_bytes)
    image_path = compare_images(embeddings)

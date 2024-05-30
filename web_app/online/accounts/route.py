
from ..repository import userRespository
from .schema import StudentInSchema, StudentOutSchema

from fastapi import APIRouter,  Depends
from fastapi.responses import JSONResponse

router = APIRouter(tags=["Accounts"], prefix="/accounts")


@router.post("/", response_model=StudentOutSchema)
async def create_student(payload: StudentInSchema = Depends()):
    return userRespository.insert_one(payload.__dict__)

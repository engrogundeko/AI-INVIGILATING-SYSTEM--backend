import io

from .exams.route_2 import router as exam_router
from .exams.route import router as ai_router
from .exams.route import search_exam
from .exams.services import evaluate_cheating
from .exams.pdf import generate_report
from .repository import (
    reportRespository,
    analysedRepo,
)

# from bson import ObjectId
from uvicorn import run
from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import StreamingResponse

from pathlib import Path
from fastapi.staticfiles import StaticFiles


app = FastAPI(title="Automatic Invigilating System")


# import os
@app.get("/analyse_exam")
async def analyse_exam(course_code: str, session: str, request: Request):

    exam_id = search_exam(session, course_code)
    data = reportRespository.find_one({"exam_id": exam_id})
    exam_metrics = data.get("exam_metrics")
    # if exam_metrics:
    #     return "Already analysed"

    return evaluate_cheating(exam_id)


@app.get("/pdf")
async def generate_pdf(course_code: str, session: str, request: Request):
    exam_id = search_exam(session, course_code)
    data = reportRespository.find_one({"exam_id": exam_id})
    if data is None:
        raise HTTPException(status_code=404, detail=f"{exam_id} is not valid")

    analysis = analysedRepo.find_one({"exam_id": exam_id})
    if analysis is None:
        raise HTTPException(
            status_code=404, detail="You have not yet analysed the exam"
        )

    if analysis["is_printed"]:
        pdf_path = analysis["pdf_path"]
    else:
        pdf_path = generate_report(exam_id)
        analysedRepo.update_one(
            {"exam_id": exam_id}, {"pdf_path": pdf_path, "is_printed": True}
        )

    with open(pdf_path, "rb") as file:
        pdf_stream = io.BytesIO(file.read())

    return StreamingResponse(
        pdf_stream,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{course_code}_exam_report.pdf"'
        },
    )


app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.parent.absolute() / "static"),
    name="static",
)
app.include_router(ai_router)
app.include_router(exam_router)
# app.include_router(server_router)
# app.include_router(accounts_router)


if __name__ == "__main__":
    run("main:app", reload=True)

from fastapi import FastAPI
from uvicorn import run
from .accounts.route import router as accounts_router
from .server.route import router as server_router
from .exams.route import router as exam_router

import os

app = FastAPI()

app.include_router(exam_router)
app.include_router(server_router)
app.include_router(accounts_router)


if __name__ == "__main__":
    run("main:app", reload=True)

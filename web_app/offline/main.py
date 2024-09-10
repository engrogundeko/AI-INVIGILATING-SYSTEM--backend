from fastapi import FastAPI
from uvicorn import run
# from .accounts.route import router as accounts_router
# from .server.route import router as server_router
from .exams.route_2 import router as exam_router
from .exams.route import router as ai_router

# from pathlib import Path
# from fastapi.staticfiles import StaticFiles

# import os

app = FastAPI(title="Automatic Invigilating System")
# app.mount(
#     "/static",
#     StaticFiles(directory=Path(__file__).parent.parent.absolute() / "static"),
#     name="static",
# )
app.include_router(ai_router)
app.include_router(exam_router)
# app.include_router(server_router)
# app.include_router(accounts_router)


if __name__ == "__main__":
    run("main:app", reload=True)

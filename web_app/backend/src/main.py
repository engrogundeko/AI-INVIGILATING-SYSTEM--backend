import subprocess
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run
from .accounts.route import router as accounts_router

import os

print(os.path.exists("./static"))
app = FastAPI()
app.include_router(accounts_router)

app.mount("/static", StaticFiles(directory="./static"), name="static")
templates = Jinja2Templates(directory="./static/templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    run("main:app", reload=True)

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

router = APIRouter()
router.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="static/templates")


@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/live", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("live.html", {"request": request})


@router.get("/livestream", response_class=HTMLResponse)
async def stream_exams(request: Request):
    return templates.TemplateResponse("livestream.html", {"request": request})

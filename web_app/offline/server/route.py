from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from fastapi.templating import Jinja2Templates

router = APIRouter()

templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/live", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("live.html", {"request": request})


@router.get("/livestream", response_class=HTMLResponse)
async def stream_exams(request: Request):
    return templates.TemplateResponse("livestream.html", {"request": request})


@router.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@router.get("/login", response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@router.get("/verify_password_code", response_class=HTMLResponse)
async def verify_password_code(request: Request):
    return templates.TemplateResponse("verify_password_code.html", {"request": request})


@router.get("/signuut", response_class=HTMLResponse)
async def signup(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


@router.get("/forgot_password", response_class=HTMLResponse)
async def forgot_password(request: Request):
    return templates.TemplateResponse("forgot_password.html", {"request": request})


@router.get("/change_password", response_class=HTMLResponse)
async def change_password(request: Request):
    return templates.TemplateResponse("change_password.html", {"request": request})


@router.get("/verify_password", response_class=HTMLResponse)
async def verify_password(request: Request):
    return templates.TemplateResponse("verify_password.html", {"request": request})


@router.get("/add_new_course", response_class=HTMLResponse)
async def add_new_course(request: Request):
    return templates.TemplateResponse("add_new_course.html", {"request": request})


@router.get("/add_new_exam", response_class=HTMLResponse)
async def add_new_exam(request: Request):
    return templates.TemplateResponse("add_new_exam.html", {"request": request})


@router.get("/add_new_room", response_class=HTMLResponse)
async def add_new_room(request: Request):
    return templates.TemplateResponse("add_new_room.html", {"request": request})


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@router.get("/student_list", response_class=HTMLResponse)
async def student_list(request: Request):
    return templates.TemplateResponse("student_list.html", {"request": request})


@router.get("/teacher_details", response_class=HTMLResponse)
async def teacher_details(request: Request):
    return templates.TemplateResponse("teacher_details.html", {"request": request})


@router.get("/exam_details", response_class=HTMLResponse)
async def exam_details(request: Request):
    return templates.TemplateResponse("exam_details.html", {"request": request})


@router.get("/exam_list", response_class=HTMLResponse)
async def exam_list(request: Request):
    return templates.TemplateResponse("exam_list.html", {"request": request})


@router.get("/user_dashboard", response_class=HTMLResponse)
async def user_dashboard(request: Request):
    return templates.TemplateResponse("user_dashboard.html", {"request": request})


@router.get("/students_details", response_class=HTMLResponse)
async def students_details(request: Request):
    return templates.TemplateResponse("students_details.html", {"request": request})


from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from uvicorn import run
import weasyprint
import io


app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/pdf")
async def generate_pdf(request: Request):
    html = templates.get_template("report.html").render({"request": request, "data": data})
    
    # Convert HTML to PDF
    pdf = weasyprint.HTML(string=html).write_pdf()
    
    # Create a streaming response
    pdf_stream = io.BytesIO(pdf)
    return StreamingResponse(pdf_stream, media_type='application/pdf', headers={'Content-Disposition': 'attachment; filename="report.pdf"'})
if __name__ == "__main__":
    run("main:app", reload=True, host=8001)

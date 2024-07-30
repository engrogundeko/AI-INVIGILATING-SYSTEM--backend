from ..repository import (
    userRespository,
    examRegistrationRespository,
    roomRespository,
    courseRespository,
)
from httpx import AsyncClient


class EmailHandler:
    api_key = ""
    api_secret = ""
    url = f"https://mssn-mailer.onrender.com/api{api_secret}send_mail{api_key}"

    async def send(self, data: dict):
        student_id = data["student_id"]
        exam_id = data["exam_id"]
        img = data["cropped_image"]

        student = userRespository.find_one({"_id": student_id})
        exam = examRegistrationRespository.find_one({"_id": exam_id})
        room = roomRespository.find_one({"_id": exam["room_id"]})
        course = courseRespository.find_one({"_id": exam["course_id"]})

        exam_date = exam["date"]
        room_name = room["name"]
        to = student["email"]
        student_name = student["first_name"]
        course_name = course["name"]

        html = self.get_email_template(
            class_room=room_name,
            to=to,
            student_name=student_name,
            date_of_exam=exam_date,
            name_of_exam=course_name,
        )

        json = {
            "html": html,
            "subject": "Incident Report - Cheating During Exam",
            "to": [
                to,
            ],
        }

        async with AsyncClient() as async_client:
            request = await async_client.post(url=self.url, data=json, files=img)
            if request.status_code == 200:
                re = request.json()
                return re

    @staticmethod
    def get_email_template(class_room, to, student_name, date_of_exam, name_of_exam):
        return f"""

            Dear {student_name},

            I hope this letter finds you well. I am writing to 
            formally report an incident of academic dishonesty 
            that occurred during the {name_of_exam} on {date_of_exam}.

            During the examination, it was observed that {student_name}, 
            a student in {class_room}, engaged in cheating. 
            Specifically, the student was found Cheating, 

            As part of the evidence, I have attached an image that clearly shows 
            {student_name} engaging in the act of cheating. 
            This image was taken "through surveillance cameras," 
            and clearly indicates the breach of academic integrity.

            This behavior is a serious violation of our institution's code 
            of conduct and academic policies. As such, I recommend that 
            appropriate disciplinary actions be taken in accordance with our academic integrity policy.

            Please find the attached image for your reference.
            I am available for any further discussions or to provide 
            additional information if needed. Your prompt attention 
            to this matter would be greatly appreciated.

            Thank you for your understanding and cooperation.

            Sincerely,
                    """

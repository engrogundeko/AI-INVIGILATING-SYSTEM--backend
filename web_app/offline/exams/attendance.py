import io
import os
from bson import ObjectId
from datetime import datetime
from ..config import (
    AIConstant, 
    AIS_MODEL, 
    UNRECOGNISED_FACE, 
    UNDETECTED_FACE)

from ..repository import (
    userRespository,
    examRegistrationRespository,
    examAttedanceRespository,
    examLocationRepo,
)
from .schema import ExamLocation, ExamAttendance, Location
from ..utils import image_utils
from ..utils import image_utils

from deepface import DeepFace
from ultralytics import YOLO
import numpy as np
from PIL import Image

def take_attendance(image, exam_id):
    file_content = image.file.read()

    # Convert file content to an image
    img = Image.open(io.BytesIO(file_content))
    arr = np.array(img)
    student_locations = []
    undetected_faces = []
    unrecognised_faces = []

    model = YOLO(AIS_MODEL)
    img, _ = image_utils.load_image(arr)
    results = model.predict(img)
    for re in results:
        for bbox in re.boxes.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            cropped_frame = img[y1:y2, x1:x2]
            exam_reg = examRegistrationRespository.find({"exam_id": exam_id})
            for exam in exam_reg:
                student_id = exam["student_id"]
                student = userRespository.find_one({"_id": ObjectId(student_id)})
                img_path = rf"{student["image"]}"
                is_present = examAttedanceRespository.find_one({"student_id": student_id, "exam_id": exam_id})
    
                if is_present is not None and is_present["status"] == "PRESENT":
                    continue
                try:
                    result = DeepFace.verify(img1_path=cropped_frame, img2_path=img_path, detector_backend="yolov8")
                except ValueError:
                    filename = os.path.join(UNRECOGNISED_FACE, datetime.now().strftime("%M:%S"))
                    img_path = AIConstant.save_image(cropped_frame, filename)
                    unrecognised_faces.append(img_path)
                    continue
                except Exception:
                    filename = os.path.join(UNDETECTED_FACE, datetime.now().strftime("%M:%S"))
                    img_path = AIConstant.save_image(cropped_frame, filename)
                    undetected_faces.append(img_path)
                    continue
                
                if result["verified"]:
                    student_locations.append(
                        Location(student_id=student_id, coordinate=(x1, y1, x2, y2)).__dict__)
                    attendance = ExamAttendance(
                        status="PRESENT",
                        exam_id=exam_id,
                        student_id=student_id,
                        coordinate=(x1, y1, x2, y2),
                    )
                    examAttedanceRespository.insert_one(attendance.__dict__)
                    
    exam = ExamLocation(
        exam_id=exam_id, 
        locations=student_locations, 
        unrecognised_faces=unrecognised_faces, 
        undetected_faces=undetected_faces
        )
    
    examLocationRepo.insert_one(exam.__dict__)
    return list(examAttedanceRespository.find({"exam_id": exam_id}))

from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    Image,
)

from .services import plot_required_graphs
from ..repository import (
    studentDataRepo,
    suspiciousReportRespository,
    examRespository,
    courseRespository,
    facultyRepository,
    departmentRepository,
    sessionRepository,
    roomRespository,
    reportRespository,
)
from bson import ObjectId
from ..config import REPORT_FOLDER
import os


# exam_id = None
def generate_pdf_report(
    file_name,
    exam_id,
    detection,
    exam_list: dict,
    key_metrics: dict,
    student_analysis,
    model_performance,
    recommendations,
    image_paths: dict,
    anomaly_analysis: dict,
    sample_images,
):
    # Create PDF Document
    pdf = SimpleDocTemplate(file_name, pagesize=A4)

    # Sample styles and story (list to hold content)
    styles = getSampleStyleSheet()
    story = []

    # Title of the document
    story.append(
        Paragraph("Automatic Invigilation System - Analytical Report", styles["Title"])
    )
    story.append(Spacer(1, 12))

    # Exam Setup Image
    story.append(Paragraph("Exam Setup Image", styles["Heading3"]))
    exam_setup_image = Image(detection, width=6 * inch, height=4 * inch)
    story.append(exam_setup_image)
    # else:
    #     story.append(Paragraph("No setup image available.", styles["Normal"]))
    story.append(Spacer(1, 20))

    # Overview Section
    story.append(Paragraph("1. Overview", styles["Heading2"]))
    # for key, value in exam_data.items():
    #     story.append(Paragraph(f"<b>{key}:</b> {value}", styles["Normal"]))
    # story.append(Spacer(1, 12))

    table_data = []
    table_data.extend(exam_list)

    table = Table(table_data, colWidths=[4.0 * inch] + [4 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 24))

    # Average Cheating Plot
    story.append(Paragraph("Average Cheating Plot", styles["Heading3"]))
    if "average_cheat" in image_paths:
        avg_path = Image(image_paths["average_cheat"], width=6 * inch, height=4 * inch)
        story.append(avg_path)
    else:
        story.append(Paragraph("No average cheating plot available.", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Cheating Rate Plot
    story.append(Paragraph("Cheating Rate Plot", styles["Heading3"]))
    if "rated_cheat" in image_paths:
        rate_image = Image(image_paths["rated_cheat"], width=6 * inch, height=4 * inch)
        story.append(rate_image)
    else:
        story.append(Paragraph("No cheating rate plot available.", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Key Metrics Section
    story.append(Paragraph("2. Key Metrics", styles["Heading2"]))
    # sample_images = key_metrics.pop("sample_images", [])[:5]
    for image in sample_images:
        story.append(Image(image, width=4 * inch, height=3 * inch))
        story.append(Spacer(1, 5))

    table_data = []
    # for row in key_metrics:
    #     cleaned_row = [str(item).replace('\xa0', ' ') for item in row]  # Replace non-breaking spaces
    table_data.extend(key_metrics)
    table = Table(table_data, colWidths=[4.0 * inch] + [4.0 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 12))

    # Confusion Matrix Chart
    story.append(Paragraph("Confusion Matrix Chart", styles["Heading3"]))
    if "confusion_matrix" in image_paths:
        insights_chart_image = Image(
            image_paths["confusion_matrix"], width=6 * inch, height=4 * inch
        )
        story.append(insights_chart_image)
    else:
        story.append(
            Paragraph("No confusion matrix chart available.", styles["Normal"])
        )
    story.append(Spacer(1, 12))

    # Anomaly Analysis Section
    story.append(Paragraph("5. Anomaly Analysis", styles["Heading2"]))
    story.append(Paragraph("Anomaly Chart", styles["Heading3"]))
    if "anomaly" in image_paths:
        insights_chart_image = Image(
            image_paths["anomaly"], width=6 * inch, height=4 * inch
        )
        story.append(insights_chart_image)
    else:
        story.append(Paragraph("No anomaly chart available.", styles["Normal"]))
    story.append(Spacer(1, 12))

    table_data = []
    table_data.extend(anomaly_analysis)

    table = Table(table_data, colWidths=[4.0 * inch] + [4 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 12))
    # for key, value in anomaly_analysis.items():
    #     story.append(Paragraph(f"<b>{key}:</b> {value}", styles["Normal"]))
    # story.append(Spacer(1, 12))

    # Cheating Probability by Student Table
    story.append(Paragraph("4. Cheating Probability by Student", styles["Heading2"]))
    table_data = [
        [
            "ID",
            "Rate Cheat/min",
            "Contamination (%)",
            "No Suspicious",
            "No Anomalies",
            "Comment",
        ]
    ]
    table_data.extend(student_analysis)

    table = Table(table_data, colWidths=[0.3 * inch] + [1.4 * inch] * 4 + [2.1 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 12))

    # Student Analysis Section
    story.append(Paragraph("5. Student Analysis", styles["Heading2"]))
    for id, (anomaly, activity) in enumerate(
        zip(image_paths.get("anomalies", []), image_paths.get("activities", []))
    ):
        id += 1
        story.append(Paragraph(f"Student {id}", styles["Heading3"]))

        # get student image
        student_ = suspiciousReportRespository.find_one(
            {"exam_id": exam_id, "student_id": id}
        )

        if student_:
            student_image = student_["image_paths"]
            if student_image:
                studen_image = Image(student_image[0], width=4 * inch, height=3 * inch)
                story.append(studen_image)

        if anomaly:
            anomalies_chart = Image(anomaly, width=4 * inch, height=3 * inch)
            story.append(anomalies_chart)
        else:
            story.append(
                Paragraph(
                    "No anomaly chart available for this student.", styles["Normal"]
                )
            )
        story.append(Spacer(1, 12))

        if activity:
            activity_chart = Image(activity, width=4 * inch, height=3 * inch)
            story.append(activity_chart)
        else:
            story.append(
                Paragraph(
                    "No activity chart available for this student.", styles["Normal"]
                )
            )
        story.append(Spacer(1, 12))

    # Model Performance Section
    story.append(Paragraph("5. Model Performance", styles["Heading2"]))
    # for key, value in model_performance.items():
    #     story.append(Paragraph(f"<b>{key}:</b> {value}", styles["Normal"]))
    # story.append(Spacer(1, 12))

    table_data = []
    table_data.extend(model_performance)

    table = Table(table_data, colWidths=[4.0 * inch] + [4 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Model Performance Chart", styles["Heading3"]))
    if "performance" in image_paths:
        insights_chart_image = Image(
            image_paths["performance"], width=4 * inch, height=3 * inch
        )
        story.append(insights_chart_image)
    else:
        story.append(
            Paragraph("No model performance chart available.", styles["Normal"])
        )
    story.append(Spacer(1, 12))

    # Recommendations Section
    story.append(Paragraph("6. Recommendations and Next Steps", styles["Heading2"]))
    for key, value in recommendations.items():
        story.append(Paragraph(f"<b>{key}:</b> {value}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Build the PDF document
    pdf.build(story)


# Generate the PDF report with images
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph


def generate_report(exam_id):
    styles = getSampleStyleSheet()  # Get default styles
    exam_datas = studentDataRepo.find_one({"exam_id": exam_id})
    exam = examRespository.find_one({"_id": ObjectId(exam_id)})
    course = courseRespository.find_one({"_id": ObjectId(exam["course_id"])})
    session = sessionRepository.find_one({"_id": ObjectId(exam["session"])})
    department = departmentRepository.find_one({"_id": ObjectId(course["department"])})
    faculty = facultyRepository.find_one({"_id": ObjectId(course["faculty"])})
    room = roomRespository.find_one({"_id": ObjectId(exam["room_id"])})
    detection = rf"{exam_datas['path']}"

    # Wrapping exam_list values in Paragraph objects
    exam_list = [
        [
            Paragraph("Date of Examination", styles["Normal"]),
            Paragraph(exam["date"], styles["Normal"]),
        ],
        [
            Paragraph("Exam Name", styles["Normal"]),
            Paragraph(exam["name"], styles["Normal"]),
        ],
        [
            Paragraph("Academic Session", styles["Normal"]),
            Paragraph(session["name"], styles["Normal"]),
        ],
        [
            Paragraph("Course Name", styles["Normal"]),
            Paragraph(course["name"], styles["Normal"]),
        ],
        [
            Paragraph("Course Code", styles["Normal"]),
            Paragraph(course["code"], styles["Normal"]),
        ],
        [
            Paragraph("Department", styles["Normal"]),
            Paragraph(department["name"], styles["Normal"]),
        ],
        [
            Paragraph("Faculty", styles["Normal"]),
            Paragraph(faculty["name"], styles["Normal"]),
        ],
        [
            Paragraph("Exam Building", styles["Normal"]),
            Paragraph(room["name"], styles["Normal"]),
        ],
        [
            Paragraph("Building Capacity", styles["Normal"]),
            Paragraph(str(room["capacity"]), styles["Normal"]),
        ],
        [
            Paragraph("Number of Students", styles["Normal"]),
            Paragraph(str(exam_datas["no_of_students"]), styles["Normal"]),
        ],
        [
            Paragraph("Duration of Exam", styles["Normal"]),
            Paragraph(f"{exam['start_time']} - {exam['end_time']}", styles["Normal"]),
        ],
        [
            Paragraph("Objective", styles["Normal"]),
            Paragraph(
                "Detect potential cheating behaviors using AI invigilation",
                styles["Normal"],
            ),
        ],
    ]

    report = reportRespository.find_one({"exam_id": exam_id})
    exam_duration = (report["end_time"] - report["start_time"]).total_seconds() / 60
    if exam_duration < 10:
        exam_duration = 10

    metrics = report["exam_metrics"]
    sample_images = metrics["sample_images"]

    # Wrapping key_metrics values in Paragraph objects
    key_metrics = [
        ["Total Suspicious Activities Detected", str(report["n_suspicious"])],
        ["False Positives", str(metrics["false_positive"])],
        ["True Positives", str(metrics["true_positive"])],
        ["False Negatives", str(metrics["false_negative"])],
    ]

    contamination_level = []
    avg_cheat = []
    n_anomalies = 0

    for ana in report["anomaly_analysis"]:
        contamination = float(ana["contamination_level"].strip("%"))
        contamination_level.append(contamination)
        avg_cheat.append(ana["average_cheating"])
        n_anomalies += ana["n_anomalies"]

    avg_contamination = sum(contamination_level) / len(contamination_level)
    avg_cheat = sum(avg_cheat) / len(avg_cheat)

    # Wrapping anomaly_analysis values in Paragraph objects
    anomaly_analysis = [
        ["No of anomalies", str(n_anomalies)],
        ["Average Contamination", f"{round(avg_contamination, 2)}%"],
        ["Average Cheating", f"{round(avg_cheat, 2)}%"],
    ]

    # Wrapping student_analysis values in Paragraph objects
    student_analysis = [
        [
            str(ana["student_id"]),
            str(_cal_rate(sum(ana["all_cheating_score"]), exam_duration)),
            str(ana["contamination_level"].strip("%")),
            str(len(ana["all_cheating_score"])),
            str(ana["n_anomalies"]),
            str(ana["comment"]),
        ]
        for ana in report["anomaly_analysis"]
    ]

    # Wrapping model_performance values in Paragraph objects
    model_performance = [
        ["SVM Detection Precision", f"{round((metrics['precision']*100), 2)}%"],
        ["SVM Detection Recall", f"{round((metrics['recall']*100), 2)}%"],
        ["SVM Detection Accuracy", f"{round((metrics['accuracy']*100), 2)}%"],
        ["SVM Detection F1-Score", f"{round((metrics['f1_score']*100), 2)}%"],
    ]

    recommendations = {
        "Improvement Areas": "Reduce false positives by fine-tuning anomaly detection models.",
        "System Optimization": "Incorporate more training data for specific behaviors.",
    }

    # Paths to images you want to attach
    paths = plot_required_graphs(exam_id)

    file_name = os.path.join(REPORT_FOLDER, exam_id)
    filename = f"{file_name}.pdf"

    # Generate the PDF report
    generate_pdf_report(
        filename,
        exam_id,
        detection,
        exam_list,
        key_metrics,
        student_analysis,
        model_performance,
        recommendations,
        paths,
        anomaly_analysis,
        sample_images,
    )

    print("PDF report with images generated successfully!")
    return filename


def _cal_rate(total_cheating_score, exam_duration):

    # Calculate the average cheating score over the exam duration
    cheating_rate = total_cheating_score / exam_duration

    # normalized_score = min(max(0, cheating_rate), 100)

    return round(cheating_rate, 2)

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image

def generate_pdf_report(file_name, exam_data, key_metrics, detection_insights, student_analysis, model_performance, recommendations, image_paths):
    # Create PDF Document
    pdf = SimpleDocTemplate(file_name, pagesize=A4)
    
    # Sample styles and story (list to hold content)
    styles = getSampleStyleSheet()
    story = []

    # Title of the document
    story.append(Paragraph("Automatic Invigilation System - Analytical Report", styles['Title']))
    story.append(Spacer(1, 12))

    # 1. Overview Section
    story.append(Paragraph("1. Overview", styles['Heading2']))
    for key, value in exam_data.items():
        story.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Attach Image (Example: Exam Setup Image)
    if 'exam_setup' in image_paths:
        story.append(Paragraph("Exam Setup Image", styles['Heading3']))
        exam_setup_image = Image(image_paths['exam_setup'], width=4*inch, height=3*inch)
        story.append(exam_setup_image)
        story.append(Spacer(1, 12))

    # 2. Key Metrics Section
    story.append(Paragraph("2. Key Metrics", styles['Heading2']))
    for key, value in key_metrics.items():
        story.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
    story.append(Spacer(1, 12))

    # 3. Detection Insights Section
    story.append(Paragraph("3. Detection Insights", styles['Heading2']))
    for key, value in detection_insights.items():
        story.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Attach Image (Example: Detection Insights Chart)
    if 'detection_insights_chart' in image_paths:
        story.append(Paragraph("Detection Insights Chart", styles['Heading3']))
        insights_chart_image = Image(image_paths['detection_insights_chart'], width=4*inch, height=3*inch)
        story.append(insights_chart_image)
        story.append(Spacer(1, 12))

    # 4. Student Analysis (Table)
    story.append(Paragraph("4. Cheating Probability by Student", styles['Heading2']))
    table_data = [["Student ID", "Anomaly Detected", "Cheating Probability (%)", "Action Taken", "Comment"]]
    table_data.extend(student_analysis)
    
    table = Table(table_data, colWidths=[1.2*inch, 1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    # 5. Model Performance Section
    story.append(Paragraph("5. Model Performance", styles['Heading2']))
    for key, value in model_performance.items():
        story.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Attach Image (Example: Model Performance Graph)
    if 'model_performance_graph' in image_paths:
        story.append(Paragraph("Model Performance Graph", styles['Heading3']))
        performance_graph_image = Image(image_paths['model_performance_graph'], width=4*inch, height=3*inch)
        story.append(performance_graph_image)
        story.append(Spacer(1, 12))

    # 6. Recommendations Section
    story.append(Paragraph("6. Recommendations and Next Steps", styles['Heading2']))
    for key, value in recommendations.items():
        story.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Build the PDF document
    pdf.build(story)


# Example data to pass into the PDF generator
exam_data = {
    "Date of Examination": "2024-09-19",
    "Number of Students": 30,
    "Duration of Exam": "10:00 AM - 12:00 PM",
    "Objective": "Detect potential cheating behaviors using AI invigilation"
}

key_metrics = {
    "Total Suspicious Activities Detected": 15,
    "Number of Alerts Raised": 10,
    "False Positives": 2,
    "True Positives": 8,
    "False Negatives": 1
}

detection_insights = {
    "Movement-based Anomalies": "60%",
    "Object Detection Anomalies": "30%",
    "Optical Flow Anomalies": "10%"
}

student_analysis = [
    ["101", "Head Turning", "85%", "Alerted", "Suspicious"],
    ["102", "No Anomaly Detected", "5%", "No Action", "Normal"],
    ["103", "Object Detection", "75%", "Alerted", "Unauthorized Device"],
]

model_performance = {
    "SVM Detection Accuracy": "92%",
    "YOLO Object Detection Accuracy": "89%",
    "LSTM/Anomaly Detection Accuracy": "91%"
}

recommendations = {
    "Improvement Areas": "Reduce false positives by fine-tuning anomaly detection models.",
    "System Optimization": "Incorporate more training data for specific behaviors."
}

# Paths to images you want to attach
image_paths = {
    'exam_setup': 'path/to/exam_setup_image.jpg',
    'detection_insights_chart': 'path/to/detection_insights_chart.png',
    'model_performance_graph': 'path/to/model_performance_graph.png'
}

# Generate the PDF report with images
generate_pdf_report("invigilation_report_with_images.pdf", exam_data, key_metrics, detection_insights, student_analysis, model_performance, recommendations, image_paths)

print("PDF report with images generated successfully!")

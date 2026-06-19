import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime

class ReportService:
    @staticmethod
    def generate_pdf(analysis, output_dir: str) -> str:
        pdf_filename = f"report_analysis_{analysis.id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
        pdf_path = os.path.join(output_dir, pdf_filename)
        
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = styles['Heading1']
        title_style.alignment = 1 # Center
        story.append(Paragraph("AI Hematologist - Analysis Report", title_style))
        story.append(Spacer(1, 12))

        # Basic Info
        normal_style = styles['Normal']
        story.append(Paragraph(f"<b>Analysis ID:</b> {analysis.id}", normal_style))
        story.append(Paragraph(f"<b>Date:</b> {analysis.created_at.strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
        story.append(Spacer(1, 12))

        # Images
        story.append(Paragraph("<b>Uploaded Image & Detections:</b>", styles['Heading2']))
        if analysis.annotated_image_path and os.path.exists(analysis.annotated_image_path):
            img = Image(analysis.annotated_image_path, width=400, height=300)
            story.append(img)
        else:
            story.append(Paragraph("Image not available.", normal_style))
        story.append(Spacer(1, 12))

        # Cell Counts
        story.append(Paragraph("<b>Cell Counts:</b>", styles['Heading2']))
        story.append(Paragraph(f"Total Cells: {analysis.total_cells}", normal_style))
        story.append(Paragraph(f"Red Blood Cells: {analysis.rbc_count}", normal_style))
        story.append(Paragraph(f"White Blood Cells: {analysis.wbc_count}", normal_style))
        story.append(Paragraph(f"Platelets: {analysis.platelet_count}", normal_style))
        story.append(Spacer(1, 12))

        # AI Analysis
        story.append(Paragraph("<b>AI Medical Explanation:</b>", styles['Heading2']))
        
        story.append(Paragraph("<b>Professional Summary:</b>", styles['Heading3']))
        story.append(Paragraph(analysis.gemini_professional or "N/A", normal_style))
        story.append(Spacer(1, 6))
        
        story.append(Paragraph("<b>Patient-Friendly Summary:</b>", styles['Heading3']))
        story.append(Paragraph(analysis.gemini_patient or "N/A", normal_style))
        story.append(Spacer(1, 6))

        story.append(Paragraph("<b>Technical Summary:</b>", styles['Heading3']))
        story.append(Paragraph(analysis.gemini_technical or "N/A", normal_style))
        story.append(Spacer(1, 12))

        # Disclaimer
        disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'], textColor='red')
        story.append(Paragraph("<i>DISCLAIMER: This analysis is for educational and research purposes and is not a medical diagnosis. Please consult a qualified healthcare professional.</i>", disclaimer_style))

        # Build PDF
        doc.build(story)
        
        return pdf_path

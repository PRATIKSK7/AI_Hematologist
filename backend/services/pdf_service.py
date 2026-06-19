import os
import time
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from backend.config import logger

class PDFService:
    def __init__(self):
        self.output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "uploads", "reports")
        os.makedirs(self.output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor("#0f172a"),
            alignment=1, # Center
            spaceAfter=30
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor("#0284c7"),
            spaceBefore=20,
            spaceAfter=10
        )
        
        self.normal_style = self.styles["Normal"]
        self.normal_style.fontSize = 10
        self.normal_style.leading = 14

    def generate_report(self, counts: dict, stats: dict, gemini_text: str, image_path: str = None, disease_risks: dict = None, overall_health_score: int = None, morphology_data: dict = None) -> str:
        timestamp = int(time.time())
        filename = f"Hematology_Report_{timestamp}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=50, leftMargin=50,
            topMargin=50, bottomMargin=50
        )
        
        story = []
        
        # Title
        story.append(Paragraph("AI Hematologist Medical Intelligence Report", self.title_style))
        
        # Cell Counts Table
        story.append(Paragraph("Cell Detection Metrics", self.heading_style))
        data = [
            ["Cell Type", "Count", "Percentage"],
            ["Red Blood Cells (RBC)", str(counts.get("rbc_count", 0)), f"{counts.get('rbc_percentage', 0)}%"],
            ["White Blood Cells (WBC)", str(counts.get("wbc_count", 0)), f"{counts.get('wbc_percentage', 0)}%"],
            ["Platelets (PLT)", str(counts.get("platelet_count", 0)), f"{counts.get('platelet_percentage', 0)}%"],
            ["Total Detectable Cells", str(counts.get("total_cells", 0)), "100%"]
        ]
        
        table = Table(data, colWidths=[200, 100, 100])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0284c7")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1"))
        ]))
        story.append(table)
        story.append(Spacer(1, 20))
        
        # RBC Morphology Breakdown
        if morphology_data and morphology_data.get("total_analyzed", 0) > 0:
            story.append(Paragraph("Red Blood Cell Morphology", self.heading_style))
            morph_data = [["Morphology Type", "Count", "Percentage"]]
            
            for cls_name, count in morphology_data.get("counts", {}).items():
                if count > 0:
                    pct = morphology_data.get("percentages", {}).get(cls_name, 0.0)
                    morph_data.append([cls_name, str(count), f"{pct}%"])
                    
            if len(morph_data) > 1:
                morph_table = Table(morph_data, colWidths=[200, 100, 100])
                morph_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#be185d")), # Pink/Red for RBC
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#fdf2f8")),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#fbcfe8"))
                ]))
                story.append(morph_table)
                story.append(Spacer(1, 20))
        
        # Disease Risk Scores
        if disease_risks and overall_health_score is not None:
            story.append(Paragraph("AI Disease Risk Assessment", self.heading_style))
            
            # Health Score Color
            hs_color = "#10b981" # Green
            if overall_health_score < 40: hs_color = "#ef4444"
            elif overall_health_score < 70: hs_color = "#f59e0b"
            
            risk_data = [
                ["Metric", "Risk Score (0-100)"],
                ["Overall Health Score", str(overall_health_score)],
                ["Anemia Risk", str(disease_risks.get('anemia', 0))],
                ["Leukemia Risk", str(disease_risks.get('leukemia', 0))],
                ["Thrombocytopenia Risk", str(disease_risks.get('thrombocytopenia', 0))],
                ["Infection Risk", str(disease_risks.get('infection', 0))]
            ]
            
            risk_table = Table(risk_data, colWidths=[200, 200])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1"))
            ]))
            story.append(risk_table)
            story.append(Spacer(1, 20))
        
        # Image
        if image_path and os.path.exists(image_path):
            story.append(Paragraph("Annotated Blood Smear", self.heading_style))
            try:
                img = Image(image_path, width=400, height=300)
                story.append(img)
            except Exception as e:
                logger.error(f"Failed to embed image in PDF: {e}")
                story.append(Paragraph("[Image preview unavailable]", self.normal_style))
            story.append(Spacer(1, 20))
            
        # Gemini AI Explanation
        story.append(Paragraph("AI Copilot Diagnostics", self.heading_style))
        
        # Process Gemini Markdown
        for line in gemini_text.split('\n'):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 8))
            elif line.startswith('##'):
                story.append(Paragraph(line.replace('##', '').strip(), self.heading_style))
            elif line.startswith('#'):
                story.append(Paragraph(line.replace('#', '').strip(), self.heading_style))
            elif line.startswith('*') or line.startswith('-'):
                story.append(Paragraph("• " + line[1:].strip(), self.normal_style))
            else:
                story.append(Paragraph(line, self.normal_style))

        # Disclaimer
        story.append(Spacer(1, 30))
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=self.styles['Normal'],
            textColor=colors.gray,
            fontSize=8,
            alignment=1
        )
        story.append(Paragraph("DISCLAIMER: This analysis is AI-assisted and is not a medical diagnosis. Please consult a licensed hematologist.", disclaimer_style))
        
        try:
            doc.build(story)
            logger.info(f"PDF generated successfully at {filepath}")
            return f"/uploads/reports/{filename}"
        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")
            raise RuntimeError(f"PDF generation failed: {e}")

import os
import shutil
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from backend.database.database import get_db
from backend.models import models, schemas
from backend.services.yolo_service import BloodCellDetector, CellCounter
from backend.services.gemini_service import GeminiService
from backend.services.report_service import ReportService

router = APIRouter()

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

detector = BloodCellDetector.get_instance()
gemini_service = GeminiService()
report_service = ReportService()

@router.post("/upload", response_model=schemas.AnalysisResponse)
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create empty analysis record
    db_analysis = models.Analysis(original_image_path=file_path)
    db.add(db_analysis)
    db.commit()
    db.refresh(db_analysis)

    return db_analysis

@router.post("/analyze/{analysis_id}", response_model=schemas.AnalysisResponse)
async def analyze_image(analysis_id: int, db: Session = Depends(get_db)):
    db_analysis = db.query(models.Analysis).filter(models.Analysis.id == analysis_id).first()
    if not db_analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    if db_analysis.annotated_image_path:
        return db_analysis # Already analyzed

    # 1. YOLO Inference
    try:
        detections, annotated_path = detector.detect(db_analysis.original_image_path, UPLOAD_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    db_analysis.annotated_image_path = annotated_path

    # Save detections to DB
    for det in detections:
        db_det = models.Detection(
            analysis_id=db_analysis.id,
            **det
        )
        db.add(db_det)
    
    # 2. Count Cells
    counts = CellCounter.count_cells(detections)
    db_analysis.rbc_count = counts["rbc_count"]
    db_analysis.wbc_count = counts["wbc_count"]
    db_analysis.platelet_count = counts["platelet_count"]
    db_analysis.total_cells = counts["total_cells"]

    # 3. Gemini Explanation
    explanations = gemini_service.generate_explanation(counts)
    db_analysis.gemini_professional = explanations["professional"]
    db_analysis.gemini_patient = explanations["patient"]
    db_analysis.gemini_technical = explanations["technical"]

    db.commit()
    db.refresh(db_analysis)

    # 4. Generate Report
    pdf_path = report_service.generate_pdf(db_analysis, REPORT_DIR)
    db_report = models.Report(analysis_id=db_analysis.id, pdf_path=pdf_path)
    db.add(db_report)
    
    db.commit()
    db.refresh(db_analysis)

    return db_analysis

@router.get("/report/{analysis_id}")
async def get_report(analysis_id: int, db: Session = Depends(get_db)):
    db_report = db.query(models.Report).filter(models.Report.analysis_id == analysis_id).first()
    if not db_report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    from fastapi.responses import FileResponse
    return FileResponse(path=db_report.pdf_path, filename=os.path.basename(db_report.pdf_path), media_type='application/pdf')

@router.get("/history", response_model=List[schemas.AnalysisResponse])
async def get_history(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    analyses = db.query(models.Analysis).order_by(models.Analysis.created_at.desc()).offset(skip).limit(limit).all()
    return analyses

@router.get("/dashboard", response_model=schemas.DashboardStats)
async def get_dashboard_stats(db: Session = Depends(get_db)):
    total_analyses = db.query(models.Analysis).count()
    total_reports = db.query(models.Report).count()

    if total_analyses > 0:
        from sqlalchemy import func
        avg_rbc = db.query(func.avg(models.Analysis.rbc_count)).scalar() or 0
        avg_wbc = db.query(func.avg(models.Analysis.wbc_count)).scalar() or 0
        avg_plat = db.query(func.avg(models.Analysis.platelet_count)).scalar() or 0
    else:
        avg_rbc, avg_wbc, avg_plat = 0, 0, 0

    recent = db.query(models.Analysis).order_by(models.Analysis.created_at.desc()).limit(5).all()

    return {
        "total_analyses": total_analyses,
        "total_reports": total_reports,
        "average_rbc": round(avg_rbc, 2),
        "average_wbc": round(avg_wbc, 2),
        "average_platelets": round(avg_plat, 2),
        "recent_analyses": recent
    }

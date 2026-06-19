from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class DetectionBase(BaseModel):
    class_name: str
    confidence: float
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float

class DetectionCreate(DetectionBase):
    pass

class DetectionResponse(DetectionBase):
    id: int
    analysis_id: int

    class Config:
        from_attributes = True

class ReportBase(BaseModel):
    pdf_path: str

class ReportResponse(ReportBase):
    id: int
    analysis_id: int
    created_at: datetime

    class Config:
        from_attributes = True

class AnalysisBase(BaseModel):
    original_image_path: str
    annotated_image_path: Optional[str] = None
    rbc_count: int = 0
    wbc_count: int = 0
    platelet_count: int = 0
    total_cells: int = 0
    gemini_professional: Optional[str] = None
    gemini_patient: Optional[str] = None
    gemini_technical: Optional[str] = None

class AnalysisCreate(AnalysisBase):
    pass

class AnalysisResponse(AnalysisBase):
    id: int
    created_at: datetime
    detections: List[DetectionResponse] = []
    report: Optional[ReportResponse] = None

    class Config:
        from_attributes = True

class DashboardStats(BaseModel):
    total_analyses: int
    total_reports: int
    average_rbc: float
    average_wbc: float
    average_platelets: float
    recent_analyses: List[AnalysisResponse]

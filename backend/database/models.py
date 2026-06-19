from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON
from datetime import datetime
from backend.database.database import Base

class AnalysisReport(Base):
    __tablename__ = "analysis_reports"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # We can store the JSON outputs as JSON or Text
    counts = Column(JSON, default={})
    statistics = Column(JSON, default={})
    morphology = Column(JSON, default={})
    disease_risks = Column(JSON, default={})
    reasoning = Column(JSON, default={})
    overall_health_score = Column(Float, default=0.0)
    
    gemini_report = Column(Text, nullable=True)
    annotated_image_url = Column(String, nullable=True)

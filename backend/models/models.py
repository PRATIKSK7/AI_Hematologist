from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.database.database import Base

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    original_image_path = Column(String, nullable=False)
    annotated_image_path = Column(String, nullable=True)
    
    rbc_count = Column(Integer, default=0)
    wbc_count = Column(Integer, default=0)
    platelet_count = Column(Integer, default=0)
    total_cells = Column(Integer, default=0)
    
    gemini_professional = Column(Text, nullable=True)
    gemini_patient = Column(Text, nullable=True)
    gemini_technical = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    detections = relationship("Detection", back_populates="analysis", cascade="all, delete-orphan")
    report = relationship("Report", back_populates="analysis", uselist=False, cascade="all, delete-orphan")

class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"))
    class_name = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    bbox_x = Column(Float, nullable=False)
    bbox_y = Column(Float, nullable=False)
    bbox_w = Column(Float, nullable=False)
    bbox_h = Column(Float, nullable=False)

    analysis = relationship("Analysis", back_populates="detections")

class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"))
    pdf_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    analysis = relationship("Analysis", back_populates="report")

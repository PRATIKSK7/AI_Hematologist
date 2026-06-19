from pydantic import BaseModel
from typing import List, Dict, Any

class CellCounts(BaseModel):
    rbc_count: int
    wbc_count: int
    platelet_count: int
    total_cells: int
    rbc_percentage: float
    wbc_percentage: float
    platelet_percentage: float
    confidence_score: float = 0.0

class DiseaseRisks(BaseModel):
    iron_deficiency_anemia: int
    hereditary_spherocytosis: int
    hemolytic_anemia: int
    leukemia: int
    infection: int

class CellPrediction(BaseModel):
    box: List[float]
    class_name: str
    confidence: float

class MorphologyData(BaseModel):
    total_analyzed: int
    counts: Dict[str, int]
    percentages: Dict[str, float]
    cell_predictions: List[CellPrediction] = []

class AnalysisResponse(BaseModel):
    counts: CellCounts
    statistics: Dict[str, Any]
    morphology: MorphologyData = None
    gemini_report: str
    annotated_image_url: str
    disease_risks: DiseaseRisks = None
    overall_health_score: int = None
    reasoning: Dict[str, str] = None

class HealthResponse(BaseModel):
    backend: bool
    yolo: bool
    gemini: bool
    database: bool = True

class ModelInfoResponse(BaseModel):
    model_path: str
    class_names: List[str]
    model_version: str

class PdfRequest(BaseModel):
    counts: Dict[str, Any]
    statistics: Dict[str, Any]
    morphology: Dict[str, Any] = None
    gemini_report: str
    annotated_image_url: str
    disease_risks: Dict[str, int] = None
    overall_health_score: int = None

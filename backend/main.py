from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from backend.api import analyze_router
from backend.config import logger
from backend.services.detector_service import DetectorService
from backend.services.gemini_service import GeminiService

app = FastAPI(
    title="AI Hematologist Inference Pipeline",
    description="Production API for YOLOv8 blood cell detection and Gemini explanations",
    version="1.0.0"
)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

app.include_router(analyze_router.router, tags=["Pipeline"])

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Backend Services...")
    
    detector = DetectorService.get_instance()
    gemini = GeminiService()
    
    # Check YOLO
    yolo_loaded = detector.is_loaded()
    if yolo_loaded:
        logger.info("✅ YOLOv8 Engine: Online")
    else:
        logger.error("❌ YOLOv8 Engine: Offline")
        
    # Check Gemini
    gemini_loaded = gemini.is_configured()
    if gemini_loaded:
        logger.info("✅ Gemini Copilot: Online")
    else:
        logger.error("❌ Gemini Copilot: Offline")
        
    # Check TensorFlow Morphology Model
    from backend.services.morphology_service import morphology_service
    tf_loaded = morphology_service.load_model()
    if tf_loaded:
        logger.info("✅ TensorFlow Morphology Model: Online")
    else:
        logger.error("❌ TensorFlow Morphology Model: Offline")
        
    # Check Database
    try:
        from backend.database.database import Base, engine
        import backend.database.models # Import models to ensure they are registered
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database Engine: Online")
        db_loaded = True
    except Exception as e:
        db_loaded = False
        logger.error(f"❌ Database Engine: Offline - {e}")
        
    # Store statuses in app state for health endpoint
    app.state.status = {
        "fastapi": True,
        "database": db_loaded,
        "gemini": gemini_loaded,
        "model": yolo_loaded and tf_loaded,
        "status": "healthy" if (db_loaded and gemini_loaded and yolo_loaded and tf_loaded) else "degraded"
    }

@app.get("/health")
def health_check():
    return app.state.status

@app.get("/")
def read_root():
    return {"message": "Welcome to AI Hematologist Inference Pipeline"}

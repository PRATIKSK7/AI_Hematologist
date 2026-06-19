import os
import shutil
import time
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from backend.schemas import AnalysisResponse, HealthResponse, ModelInfoResponse, CellCounts, PdfRequest
from backend.services.detector_service import DetectorService
from backend.services.count_cells import calculate_cell_counts, calculate_statistics, remove_duplicate_detections
from backend.config import settings
from backend.services.gemini_service import GeminiService
from backend.services.pdf_service import PDFService
from backend.services.disease_predictor import DiseasePredictor
from backend.services.morphology_service import morphology_service
import cv2
from backend.config import logger

router = APIRouter()

detector = DetectorService.get_instance()
gemini = GeminiService()
pdf_service = PDFService()

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type uploaded: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Successfully saved uploaded file to {file_path}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Could not save file")

    # 1. Inference
    t0 = time.time()
    try:
        detections, annotated_image_path = detector.detect(file_path, UPLOAD_DIR)
        inference_time = time.time() - t0
        logger.info(f"YOLO Inference completed in {inference_time:.3f}s. Found {len(detections)} detections.")
    except RuntimeError as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=503, detail=f"Model inference failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during inference: {str(e)}")

    # 2. Count Cells & Post-Process
    try:
        # Advanced Duplicate Detection Removal
        filtered_detections = remove_duplicate_detections(detections)
        logger.info(f"Duplicate removal eliminated {len(detections) - len(filtered_detections)} redundant bounding boxes.")
        
        counts_dict = calculate_cell_counts(filtered_detections)
        stats_dict = calculate_statistics(filtered_detections)
        if counts_dict["total_cells"] == 0:
            logger.info("No cells detected in the image.")
    except Exception as e:
        logger.error(f"Error calculating cell counts: {e}")
        raise HTTPException(status_code=500, detail=f"Error calculating cell statistics: {str(e)}")

    # 3. RBC Morphology Analysis
    try:
        # Extract RBC boxes: assuming class_id 0 or class_name 'RBC'
        rbc_boxes = [d["bbox"] for d in filtered_detections if d.get("class_name", "").upper() == "RBC" or d.get("class_id") == 0]
        
        logger.info(f"[DEBUG] Found {len(rbc_boxes)} RBC boxes from YOLO.")
        
        if rbc_boxes:
            # Read original image for cropping
            image_np = cv2.imread(file_path)
            if image_np is not None:
                logger.info("[DEBUG] Loaded image for cropping. Converting BGR to RGB for EfficientNet preprocess_input compatibility.")
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                morphology_data = morphology_service.analyze_rbcs(image_np, rbc_boxes)
                logger.info(f"Morphology analysis complete for {len(rbc_boxes)} RBCs. Extracted {morphology_data.get('total_analyzed', 0)} crops.")
                logger.info(f"[DEBUG] Morphology Percentages: {morphology_data.get('percentages')}")
            else:
                logger.error("[DEBUG] image_np is None when trying to read for cropping.")
                morphology_data = morphology_service._empty_stats()
        else:
            logger.info("[DEBUG] No RBC boxes found, skipping morphology.")
            morphology_data = morphology_service._empty_stats()
    except Exception as e:
        logger.error(f"Morphology extraction error: {e}")
        morphology_data = morphology_service._empty_stats()

    # 4. Disease Risk Prediction
    try:
        risk_data = DiseasePredictor.calculate_risk(
            rbc_count=counts_dict["rbc_count"],
            wbc_count=counts_dict["wbc_count"],
            platelet_count=counts_dict["platelet_count"],
            morphology_percentages=morphology_data.get("percentages", {})
        )
        logger.info(f"Disease risk prediction completed. Health Score: {risk_data['overall_health_score']}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Error calculating disease risks: {e}")
        risk_data = {
            "disease_risks": {
                "iron_deficiency_anemia": 0,
                "hereditary_spherocytosis": 0,
                "hemolytic_anemia": 0,
                "leukemia": 0,
                "infection": 0
            },
            "overall_health_score": 50,
            "reasoning": {}
        }
        
    # 4.5 Overlay Morphology Predictions on Annotated Image
    try:
        if morphology_data and morphology_data.get("cell_predictions") and annotated_image_path:
            annotated_img = cv2.imread(annotated_image_path)
            if annotated_img is not None:
                for pred in morphology_data["cell_predictions"]:
                    box = pred["box"]
                    class_name = pred["class_name"]
                    conf = pred["confidence"]
                    x1, y1 = int(box[0]), int(box[1])
                    
                    text = f"{class_name} {conf:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    
                    # Background for text
                    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    cv2.rectangle(annotated_img, (x1, y1 - th - 4), (x1 + tw, y1), (0, 0, 0), -1)
                    cv2.putText(annotated_img, text, (x1, y1 - 2), font, font_scale, (0, 255, 255), thickness)
                    
                cv2.imwrite(annotated_image_path, annotated_img)
                logger.info("[DEBUG] Successfully overlaid morphology predictions on annotated image.")
    except Exception as e:
        logger.error(f"Error overlaying morphology text on image: {e}")

    # 5. Gemini Report
    t1 = time.time()
    try:
        gemini_report = gemini.generate_report(counts_dict, stats_dict, risk_data, morphology_data)
        gemini_time = time.time() - t1
        logger.info(f"Gemini generation completed in {gemini_time:.3f}s")
    except Exception as e:
        logger.error(f"Gemini generation completely failed: {e}")
        gemini_report = f"Error generating explanation: {str(e)}\n\nThis analysis is AI-assisted and is not a medical diagnosis."

    # 6. Save to Database
    try:
        from backend.database.database import SessionLocal
        from backend.database.models import AnalysisReport
        db = SessionLocal()
        new_report = AnalysisReport(
            counts=counts_dict,
            statistics=stats_dict,
            morphology=morphology_data,
            disease_risks=risk_data.get("disease_risks", {}),
            reasoning=risk_data.get("reasoning", {}),
            overall_health_score=risk_data.get("overall_health_score", 0),
            gemini_report=gemini_report,
            annotated_image_url=f"/uploads/{os.path.basename(annotated_image_path)}"
        )
        db.add(new_report)
        db.commit()
        db.close()
        logger.info("Successfully saved report to SQLite database.")
    except Exception as e:
        logger.error(f"Failed to save report to database: {e}")

    try:
        return AnalysisResponse(
            counts=CellCounts(**counts_dict),
            statistics=stats_dict,
            morphology=morphology_data,
            gemini_report=gemini_report,
            annotated_image_url=f"/uploads/{os.path.basename(annotated_image_path)}",
            disease_risks=risk_data["disease_risks"],
            overall_health_score=risk_data["overall_health_score"],
            reasoning=risk_data["reasoning"]
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Failed to generate AnalysisResponse: {e}")
        raise HTTPException(status_code=500, detail=f"Response serialization failed: {str(e)}")

@router.post("/generate-pdf")
async def generate_pdf(request: PdfRequest):
    try:
        # Convert annotated_image_url to local path
        image_filename = os.path.basename(request.annotated_image_url)
        local_image_path = os.path.join(UPLOAD_DIR, image_filename)
        
        pdf_url = pdf_service.generate_report(
            counts=request.counts,
            stats=request.statistics,
            gemini_text=request.gemini_report,
            image_path=local_image_path if os.path.exists(local_image_path) else None,
            disease_risks=request.disease_risks,
            overall_health_score=request.overall_health_score,
            morphology_data=request.morphology
        )
        return {"pdf_url": pdf_url}
    except Exception as e:
        logger.error(f"Failed to generate PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gemini-health")
async def gemini_health():
    response = {
        "api_key_loaded": False,
        "sdk_installed": True, # Hardcoded as True if this file compiles
        "model_accessible": False,
        "test_generation": False,
        "error": None
    }
    
    try:
        if not gemini.api_key:
            response["error"] = "No API key found in .env"
            return response
            
        response["api_key_loaded"] = True
        
        if not gemini.is_configured():
            response["error"] = "Client failed to initialize"
            return response
            
        # Test Generation
        test_res = gemini.test_connection()
        response["model_accessible"] = True
        response["test_generation"] = True
        
    except Exception as e:
        response["error"] = str(e)
        
    return response

@router.get("/test-gemini")
async def test_gemini():
    try:
        gemini = GeminiService()
        response_text = gemini.test_connection()
        import google.genai as genai_module
        return {
            "success": True, 
            "sdk": f"google-genai v{genai_module.__version__}", 
            "authentication": "API Key", 
            "response": response_text
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.get("/debug-inference")
async def debug_inference(image_filename: str = "t1.jpeg"):
    if not detector.is_loaded():
        return {"model_loaded": False, "error": "Model not loaded"}
    
    file_path = os.path.join(UPLOAD_DIR, image_filename)
    if not os.path.exists(file_path):
        return {"error": f"Image {image_filename} not found"}
        
    try:
        # We temporarily force a low confidence to see all boxes
        results = detector.model(file_path, conf=0.1, verbose=False)
        result = results[0]
        
        raw_detections = []
        for box in result.boxes:
            raw_detections.append({
                "class_id": int(box.cls[0].item()),
                "class_name": detector.model.names[int(box.cls[0].item())],
                "confidence": float(box.conf[0].item()),
                "bbox": box.xyxy[0].tolist()
            })
            
        counts = calculate_cell_counts(raw_detections)
        
        return {
            "model_loaded": True,
            "image_tested": image_filename,
            "detections": raw_detections,
            "counts": counts,
            "class_names": detector.get_class_names()
        }
    except Exception as e:
        return {"error": str(e)}

@router.get("/debug-detections")
async def debug_detections(image_filename: str = "t2.jpg"):
    if not detector.is_loaded():
        return {"error": "Model not loaded"}
    
    file_path = os.path.join(UPLOAD_DIR, image_filename)
    if not os.path.exists(file_path):
        return {"error": f"Image {image_filename} not found"}
        
    try:
        # Raw extreme bounds (where the 266+ duplicates were coming from)
        results_raw = detector.model.predict(source=file_path, conf=0.001, iou=0.7, verbose=False)[0]
        
        # New optimized inference with Area filtering and Strict bounds
        results_filtered = detector.model.predict(
            source=file_path, 
            imgsz=settings.yolo_image_size, 
            conf=settings.yolo_confidence, 
            iou=settings.yolo_iou, 
            max_det=1000, 
            half=True, 
            verbose=False
        )[0]
        
        filtered_boxes = []
        rejected_platelets = 0
        
        thresholds = {
            "RBC": 0.15,
            "WBC": 0.10,
            "Platelets": 0.001
        }
        
        for box in results_filtered.boxes:
            class_id = int(box.cls[0].item())
            class_name = detector.model.names[class_id]
            confidence = float(box.conf[0].item())
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            
            if class_name == "Platelets" and area > 1000:
                rejected_platelets += 1
                continue
                
            if confidence >= thresholds.get(class_name, 0.15):
                filtered_boxes.append({
                    "class_id": class_id,
                    "class_name": class_name
                })
            
        counts = calculate_cell_counts(filtered_boxes)
        
        return {
            "image_tested": image_filename,
            "raw_detection_count": len(results_raw.boxes),
            "filtered_detection_count": len(filtered_boxes),
            "false_positive_platelets_rejected_by_area": rejected_platelets,
            "rbc_count": counts["rbc_count"],
            "wbc_count": counts["wbc_count"],
            "platelet_count": counts["platelet_count"]
        }
    except Exception as e:
        return {"error": str(e)}

@router.get("/debug/status")
async def debug_status():
    try:
        return {
            "backend": True,
            "yolo_loaded": detector.is_loaded() if detector else False,
            "gemini_available": gemini.is_configured() if gemini else False,
            "model_path": detector.model_path if detector and detector.is_loaded() else None,
            "api_version": "1.0.0",
            "last_error": None
        }
    except Exception as e:
        return {
            "backend": True,
            "yolo_loaded": False,
            "gemini_available": False,
            "model_path": None,
            "api_version": "1.0.0",
            "last_error": str(e)
        }

@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    if not detector.is_loaded():
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    return ModelInfoResponse(
        model_path=detector.model_path,
        class_names=detector.get_class_names(),
        model_version="YOLOv8"
    )

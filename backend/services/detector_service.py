import os
import cv2
from ultralytics import YOLO
from backend.config import settings, logger

class DetectorService:
    _instance = None
    
    def __init__(self):
        self.model_path = settings.model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
                logger.info(f"YOLO model loaded successfully from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
        else:
            logger.error(f"YOLO model file not found at {self.model_path}")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_loaded(self) -> bool:
        return self.model is not None

    def get_class_names(self):
        return list(self.model.names.values()) if self.model else []

    def detect(self, image_path: str, output_dir: str, iou_threshold: float = 0.45):
        if not self.is_loaded():
            raise RuntimeError("YOLO model is not loaded.")
        
        try:
            # Use optimal prediction constraints requested by user + performance tuning
            results = self.model.predict(
                source=image_path, 
                imgsz=settings.yolo_image_size, 
                conf=settings.yolo_confidence, 
                iou=settings.yolo_iou, 
                max_det=1000, 
                half=True, # FP16 optimization for speed < 2s
                verbose=False
            )
            
            os.makedirs(output_dir, exist_ok=True)
            annotated_image_path = os.path.join(output_dir, "annotated_" + os.path.basename(image_path))
            
            result = results[0]
            
            # Custom Strict Thresholds
            thresholds = {
                "RBC": 0.15,
                "WBC": 0.10,
                "Platelets": 0.001
            }
            
            # Colors: BGR format for OpenCV
            colors = {
                "RBC": (0, 0, 255),        # Red
                "WBC": (128, 0, 128),      # Purple
                "Platelets": (0, 255, 0)   # Green
            }
            
            # Load original image for custom OpenCV drawing
            img = cv2.imread(image_path)
            
            detections = []
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0].item())
                
                # Bounding Box Area Calculation
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                area = (x2 - x1) * (y2 - y1)
                
                # Biological Size Filtering (Reject large platelets)
                if class_name == "Platelets" and area > 1000:
                    continue
                
                # Class-specific confidence filtering
                if confidence >= thresholds.get(class_name, 0.15):
                    detections.append({
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2]
                    })
                    
                    # Custom OpenCV Bounding Box Drawing (Thin, No Conf, Color-Coded)
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    color = colors.get(class_name, (255, 255, 255))
                    
                    # Thin bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
                    
                    # Label background
                    (w, h), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(img, (x1, y1 - 15), (x1 + w, y1), color, -1)
                    
                    # Label text
                    cv2.putText(img, class_name, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.imwrite(annotated_image_path, img)

            return detections, annotated_image_path
        except Exception as e:
            logger.error(f"Inference failed on image {image_path}: {e}", exc_info=True)
            raise RuntimeError(f"Inference failed: {e}")

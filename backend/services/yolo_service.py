import os
import cv2
from ultralytics import YOLO

class BloodCellDetector:
    _instance = None
    
    def __init__(self):
        # Path to best.pt
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "weights", "best.pt")
        # Load model only once
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            self.model = None
            print(f"Warning: YOLO model not found at {model_path}")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def detect(self, image_path: str, output_dir: str):
        if self.model is None:
            raise ValueError("YOLO model not loaded.")
        
        results = self.model(image_path)
        
        # Save annotated image
        annotated_image_path = os.path.join(output_dir, "annotated_" + os.path.basename(image_path))
        
        # We need the first result (since we passed one image)
        result = results[0]
        # Save the image with boxes
        annotated_img = result.plot()
        cv2.imwrite(annotated_image_path, annotated_img)

        # Parse detections
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0].item())
            
            # xywh format
            x, y, w, h = box.xywh[0].tolist()
            
            detections.append({
                "class_name": class_name,
                "confidence": confidence,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h
            })

        return detections, annotated_image_path

class CellCounter:
    @staticmethod
    def count_cells(detections: list) -> dict:
        rbc_count = 0
        wbc_count = 0
        platelet_count = 0
        
        for det in detections:
            # Based on user classes: 0 = RBC, 1 = WBC, 2 = Platelets. The string name depends on training.
            # We match by string if standard names were used, or we just map standard.
            cname = det["class_name"].lower()
            if "rbc" in cname or "red" in cname or cname == "0":
                rbc_count += 1
            elif "wbc" in cname or "white" in cname or cname == "1":
                wbc_count += 1
            elif "platelet" in cname or cname == "2":
                platelet_count += 1
                
        total_cells = rbc_count + wbc_count + platelet_count
        
        rbc_percentage = round((rbc_count / total_cells * 100), 1) if total_cells > 0 else 0
        wbc_percentage = round((wbc_count / total_cells * 100), 1) if total_cells > 0 else 0
        platelet_percentage = round((platelet_count / total_cells * 100), 1) if total_cells > 0 else 0

        return {
            "total_cells": total_cells,
            "rbc_count": rbc_count,
            "wbc_count": wbc_count,
            "platelet_count": platelet_count,
            "rbc_percentage": rbc_percentage,
            "wbc_percentage": wbc_percentage,
            "platelet_percentage": platelet_percentage
        }

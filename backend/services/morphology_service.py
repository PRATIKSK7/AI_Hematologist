import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# The training script uses image_dataset_from_directory which sorts alphabetically
MORPHOLOGY_CLASSES = [
    'BurrCell', 
    'Elliptocyte', 
    'Hypochromia', 
    'Macrocyte', 
    'Microcyte', 
    'Normal', 
    'Ovalocyte', 
    'Schistocyte', 
    'Spherocyte', 
    'Stomatocyte', 
    'TargetCell'
]

class MorphologyService:
    def __init__(self):
        # Point to the user's original Keras 3 model in the root project
        self.model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "morphology_model.keras")
        self.model = None
        self.img_size = (224, 224)
        
    def load_model(self):
        if self.model is None:
            if not os.path.exists(self.model_path):
                print(f"[WARNING] Morphology model not found at {self.model_path}")
                return False
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                print("[INFO] Morphology model loaded successfully.")
                return True
            except Exception as e:
                print(f"[ERROR] Failed to load morphology model: {e}")
                return False
        return True

    def analyze_rbcs(self, image_np: np.ndarray, rbc_boxes: list) -> dict:
        """
        Takes the original image and bounding boxes of detected RBCs.
        Crops each RBC, runs it through the morphology classifier, and returns statistics.
        """
        # If model is not loaded/available, return default empty stats
        if not self.load_model():
            print("[DEBUG] analyze_rbcs: Model not loaded. Returning empty stats.")
            return self._empty_stats()
            
        if not rbc_boxes:
            print("[DEBUG] analyze_rbcs: No RBC boxes provided. Returning empty stats.")
            return self._empty_stats()

        print(f"[DEBUG] analyze_rbcs: Received {len(rbc_boxes)} RBC boxes for processing.")
        
        # Create debug crops directory
        debug_crops_dir = os.path.join(os.path.dirname(__file__), "..", "..", "uploads", "debug_crops")
        os.makedirs(debug_crops_dir, exist_ok=True)
        
        crops = []
        valid_boxes = []
        h, w = image_np.shape[:2]
        
        saved_crops_count = 0

        for box in rbc_boxes:
            x1, y1, x2, y2 = box
            
            # Add a small padding to the box (e.g., 5%) to capture full cell
            pad_x = int((x2 - x1) * 0.05)
            pad_y = int((y2 - y1) * 0.05)
            
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)
            
            # Crop
            crop = image_np[int(cy1):int(cy2), int(cx1):int(cx2)]
            
            # Reject empty crops and tiny background-only crops
            if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                continue
                
            # Save debug crop before resizing
            if saved_crops_count < 50:
                # Assuming image_np is RGB from router, we must convert back to BGR for cv2.imwrite
                debug_crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                crop_path = os.path.join(debug_crops_dir, f"crop_{saved_crops_count}.jpg")
                cv2.imwrite(crop_path, debug_crop_bgr)
                saved_crops_count += 1
                
            # EXACT User-requested Pre-processing flow per crop
            crop = cv2.resize(crop, self.img_size)
            # Router passes RGB already, but adding logic for robustness if called from elsewhere
            # crop = cv2.cvtColor(crop,cv2.COLOR_BGR2RGB) # Handled safely in batch step or router
            crop = crop.astype(np.float32)
            # Expand dims is handled by the batching array below
            
            crops.append(crop)
            valid_boxes.append(box)
            
        print(f"[DEBUG] analyze_rbcs: Successfully cropped {len(crops)} RBCs out of {len(rbc_boxes)}. Saved {saved_crops_count} debug crops.")
        print(f"[DEBUG] analyze_rbcs: Input shape checking - Target size: {self.img_size}")
        
        if not crops:
            return self._empty_stats()

        # Batch prediction
        batch_crops = np.array(crops)
        print(f"[DEBUG] analyze_rbcs: Pre-processing {len(batch_crops)} crops with EfficientNet preprocess_input.")
        batch_crops = preprocess_input(batch_crops) # EfficientNet preprocessing
        
        preds = self.model.predict(batch_crops, verbose=0)
        
        # Calculate statistics and track individual cell predictions
        counts = {cls: 0 for cls in MORPHOLOGY_CLASSES}
        total_rbcs = len(preds)
        
        cell_predictions = []
        
        for idx, pred_vector in enumerate(preds):
            # Sort indices to get top-3 predictions
            top_indices = np.argsort(pred_vector)[::-1]
            top_3_indices = top_indices[:3]
            
            top_class = MORPHOLOGY_CLASSES[top_3_indices[0]]
            top_conf = float(pred_vector[top_3_indices[0]])
            
            # Ensure box coordinates are native python types (int)
            native_box = [int(coord) for coord in valid_boxes[idx]]
            
            counts[top_class] += 1
            cell_predictions.append({
                "box": native_box,
                "class_name": top_class,
                "confidence": top_conf
            })
            
            # Exact requested output format for first 20 cells
            if idx < 20:
                print(f"\nCell {idx+1}")
                print(f"Prediction vector: {np.round(pred_vector, 4)}")
                for rank, c_idx in enumerate(top_3_indices):
                    print(f"{MORPHOLOGY_CLASSES[c_idx]} {pred_vector[c_idx]:.2f}")
            
        print(f"[DEBUG] analyze_rbcs: Morphology Distribution: {counts}")
            
        percentages = {
            cls: round((count / total_rbcs) * 100, 2)
            for cls, count in counts.items()
        }
        
        return {
            "total_analyzed": total_rbcs,
            "counts": counts,
            "percentages": percentages,
            "cell_predictions": cell_predictions
        }
        
    def _empty_stats(self):
        return {
            "total_analyzed": 0,
            "counts": {cls: 0 for cls in MORPHOLOGY_CLASSES},
            "percentages": {cls: 0.0 for cls in MORPHOLOGY_CLASSES},
            "cell_predictions": []
        }

# Singleton instance
morphology_service = MorphologyService()

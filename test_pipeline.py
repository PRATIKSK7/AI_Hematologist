import os
import sys
import cv2
import numpy as np

# Add the backend path to sys.path to resolve imports
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from backend.services.morphology_service import MorphologyService
from backend.services.disease_predictor import DiseasePredictor

def test_pipeline():
    print("=== Testing Morphology Service Pipeline ===")
    morph_service = MorphologyService()
    
    # Create a dummy image (e.g. 800x600 RGB)
    dummy_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    
    # Create 5 dummy RBC bounding boxes [x1, y1, x2, y2]
    # We will make them random sized crops
    dummy_boxes = [
        [100, 100, 150, 150],
        [200, 200, 260, 260],
        [300, 300, 320, 320], # tiny crop
        [400, 400, 480, 480],
        [500, 500, 530, 530]
    ]
    
    print("Testing analyze_rbcs...")
    stats = morph_service.analyze_rbcs(dummy_image, dummy_boxes)
    print("\nMorphology Stats Result:")
    print(stats)
    
    print("\n=== Testing Disease Predictor ===")
    predictor = DiseasePredictor()
    
    wbc_count = 5
    rbc_count = 50
    platelet_count = 20
    
    # Manually test trigger thresholds for Spherocyte
    test_percentages = {"Spherocyte": 5.0, "Normal": 95.0}
    
    risk = predictor.calculate_risk(wbc_count, rbc_count, platelet_count, test_percentages)
    print("\nDisease Risk Result (Expected: Hereditary Spherocytosis):")
    print(risk)

if __name__ == "__main__":
    test_pipeline()

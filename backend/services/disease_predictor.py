import logging

logger = logging.getLogger(__name__)

class DiseasePredictor:
    """
    AI Risk Assessment Engine
    Calculates disease risk scores (0-100) based on YOLOv8 detected cell counts.
    Disclaimer: This is an AI-generated educational assessment. Not a medical diagnosis.
    """
    
    @staticmethod
    def calculate_risk(wbc_count: int, rbc_count: int, platelet_count: int, morphology_percentages: dict) -> dict:
        """
        Calculate health score and identify potential diseases.
        Returns a dictionary with 'disease_risks', 'overall_health_score', and 'reasoning'.
        """
        threshold = 1.0
        
        ida_risk, hs_risk, ha_risk, leuk_risk, inf_risk = 0, 0, 0, 0, 0
        ida_reason, hs_reason, ha_reason, leuk_reason, inf_reason = "Normal", "Normal", "Normal", "Normal", "Normal"
        
        # 1. Microcyte + Hypochromia -> Iron Deficiency
        if morphology_percentages.get("Microcyte", 0.0) > threshold and morphology_percentages.get("Hypochromia", 0.0) > threshold:
            ida_risk = 80
            ida_reason = "High Microcytes and Hypochromia detected. Indicates Iron Deficiency Anemia."
        elif morphology_percentages.get("TargetCell", 0.0) > threshold:
            ida_risk = 60
            ida_reason = "Target cells detected. Possible Thalassemia."
            
        # 2. Spherocyte -> Hereditary Spherocytosis
        if morphology_percentages.get("Spherocyte", 0.0) > threshold:
            hs_risk = 90
            hs_reason = "Spherocytes detected. Possible Hereditary Spherocytosis."
            
        # 3. Schistocyte -> Hemolytic Anemia
        if morphology_percentages.get("Schistocyte", 0.0) > threshold:
            ha_risk = 95
            ha_reason = "Schistocytes detected. Possible Hemolytic Anemia."
        elif morphology_percentages.get("Elliptocyte", 0.0) > threshold:
            ha_risk = 50
            ha_reason = "Elliptocytes detected. Possible Hereditary Elliptocytosis."
            
        # 4. WBC / Platelet proxy for Leukemia / Infection
        if wbc_count > 100:
            leuk_risk = 85
            leuk_reason = "Abnormally high WBC count."
        if wbc_count > 15 and wbc_count <= 100:
            inf_risk = 70
            inf_reason = "Elevated WBC count indicating possible infection."
            
        score = 100 - max([ida_risk, hs_risk, ha_risk, leuk_risk, inf_risk])
        score = max(0, min(100, score))
        
        logger.info(f"Disease risk prediction completed. Health Score: {score}")
        
        return {
            "disease_risks": {
                "iron_deficiency_anemia": ida_risk,
                "hereditary_spherocytosis": hs_risk,
                "hemolytic_anemia": ha_risk,
                "leukemia": leuk_risk,
                "infection": inf_risk
            },
            "overall_health_score": score,
            "reasoning": {
                "iron_deficiency_anemia": ida_reason,
                "hereditary_spherocytosis": hs_reason,
                "hemolytic_anemia": ha_reason,
                "leukemia": leuk_reason,
                "infection": inf_reason
            }
        }

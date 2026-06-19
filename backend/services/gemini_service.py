import os
import time
import logging
from google import genai
from google.genai import types
from backend.config import logger as system_logger
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=env_path)

# Remove any OAuth or Google credential authentication
os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

# Setup dedicated Gemini Logger
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)
gemini_logger = logging.getLogger("gemini_service")
gemini_logger.setLevel(logging.INFO)
if not gemini_logger.handlers:
    fh = logging.FileHandler(os.path.join(log_dir, "gemini.log"))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    gemini_logger.addHandler(fh)

class GeminiService:
    def __init__(self):
        # Load API key from GEMINI_API_KEY inside .env
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip(' "\'')
        self.client = None
        
        system_logger.info(f"API Key Found: {bool(self.api_key)}")
        gemini_logger.info(f"API Key initialization check: {bool(self.api_key)}")
        
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                gemini_logger.info("Gemini Client Initialized Successfully")
            except Exception as e:
                system_logger.error(f"Failed to initialize Gemini Client: {e}")
                gemini_logger.error(f"Initialization Error: {e}", exc_info=True)
        else:
            system_logger.warning("GEMINI_API_KEY not set in .env.")
            gemini_logger.warning("No API key found in .env")

    def is_configured(self) -> bool:
        return self.client is not None

    def test_connection(self) -> str:
        """Endpoint helper to test if Gemini can generate a response"""
        if not self.is_configured():
            raise ValueError("API key not configured")
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents='Say hello',
                config=types.GenerateContentConfig(temperature=0.1)
            )
            return response.text
        except Exception as e:
            if "429" in str(e):
                gemini_logger.warning("Gemini test connection hit rate limits, but API key is valid.")
                return "Valid API Key (Quota Exceeded)"
            gemini_logger.error(f"test_connection failed: {e}")
            raise RuntimeError(f"API request failed: {e}")

    def generate_local_fallback(self, counts: dict, stats: dict, error_msg: str, risk_data: dict = None) -> str:
        """Generates a professional report locally if Gemini completely fails."""
        gemini_logger.warning("Executing local fallback generator.")
        
        fallback = f"""## Professional Hematology Summary
The analysis of this blood smear indicates a total of {counts['total_cells']} detected cells. 
Red Blood Cells (RBC) constitute {counts['rbc_percentage']}% ({counts['rbc_count']} cells) of the sample. 
White Blood Cells (WBC) account for {counts['wbc_percentage']}% ({counts['wbc_count']} cells). 
Platelets represent {counts['platelet_percentage']}% ({counts['platelet_count']} cells).

## Patient-Friendly Explanation
We looked at {counts['total_cells']} cells in your blood sample.
- **Red Blood Cells:** {counts['rbc_count']} (These carry oxygen)
- **White Blood Cells:** {counts['wbc_count']} (These fight infection)
- **Platelets:** {counts['platelet_count']} (These help blood clot)
This is a small snapshot and your doctor will review the full picture.

## Technical AI Analysis
Total Detections: {stats.get('total_detections', 0)}
Average Confidence: {stats.get('avg_confidence', 0) * 100:.2f}%

*Note: Gemini service unavailable. Showing AI-free analysis. (local fallback engine)*

AI-generated educational assessment. Not a medical diagnosis."""
        return fallback

    def generate_report(self, counts: dict, stats: dict, risk_data: dict = None, morphology_data: dict = None) -> str:
        if not self.is_configured():
            gemini_logger.warning("generate_report called but API key is missing.")
            return self.generate_local_fallback(counts, stats, "API Key Missing", risk_data)

        validation_warning = ""
        if counts['wbc_count'] == 0:
            validation_warning += "\nWARNING: No White Blood Cells (WBC) detected. This is highly unusual and may indicate a detection failure or severe leukopenia. Address this anomaly."
        if counts['platelet_count'] == 0:
            validation_warning += "\nWARNING: No Platelets detected. This is a critical anomaly (thrombocytopenia) or a limitation of the current AI model threshold. Discuss this limitation."
        if counts['rbc_count'] > 500:
            validation_warning += "\nWARNING: Extremely high RBC count detected. This could be an artifact of overlapping detection boxes. Mention this potential technical error."
            
        morphology_str = "RBC Morphology Data Not Available"
        if morphology_data and morphology_data.get("total_analyzed", 0) > 0:
            morphology_str = "Detailed RBC Morphology Breakdown:\n"
            for cls, pct in morphology_data.get("percentages", {}).items():
                morphology_str += f"        - {cls}: {pct}% ({morphology_data['counts'][cls]} cells)\n"

        prompt = f"""
        You are an expert AI Hematologist. Analyze the following cell count data, morphology findings, and AI-generated disease risk scores from a blood smear image:
        
        Total Cells: {counts['total_cells']}
        Red Blood Cells (RBC): {counts['rbc_count']} ({counts['rbc_percentage']}%)
        White Blood Cells (WBC): {counts['wbc_count']} ({counts['wbc_percentage']}%)
        Platelets: {counts['platelet_count']} ({counts['platelet_percentage']}%)
        
        {morphology_str}
        
        Detection Statistics:
        Total Detections: {stats['total_detections']}
        Average Confidence: {stats['avg_confidence'] * 100:.2f}%
        
        AI Disease Risk Assessment (0-100 scale):
        Overall Health Score: {risk_data['overall_health_score'] if risk_data else 'N/A'}
        Iron Deficiency Anemia / Thalassemia: {risk_data['disease_risks']['iron_deficiency_anemia'] if risk_data else 'N/A'}
        Hereditary Spherocytosis: {risk_data['disease_risks']['hereditary_spherocytosis'] if risk_data else 'N/A'}
        Hemolytic Anemia / DIC: {risk_data['disease_risks']['hemolytic_anemia'] if risk_data else 'N/A'}
        Leukemia Risk: {risk_data['disease_risks']['leukemia'] if risk_data else 'N/A'}
        Infection Risk: {risk_data['disease_risks']['infection'] if risk_data else 'N/A'}
        
        {validation_warning}

        Provide a structured report containing exactly these sections, clearly marked with markdown headers (H2):
        ## Executive Summary
        ## Cell Count & Morphology Interpretation
        ## Disease Risk Analysis
        ## Potential Concerns
        ## Recommendations
        ## Follow-up Suggestions

        Include this EXACT disclaimer at the end of the report:
        "AI-generated educational assessment. Not a medical diagnosis."
        """

        gemini_logger.info(f"Initiating report generation. Request Payload: {counts}")
        
        models_to_try = ['gemini-2.5-flash', 'gemini-2.5-pro']
        max_retries = 3
        base_delay = 2 # seconds
        last_error_msg = ""
        
        for model in models_to_try:
            gemini_logger.info(f"Attempting inference with model: {model}")
            for attempt in range(max_retries):
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.2,
                        )
                    )
                    gemini_logger.info(f"Successfully generated report using {model} on attempt {attempt+1}")
                    return response.text
                except Exception as e:
                    last_error_msg = str(e)
                    gemini_logger.error(f"[{model}] attempt {attempt+1}/{max_retries} failed: {e}")
                    
                    if attempt < max_retries - 1:
                        time.sleep(base_delay ** attempt)
                        
        gemini_logger.error(f"All models and retries exhausted. Final error: {last_error_msg}")
        return self.generate_local_fallback(counts, stats, "API/Network Error", risk_data)

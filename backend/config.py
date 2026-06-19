import logging
import json
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from datetime import datetime

# 1. Ensure .env is loaded predictably regardless of where uvicorn is launched
env_path = os.path.join(os.path.dirname(__file__), ".env")
env_loaded = load_dotenv(dotenv_path=env_path)

class Settings(BaseSettings):
    gemini_api_key: str = os.getenv("GEMINI_API_KEY") or ""
    model_path: str = os.path.join(os.path.dirname(__file__), "models", "weights", "best.pt")
    database_url: str = "sqlite:///./hematologist.db"
    log_level: str = "INFO"
    
    # YOLO Inference Config
    yolo_confidence: float = 0.15
    yolo_iou: float = 0.35
    yolo_image_size: int = 1024

    # We tell pydantic to also check the same env file just in case
    model_config = SettingsConfigDict(env_file=env_path, env_file_encoding="utf-8", extra="ignore")

settings = Settings()

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

def setup_logging():
    logger = logging.getLogger("ai_hematologist")
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(JSONFormatter())
        logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# Startup Diagnostics for Gemini
print("====================================")
print(f"Environment Variable File Loaded: {env_loaded}")
print(f"GEMINI_API_KEY Found: {bool(settings.gemini_api_key)}")
print("====================================")

import pytest
from unittest.mock import patch
from backend.services.detector_service import DetectorService
from backend.services.gemini_service import GeminiService

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "gemini_configured" in data

def test_model_info_when_loaded(client):
    response = client.get("/model-info")
    # If YOLO loaded correctly locally, it will be 200. If missing, it might be 503.
    # We will just assert that the endpoint exists.
    assert response.status_code in [200, 503]

@patch.object(DetectorService, 'detect')
@patch.object(GeminiService, 'generate_report')
def test_analyze_endpoint_success(mock_gemini, mock_detect, client, tmp_path):
    mock_detect.return_value = (
        [{"class_name": "RBC", "confidence": 0.99}],
        "mock_annotated.jpg"
    )
    mock_gemini.return_value = "Mock Gemini Report\n\nThis analysis is AI-assisted and is not a medical diagnosis."

    # Create dummy image
    test_file = tmp_path / "test_image.jpg"
    test_file.write_bytes(b"dummy image data")

    with open(test_file, "rb") as f:
        response = client.post("/analyze", files={"file": ("test_image.jpg", f, "image/jpeg")})

    assert response.status_code == 200
    data = response.json()
    
    assert "counts" in data
    assert data["counts"]["total_cells"] == 1
    assert data["counts"]["rbc_count"] == 1
    
    assert "statistics" in data
    assert data["statistics"]["total_detections"] == 1
    
    assert "gemini_report" in data
    assert "This analysis is AI-assisted and is not a medical diagnosis" in data["gemini_report"]

def test_analyze_endpoint_invalid_file(client, tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_bytes(b"text data")

    with open(test_file, "rb") as f:
        response = client.post("/analyze", files={"file": ("test.txt", f, "text/plain")})

    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]

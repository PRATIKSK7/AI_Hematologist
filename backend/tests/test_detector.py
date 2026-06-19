from unittest.mock import patch, MagicMock
from backend.services.detector_service import DetectorService
import numpy as np

@patch("backend.services.detector_service.YOLO")
@patch("backend.services.detector_service.os.path.exists")
def test_detector_service_singleton(mock_exists, mock_yolo):
    mock_exists.return_value = True
    
    detector1 = DetectorService.get_instance()
    detector2 = DetectorService.get_instance()
    
    assert detector1 is detector2
    assert detector1.is_loaded() is True

@patch("backend.services.detector_service.YOLO")
@patch("backend.services.detector_service.os.path.exists")
def test_detector_service_inference(mock_exists, mock_yolo, tmp_path):
    mock_exists.return_value = True
    
    mock_model_instance = MagicMock()
    mock_model_instance.names = {0: "RBC", 1: "WBC"}
    
    mock_result = MagicMock()
    
    class MockTensor:
        def __init__(self, val):
            self.val = val
        def item(self):
            return self.val

    mock_box = MagicMock()
    mock_box.cls = [MockTensor(0)]
    mock_box.conf = [MockTensor(0.95)]
    
    mock_result.plot.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_result.boxes = [mock_box]
    
    mock_model_instance.return_value = [mock_result]
    mock_yolo.return_value = mock_model_instance

    detector = DetectorService()
    assert detector.is_loaded() is True
    
    output_dir = tmp_path / "uploads"
    detections, annotated_path = detector.detect("fake_image.jpg", str(output_dir))
    
    assert len(detections) == 1
    assert detections[0]["class_name"] == "RBC"
    assert detections[0]["confidence"] == 0.95
    assert "annotated_fake_image.jpg" in annotated_path

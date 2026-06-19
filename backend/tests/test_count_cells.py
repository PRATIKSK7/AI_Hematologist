from backend.services.count_cells import calculate_cell_counts, calculate_statistics

def test_calculate_cell_counts():
    mock_detections = [
        {"class_name": "RBC", "confidence": 0.95},
        {"class_name": "RBC", "confidence": 0.90},
        {"class_name": "WBC", "confidence": 0.88},
        {"class_name": "Platelets", "confidence": 0.92},
        {"class_name": "Platelets", "confidence": 0.85},
    ]

    counts = calculate_cell_counts(mock_detections)
    
    assert counts["total_cells"] == 5
    assert counts["rbc_count"] == 2
    assert counts["wbc_count"] == 1
    assert counts["platelet_count"] == 2
    
    assert counts["rbc_percentage"] == 40.0
    assert counts["wbc_percentage"] == 20.0
    assert counts["platelet_percentage"] == 40.0

def test_calculate_empty_counts():
    counts = calculate_cell_counts([])
    assert counts["total_cells"] == 0
    assert counts["rbc_percentage"] == 0.0

def test_calculate_statistics():
    mock_detections = [
        {"class_name": "RBC", "confidence": 0.90},
        {"class_name": "WBC", "confidence": 0.80},
    ]
    stats = calculate_statistics(mock_detections)
    assert stats["total_detections"] == 2
    assert stats["avg_confidence"] == 0.85

def test_empty_statistics():
    stats = calculate_statistics([])
    assert stats["total_detections"] == 0
    assert stats["avg_confidence"] == 0.0

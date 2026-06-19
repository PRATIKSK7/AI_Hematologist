from typing import List, Dict, Any
import math

def remove_duplicate_detections(detections: List[Dict[str, Any]], distance_threshold: float = 30.0) -> List[Dict[str, Any]]:
    """
    Advanced Post-Processing: Removes overlapping bounding boxes by calculating centroid distances.
    Only keeps the highest confidence detection if two centroids of the same class are too close.
    """
    if not detections:
        return []
        
    filtered = []
    
    # Sort by confidence descending so we naturally keep the best ones
    sorted_dets = sorted(detections, key=lambda x: x.get("confidence", 0), reverse=True)
    
    for det in sorted_dets:
        bbox = det.get("bbox")
        if not bbox:
            filtered.append(det)
            continue
            
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        det["centroid"] = (cx, cy)
        
        is_duplicate = False
        for f in filtered:
            if f.get("class_name") != det.get("class_name"):
                continue # Only apply NMS/Centroid check within same class
            if "centroid" not in f:
                continue
            
            fcx, fcy = f["centroid"]
            dist = math.hypot(cx - fcx, cy - fcy)
            if dist < distance_threshold:
                is_duplicate = True
                break
                
        if not is_duplicate:
            filtered.append(det)
            
    # Clean up centroid keys before returning
    for f in filtered:
        f.pop("centroid", None)
        
    return filtered

def calculate_cell_counts(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    rbc_count = 0
    wbc_count = 0
    platelet_count = 0
    
    for det in detections:
        cname = str(det.get("class_name", "")).lower()
        if "rbc" in cname or "red" in cname or cname == "0":
            rbc_count += 1
        elif "wbc" in cname or "white" in cname or cname == "1":
            wbc_count += 1
        elif "platelet" in cname or cname == "2":
            platelet_count += 1
            
    total_cells = rbc_count + wbc_count + platelet_count
    
    rbc_percentage = round((rbc_count / total_cells * 100), 1) if total_cells > 0 else 0.0
    wbc_percentage = round((wbc_count / total_cells * 100), 1) if total_cells > 0 else 0.0
    platelet_percentage = round((platelet_count / total_cells * 100), 1) if total_cells > 0 else 0.0

    confidence_score = round(sum(d.get("confidence", 0) for d in detections) / len(detections), 4) if detections else 0.0

    return {
        "rbc_count": rbc_count,
        "wbc_count": wbc_count,
        "platelet_count": platelet_count,
        "total_cells": total_cells,
        "rbc_percentage": rbc_percentage,
        "wbc_percentage": wbc_percentage,
        "platelet_percentage": platelet_percentage,
        "confidence_score": confidence_score
    }

def calculate_statistics(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not detections:
        return {"avg_confidence": 0.0, "total_detections": 0, "class_confidences": {}}
        
    avg_conf = sum(d["confidence"] for d in detections) / len(detections)
    
    # Class-specific confidence averages
    class_confs = {}
    class_counts = {}
    for d in detections:
        cname = d.get("class_name", "Unknown")
        class_confs[cname] = class_confs.get(cname, 0) + d["confidence"]
        class_counts[cname] = class_counts.get(cname, 0) + 1
        
    for cname in class_confs:
        class_confs[cname] = round(class_confs[cname] / class_counts[cname], 4)
        
    return {
        "avg_confidence": round(avg_conf, 4),
        "total_detections": len(detections),
        "class_confidences": class_confs
    }

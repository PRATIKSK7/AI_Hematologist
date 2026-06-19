#!/bin/bash

# ==============================================================================
# AI HEMATOLOGIST - PRODUCTION TRAINING SCRIPT
# ==============================================================================
# This script executes a production-grade retraining of the YOLOv8s model
# targeting the BCCD dataset to fix massive class imbalance (11:1 RBC:Platelet).
# 
# Key Optimizations:
# - imgsz=1024 : High resolution for detecting microscopic Platelets.
# - mixup=0.2 & mosaic=1.0 : Augmentations to improve minority class recall.
# - optimizer=AdamW : Robust weight decay handling.
# - patience=30 : Early stopping for efficiency.
# ==============================================================================

# Activate virtual environment if running inside backend
if [ -d "backend/venv" ]; then
    echo "Activating virtual environment..."
    source backend/venv/bin/activate
fi

echo "Starting YOLOv8s Production Training Pipeline..."

# Assuming data.yaml exists in the datasets directory or we pass the absolute path
DATASET_PATH="/Users/pratikskanoj/Downloads/AI_Hematologist/datasets/bccd/data.yaml"

yolo task=detect mode=train \
    model=yolov8s.pt \
    data=$DATASET_PATH \
    epochs=150 \
    imgsz=1024 \
    batch=4 \
    patience=30 \
    optimizer=AdamW \
    mixup=0.2 \
    mosaic=1.0 \
    project="AI_Hematologist_Production" \
    name="bccd_run" \
    exist_ok=True

echo "Training pipeline initiated/completed. Weights will be saved in AI_Hematologist_Production/bccd_run/weights/"

"""
Configuration Module
====================

Stores file paths, model hyperparameters, and UI settings for the project.
Centralizing these values avoids scattering "magic numbers" across different modules.
"""

import os
from typing import List, Dict, Any

# =============================================================================
# Directories & File Paths
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Filenames
RAW_DATA_FILE = "breast_cancer.csv"
PROCESSED_DATA_FILE = "breast_cancer_cleaned.csv"
PIPELINE_FILE = "pipeline.pkl"
MEDIANS_FILE = "feature_medians.json"

# =============================================================================
# Machine Learning Constants
# =============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COLUMN = "diagnosis"
POSITIVE_LABEL = "M"
NEGATIVE_LABEL = "B"

# Selected features derived from Random Forest feature importance
SELECTED_FEATURES: List[str] = [
    "concave points_worst",
    "perimeter_worst",
    "concave points_mean",
    "radius_worst",
    "area_worst"
]

# Random Forest Hyperparameters
MODEL_HYPERPARAMETERS: Dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 10,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

# =============================================================================
# UI Constants
# =============================================================================
UI_SLIDER_BOUNDS: Dict[str, Dict[str, float]] = {
    "concave points_worst": {"minimum": 0.0, "maximum": 0.3, "default": 0.10, "step": 0.01},
    "perimeter_worst": {"minimum": 50.0, "maximum": 255.0, "default": 97.66, "step": 1.0},
    "concave points_mean": {"minimum": 0.0, "maximum": 0.25, "default": 0.03, "step": 0.01},
    "radius_worst": {"minimum": 7.0, "maximum": 40.0, "default": 14.97, "step": 0.1},
    "area_worst": {"minimum": 100.0, "maximum": 4500.0, "default": 686.5, "step": 10.0},
}

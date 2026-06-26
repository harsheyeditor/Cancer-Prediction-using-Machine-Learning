"""
Configuration Module - Cancer Prediction Project
================================================

This module serves as the single source of truth for all project configurations.
It stores file paths, magic numbers, model hyperparameters, and column definitions.

Author: Cancer Prediction ML Project
"""

import os
from typing import List

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
MODEL_HYPERPARAMETERS = {
    "n_estimators": 200,
    "max_depth": 10,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

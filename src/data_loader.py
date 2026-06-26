"""
Data Loader Module - Cancer Prediction Project
================================================

This module handles loading raw and processed datasets.
It provides a single, consistent interface for accessing data
throughout the project, regardless of whether the data comes
from sklearn, a CSV file, or a URL.

Why a separate data_loader module?
----------------------------------
1. DRY Principle: Load data once, use everywhere
2. Single source of truth: If the data path changes, fix it in one place
3. Testability: Easy to mock for unit tests
4. Consistency: All notebooks use the same loading logic

Author: Cancer Prediction ML Project
"""

import os
import logging
from typing import Tuple, Optional

import pandas as pd
import numpy as np

# Configure logging for this module
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================
# Using relative paths from the project root
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RAW_FILENAME = "breast_cancer.csv"
PROCESSED_FILENAME = "breast_cancer_cleaned.csv"


def load_from_sklearn() -> pd.DataFrame:
    """
    Load the Wisconsin Breast Cancer dataset from sklearn.

    This is the primary data source. sklearn bundles a clean copy
    of the UCI dataset, making it ideal for reproducibility.

    Returns:
        pd.DataFrame: Raw dataset with 569 rows and 32 columns
                      (30 features + 'id' + 'diagnosis')

    Raises:
        ImportError: If sklearn is not installed

    Example:
        >>> df = load_from_sklearn()
        >>> print(df.shape)
        (569, 32)
    """
    try:
        from sklearn.datasets import load_breast_cancer

        # Load the sklearn dataset
        data = load_breast_cancer()

        # Convert to DataFrame with proper column names
        df = pd.DataFrame(data.data, columns=data.feature_names)
        
        # Rename sklearn columns to match project standards (e.g. 'worst radius' -> 'radius_worst')
        rename_map = {
            'mean radius': 'radius_mean', 'mean texture': 'texture_mean', 'mean perimeter': 'perimeter_mean',
            'mean area': 'area_mean', 'mean smoothness': 'smoothness_mean', 'mean compactness': 'compactness_mean',
            'mean concavity': 'concavity_mean', 'mean concave points': 'concave points_mean', 'mean symmetry': 'symmetry_mean',
            'mean fractal dimension': 'fractal dimension_mean', 'radius error': 'radius_se', 'texture error': 'texture_se',
            'perimeter error': 'perimeter_se', 'area error': 'area_se', 'smoothness error': 'smoothness_se',
            'compactness error': 'compactness_se', 'concavity error': 'concavity_se', 'concave points error': 'concave points_se',
            'symmetry error': 'symmetry_se', 'fractal dimension error': 'fractal dimension_se', 'worst radius': 'radius_worst',
            'worst texture': 'texture_worst', 'worst perimeter': 'perimeter_worst', 'worst area': 'area_worst',
            'worst smoothness': 'smoothness_worst', 'worst compactness': 'compactness_worst', 'worst concavity': 'concavity_worst',
            'worst concave points': 'concave points_worst', 'worst symmetry': 'symmetry_worst', 'worst fractal dimension': 'fractal dimension_worst'
        }
        df = df.rename(columns=rename_map)

        # Add target column: sklearn uses 0=malignant, 1=benign
        # We'll map to string labels first for clarity
        df["diagnosis"] = pd.Series(data.target).map({0: "M", 1: "B"})

        # Add an ID column (sklearn doesn't include one)
        df.insert(0, "id", range(1, len(df) + 1))

        logger.info(
            "Loaded breast cancer dataset from sklearn: %d rows, %d columns",
            df.shape[0],
            df.shape[1],
        )
        return df

    except ImportError as e:
        logger.error("sklearn is not installed: %s", e)
        raise


def save_raw_data(df: pd.DataFrame) -> str:
    """
    Save raw dataset to the data/raw/ directory.

    We always keep a copy of the original, unmodified data.
    This is critical for reproducibility - you should always be
    able to trace back to the original source.

    Args:
        df: The raw DataFrame to save

    Returns:
        str: Path to the saved file
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    filepath = os.path.join(RAW_DATA_DIR, RAW_FILENAME)
    df.to_csv(filepath, index=False)
    logger.info("Raw data saved to: %s", filepath)
    return filepath


def load_raw_data() -> pd.DataFrame:
    """
    Load raw data from the data/raw/ directory.

    Returns:
        pd.DataFrame: The raw dataset

    Raises:
        FileNotFoundError: If raw data hasn't been saved yet
    """
    filepath = os.path.join(RAW_DATA_DIR, RAW_FILENAME)
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Raw data not found at {filepath}. "
            "Run load_from_sklearn() and save_raw_data() first."
        )
    df = pd.read_csv(filepath)
    logger.info("Loaded raw data from: %s (%d rows)", filepath, len(df))
    return df


def load_processed_data() -> pd.DataFrame:
    """
    Load cleaned/processed data from the data/processed/ directory.

    Returns:
        pd.DataFrame: The cleaned dataset

    Raises:
        FileNotFoundError: If processed data hasn't been created yet
    """
    filepath = os.path.join(PROCESSED_DATA_DIR, PROCESSED_FILENAME)
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Processed data not found at {filepath}. "
            "Run the preprocessing notebook (02_preprocessing.ipynb) first."
        )
    df = pd.read_csv(filepath)
    logger.info("Loaded processed data from: %s (%d rows)", filepath, len(df))
    return df


def get_feature_descriptions() -> dict:
    """
    Return a dictionary mapping feature names to human-readable descriptions.

    The Wisconsin Breast Cancer dataset has 30 features computed from
    digitized FNA (Fine Needle Aspirate) images. Each cell nucleus has
    10 base measurements, computed as mean, standard error, and "worst"
    (mean of the 3 largest values).

    Returns:
        dict: Feature name -> description mapping
    """
    base_features = {
        "radius": "Mean distance from center to perimeter points",
        "texture": "Standard deviation of gray-scale values",
        "perimeter": "Perimeter of the cell nucleus",
        "area": "Area of the cell nucleus",
        "smoothness": "Local variation in radius lengths",
        "compactness": "Perimeter^2 / Area - 1.0",
        "concavity": "Severity of concave portions of the contour",
        "concave points": "Number of concave portions of the contour",
        "symmetry": "Symmetry of the cell nucleus",
        "fractal dimension": "Coastline approximation - 1 (complexity)",
    }

    descriptions = {}
    for suffix, suffix_desc in [
        ("mean", "Average value"),
        ("se", "Standard error"),
        ("worst", "Mean of 3 largest values"),
    ]:
        for feature, desc in base_features.items():
            col_name = f"{feature}_{suffix}" if "_" not in feature else f"{feature.replace(' ', '_')}_{suffix}"
            # Handle sklearn column naming (uses spaces)
            sklearn_name = f"{feature} {suffix}" if " " not in feature else f"{feature} {suffix}"
            descriptions[sklearn_name] = f"{desc} ({suffix_desc})"

    return descriptions

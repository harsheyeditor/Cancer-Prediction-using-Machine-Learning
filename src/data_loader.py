"""
Data Loader Module
==================

Handles loading raw and processed datasets. Centralizing data 
loading logic ensures a single source of truth for file paths 
and formats, keeping training and exploratory code consistent.
"""

import os
import logging
from typing import Tuple, Optional

import pandas as pd
import numpy as np

# Configure logging for this module
logger = logging.getLogger(__name__)


from src import config


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
            'mean radius': 'radius_mean',
            'mean texture': 'texture_mean',
            'mean perimeter': 'perimeter_mean',
            'mean area': 'area_mean',
            'mean smoothness': 'smoothness_mean',
            'mean compactness': 'compactness_mean',
            'mean concavity': 'concavity_mean',
            'mean concave points': 'concave points_mean',
            'mean symmetry': 'symmetry_mean',
            'mean fractal dimension': 'fractal dimension_mean',
            'radius error': 'radius_se',
            'texture error': 'texture_se',
            'perimeter error': 'perimeter_se',
            'area error': 'area_se',
            'smoothness error': 'smoothness_se',
            'compactness error': 'compactness_se',
            'concavity error': 'concavity_se',
            'concave points error': 'concave points_se',
            'symmetry error': 'symmetry_se',
            'fractal dimension error': 'fractal dimension_se',
            'worst radius': 'radius_worst',
            'worst texture': 'texture_worst',
            'worst perimeter': 'perimeter_worst',
            'worst area': 'area_worst',
            'worst smoothness': 'smoothness_worst',
            'worst compactness': 'compactness_worst',
            'worst concavity': 'concavity_worst',
            'worst concave points': 'concave points_worst',
            'worst symmetry': 'symmetry_worst',
            'worst fractal dimension': 'fractal dimension_worst'
        }
        df = df.rename(columns=rename_map)

        # Add target column: sklearn uses 0=malignant, 1=benign
        # We'll map to string labels first for clarity
        df["diagnosis"] = pd.Series(data.target).map({0: config.POSITIVE_LABEL, 1: config.NEGATIVE_LABEL})

        # Add an ID column (sklearn doesn't include one)
        df.insert(0, "id", range(1, len(df) + 1))

        logger.info(
            "Loaded breast cancer dataset from sklearn: %d rows, %d columns",
            df.shape[0],
            df.shape[1],
        )
        return df

    except Exception as e:
        logger.error("Failed to load sklearn dataset: %s", e)
        raise


def save_raw_data(df: pd.DataFrame) -> str:
    """
    Save raw dataset to the data/raw/ directory.
    
    Keeping an unmodified copy of the raw data allows us to 
    safely rerun or alter preprocessing steps later without 
    re-downloading.

    Args:
        df: The raw DataFrame to save

    Returns:
        str: Path to the saved file
    """
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    filepath = os.path.join(config.RAW_DATA_DIR, config.RAW_DATA_FILE)
    
    try:
        df.to_csv(filepath, index=False)
        logger.info("Raw data saved to: %s", filepath)
        return filepath
    except Exception as e:
        logger.error("Failed to save raw data: %s", e)
        raise


def load_raw_data() -> pd.DataFrame:
    """
    Load raw data from the data/raw/ directory.

    Returns:
        pd.DataFrame: The raw dataset

    Raises:
        FileNotFoundError: If raw data hasn't been saved yet
    """
    filepath = os.path.join(config.RAW_DATA_DIR, config.RAW_DATA_FILE)
    if not os.path.exists(filepath):
        error_msg = f"Raw data not found at {filepath}. Run load_from_sklearn() and save_raw_data() first."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    try:
        df = pd.read_csv(filepath)
        logger.info("Loaded raw data from: %s (%d rows)", filepath, len(df))
        return df
    except Exception as e:
        logger.error("Error reading raw data CSV: %s", e)
        raise


def load_processed_data() -> pd.DataFrame:
    """
    Load cleaned/processed data from the data/processed/ directory.

    Returns:
        pd.DataFrame: The cleaned dataset

    Raises:
        FileNotFoundError: If processed data hasn't been created yet
    """
    filepath = os.path.join(config.PROCESSED_DATA_DIR, config.PROCESSED_DATA_FILE)
    if not os.path.exists(filepath):
        error_msg = f"Processed data not found at {filepath}. Run the preprocessing pipeline first."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    try:
        df = pd.read_csv(filepath)
        logger.info("Loaded processed data from: %s (%d rows)", filepath, len(df))
        return df
    except Exception as e:
        logger.error("Error reading processed data CSV: %s", e)
        raise



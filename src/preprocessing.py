"""
Preprocessing Module - Cancer Prediction Project
==================================================

Reusable functions for data cleaning and preprocessing.
This module is imported by notebooks and the main pipeline to ensure
consistent preprocessing across all stages.

Why a separate preprocessing module?
-------------------------------------
1. DRY: Same cleaning logic used in notebooks AND production pipeline
2. Testable: Functions can be unit tested independently
3. Versioned: Changes to preprocessing are tracked in Git
4. Portable: Easy to import in Flask/FastAPI deployment

Author: Cancer Prediction ML Project
"""

import logging
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src import config

# Configure logging
logger = logging.getLogger(__name__)


def drop_unnecessary_columns(
    df: pd.DataFrame,
    columns_to_drop: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Remove columns that are not useful for prediction.

    Default columns dropped:
    - 'id': Patient identifier with no predictive value
    - 'Unnamed: 32': Artifact from CSV formatting

    Args:
        df: Input DataFrame
        columns_to_drop: List of column names to remove.
                         If None, uses default list.

    Returns:
        pd.DataFrame: DataFrame with specified columns removed

    Example:
        >>> df_clean = drop_unnecessary_columns(df)
        >>> 'id' in df_clean.columns
        False
    """
    if columns_to_drop is None:
        columns_to_drop = ["id", "Unnamed: 32"]

    existing = [col for col in columns_to_drop if col in df.columns]
    df_cleaned = df.drop(columns=existing, errors="ignore")

    logger.info(
        "Dropped columns: %s (requested: %s)",
        existing,
        columns_to_drop,
    )
    return df_cleaned


def encode_target(
    df: pd.DataFrame,
    target_col: str = None,
    positive_label: str = None,
    negative_label: str = None,
) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Encode the binary target variable to numeric (0/1).

    Malignant (M) -> 1 (positive class)
    Benign (B)    -> 0 (negative class)

    Args:
        df: Input DataFrame with string target column
        target_col: Name of the target column (defaults to config.TARGET_COLUMN)
        positive_label: Label for the positive class (defaults to config.POSITIVE_LABEL)
        negative_label: Label for the negative class (defaults to config.NEGATIVE_LABEL)

    Returns:
        Tuple of:
        - pd.DataFrame: DataFrame with encoded target
        - LabelEncoder: Fitted encoder (for inverse_transform later)

    Raises:
        ValueError: If target column contains unexpected values

    Example:
        >>> df_encoded, encoder = encode_target(df)
        >>> df_encoded['diagnosis'].unique()
        array([1, 0])
    """
    target_col = target_col or config.TARGET_COLUMN
    positive_label = positive_label or config.POSITIVE_LABEL
    negative_label = negative_label or config.NEGATIVE_LABEL

    # Validate target values
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    unique_vals = df[target_col].dropna().unique()
    expected = {positive_label, negative_label}
    actual = set(unique_vals)

    if not actual.issubset(expected):
        raise ValueError(
            f"Unexpected values in '{target_col}': {actual - expected}. "
            f"Expected only {expected}."
        )

    # Apply encoding
    le = LabelEncoder()
    df = df.copy()  # Don't modify the original
    df[target_col] = le.fit_transform(df[target_col])

    # Verify: B=0, M=1 (LabelEncoder sorts alphabetically)
    assert le.transform([negative_label])[0] == 0, f"{negative_label} should be 0"
    assert le.transform([positive_label])[0] == 1, f"{positive_label} should be 1"

    logger.info(
        "Encoded '%s': %s->0, %s->1",
        target_col,
        negative_label,
        positive_label,
    )
    return df, le


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Strategy:
    1. Drop columns where more than `threshold` fraction of values are missing
    2. Fill remaining missing values using the specified strategy

    Args:
        df: Input DataFrame
        strategy: Imputation strategy ('mean', 'median', 'mode', 'drop')
        threshold: Drop column if more than this fraction is missing (0-1)

    Returns:
        pd.DataFrame: DataFrame with missing values handled
    """
    df = df.copy()

    # Check for missing values
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()

    if cols_to_drop:
        logger.warning("Dropping columns with >%.0f%% missing: %s", threshold * 100, cols_to_drop)
        df = df.drop(columns=cols_to_drop)

    # Impute remaining missing values
    remaining_missing = df.isnull().sum()
    cols_with_missing = remaining_missing[remaining_missing > 0].index.tolist()

    if cols_with_missing:
        if strategy == "median":
            df[cols_with_missing] = df[cols_with_missing].fillna(
                df[cols_with_missing].median()
            )
        elif strategy == "mean":
            df[cols_with_missing] = df[cols_with_missing].fillna(
                df[cols_with_missing].mean()
            )
        elif strategy == "mode":
            for col in cols_with_missing:
                df[col] = df[col].fillna(df[col].mode()[0])
        elif strategy == "drop":
            df = df.dropna()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        logger.info(
            "Imputed %d columns using '%s' strategy",
            len(cols_with_missing),
            strategy,
        )
    else:
        logger.info("No missing values found - no imputation needed")

    return df


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Run the complete cleaning pipeline.

    This is the main entry point for data cleaning. It performs:
    1. Drop unnecessary columns (id, Unnamed: 32)
    2. Handle missing values (if any)
    3. Remove duplicates
    4. Encode the target variable

    Args:
        df: Raw DataFrame as loaded from data/raw/

    Returns:
        Tuple of:
        - pd.DataFrame: Fully cleaned DataFrame
        - LabelEncoder: Fitted label encoder for the target

    Example:
        >>> from data_loader import load_raw_data
        >>> raw_df = load_raw_data()
        >>> clean_df, encoder = clean_dataset(raw_df)
        >>> clean_df.shape
        (569, 31)
    """
    logger.info("Starting cleaning pipeline...")

    # Step 1: Drop unnecessary columns
    df = drop_unnecessary_columns(df)

    # Step 2: Handle missing values
    df = handle_missing_values(df)

    # Step 3: Remove duplicates
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        df = df.drop_duplicates()
        logger.info("Removed %d duplicate rows", n_dupes)
    else:
        logger.info("No duplicates found")

    # Step 4: Encode target
    df, le = encode_target(df)

    logger.info(
        "Cleaning pipeline complete: %d rows x %d columns",
        df.shape[0],
        df.shape[1],
    )
    return df, le




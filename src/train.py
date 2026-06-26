"""
Training Module - Cancer Prediction Project
==================================================

Standalone script for training the final model.
This script loads the dataset, processes it using the shared preprocessing module,
splits it into train and test sets, trains a pipeline, evaluates it,
and saves the artifacts to the models/ directory.

Usage:
    python -m src.train
"""

import os
import json
import logging

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from src import config
from src.data_loader import load_from_sklearn, save_raw_data
from src.preprocessing import clean_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directories():
    """Ensure all required output directories exist."""
    os.makedirs(os.path.dirname(os.path.join(config.PROCESSED_DATA_DIR, config.PROCESSED_DATA_FILE)), exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)

def train_and_save_model():
    """Train the final production pipeline and save artifacts."""
    create_directories()
    
    # 1. Load Data
    logger.info("Loading breast cancer dataset...")
    df_raw = load_from_sklearn()
    save_raw_data(df_raw)
    
    # 2. Preprocess Data
    logger.info("Cleaning dataset...")
    df_clean, _ = clean_dataset(df_raw)
    
    # 3. Feature Selection
    selected_features = config.SELECTED_FEATURES
    for f in selected_features:
        if f not in df_clean.columns:
            raise KeyError(f"Feature '{f}' not found in dataset. Pipeline cannot proceed.")
            
    X = df_clean[selected_features]
    y = df_clean[config.TARGET_COLUMN]
    
    # Save processed data
    processed_path = os.path.join(config.PROCESSED_DATA_DIR, config.PROCESSED_DATA_FILE)
    df_clean.to_csv(processed_path, index=False)
    logger.info("Saved processed data to %s", processed_path)
    
    # Calculate and save feature medians for fallback during inference
    medians = X.median().to_dict()
    medians_path = os.path.join(config.MODEL_DIR, config.MEDIANS_FILE)
    with open(medians_path, 'w') as f:
        json.dump(medians, f, indent=4)
    logger.info("Saved feature medians to %s", medians_path)

    # 4. Train-Test Split (Reproducible)
    logger.info("Splitting data into train and test sets (test_size=%.2f)...", config.TEST_SIZE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    # 5. Build and Train Pipeline
    logger.info("Building and training the ML pipeline...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(**config.MODEL_HYPERPARAMETERS))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # 6. Evaluate Model
    logger.info("Evaluating model on test set...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    logger.info("Evaluation Results:")
    logger.info(f"  Accuracy:  {acc:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")
    logger.info(f"  ROC-AUC:   {roc_auc:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=[config.NEGATIVE_LABEL, config.POSITIVE_LABEL]))
    
    # 7. Save Pipeline
    pipeline_out = os.path.join(config.MODEL_DIR, config.PIPELINE_FILE)
    joblib.dump(pipeline, pipeline_out)
    logger.info(f"[PASS] Pipeline saved to: {pipeline_out}")
    
    # Smoke Test
    try:
        from src.predict import CancerPredictor
        predictor = CancerPredictor(model_dir=config.MODEL_DIR)
        test_patient = {f: 10.0 for f in selected_features}
        res = predictor.predict(test_patient)
        logger.info(f"Smoke test successful: Prediction = {res['diagnosis']}")
    except ImportError as e:
        logger.warning(f"Smoke test skipped: Could not import CancerPredictor ({e}).")
    except Exception as e:
        logger.error(f"Smoke test failed: {e}")

if __name__ == "__main__":
    train_and_save_model()

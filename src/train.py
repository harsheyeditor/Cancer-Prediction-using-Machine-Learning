"""
Training Module - Cancer Prediction Project
==================================================

Standalone script for training the final model.
This script loads the dataset, processes it using the shared preprocessing module,
trains the model, and saves the artifacts to the models/ directory.

Usage:
    python src/train.py
"""

import os
import json
import logging
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data_loader import load_from_sklearn, save_raw_data
from preprocessing import clean_dataset, scale_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Absolute paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "breast_cancer_cleaned.csv")
FEATURES_PATH = os.path.join(DATA_DIR, "processed", "selected_features.json")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_OUT = os.path.join(MODEL_DIR, "cancer_prediction_model.pkl")
SCALER_OUT = os.path.join(MODEL_DIR, "scaler.pkl")

def create_directories():
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_save_model():
    """Train the final production model and save artifacts."""
    create_directories()
    
    # 1. Load Data using data_loader
    logger.info("Loading breast cancer dataset using data_loader...")
    df_raw = load_from_sklearn()
    
    # Save the raw data (which now has correct feature names and 'M'/'B' targets)
    save_raw_data(df_raw)
    
    # 2. Preprocess Data using preprocessing module
    logger.info("Cleaning dataset...")
    # clean_dataset drops 'id', handles missing values, and encodes M->1, B->0
    df_clean, label_encoder = clean_dataset(df_raw)
    
    # 3. Feature Selection
    selected_features = [
        'concave points_worst',
        'perimeter_worst',
        'concave points_mean',
        'radius_worst',
        'area_worst'
    ]
    
    # Verify features exist
    for f in selected_features:
        if f not in df_clean.columns:
            logger.error(f"Feature {f} not found in dataframe.")
            
    X = df_clean[selected_features]
    y = df_clean['diagnosis']
    
    # Save processed data and selected features
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
    with open(FEATURES_PATH, 'w') as f:
        json.dump(selected_features, f)
    logger.info("Saved processed data and features.")
    
    # 4. Scale features
    logger.info("Scaling features...")
    # scale_features takes (X_train, X_test). Since we are training the final model on ALL data,
    # we just pass X as both train and test to get the fitted scaler and scaled data.
    X_scaled, _, scaler = scale_features(X, X)
    
    # 5. Train model
    logger.info("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)
    
    # 6. Save artifacts
    joblib.dump(model, MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)
    logger.info(f"[PASS] Model saved to: {MODEL_OUT}")
    logger.info(f"[PASS] Scaler saved to: {SCALER_OUT}")
    
    # Validation Check
    if os.path.exists(MODEL_OUT) and os.path.exists(SCALER_OUT):
        logger.info("Validation: Model artifacts exist.")
    else:
        logger.error("Validation: Model artifacts missing.")
        
    # Smoke Test
    try:
        from predict import CancerPredictor
        predictor = CancerPredictor(model_dir=MODEL_DIR, data_dir=DATA_DIR)
        test_patient = {f: 10.0 for f in selected_features}
        res = predictor.predict(test_patient)
        logger.info(f"Smoke test successful: Prediction = {res['diagnosis']}")
    except ImportError:
        logger.warning("Smoke test skipped: Could not import CancerPredictor (ensure you run from root or src).")
    except Exception as e:
        logger.error(f"Smoke test failed: {e}")

if __name__ == "__main__":
    train_and_save_model()

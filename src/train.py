"""
Training Module — Cancer Prediction Project
==================================================

Standalone script for training the final model. 
This script loads the cleaned data, scales it, trains the model,
and saves the artifacts (model + scaler) to the models/ directory.

Usage:
    python src/train.py
"""

import os
import json
import logging
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from preprocessing import scale_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_and_save_model():
    """Train the final production model and save artifacts."""
    
    # Define paths
    data_path = os.path.join("..", "data", "processed", "breast_cancer_cleaned.csv")
    features_path = os.path.join("..", "data", "processed", "selected_features.json")
    model_dir = os.path.join("..", "models")
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Load data
    logger.info("Loading processed data...")
    df = pd.read_csv(data_path)
    
    # Load selected features
    with open(features_path, 'r') as f:
        selected_features = json.load(f)
        
    X = df[selected_features]
    y = df['diagnosis']
    logger.info(f"Loaded {len(selected_features)} features for {len(X)} samples.")
    
    # For the final production model, we train on ALL available data
    # (We already evaluated the model's generalization in Phase 6-9 using splits/CV)
    logger.info("Scaling features on full dataset...")
    
    # We use our scaler from preprocessing
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )
    
    # Initialize and train model
    # We found Random Forest to be robust during cross-validation
    logger.info("Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_scaled, y)
    
    # Save artifacts
    model_out = os.path.join(model_dir, "cancer_prediction_model.pkl")
    scaler_out = os.path.join(model_dir, "scaler.pkl")
    
    joblib.dump(model, model_out)
    joblib.dump(scaler, scaler_out)
    
    logger.info(f"✅ Model saved to: {model_out}")
    logger.info(f"✅ Scaler saved to: {scaler_out}")
    logger.info("Training pipeline complete.")

if __name__ == "__main__":
    # Ensure script is run from src/ directory or adjust paths
    current_dir = os.path.basename(os.getcwd())
    if current_dir != "src":
        logger.warning("This script expects to be run from the src/ directory.")
        logger.warning("Attempting to adjust paths... (running from root)")
        
        # Simple path adjustment if run from root
        def adjust_path(p): return p.replace("../", "")
        
        data_path = adjust_path("../data/processed/breast_cancer_cleaned.csv")
        features_path = adjust_path("../data/processed/selected_features.json")
        model_dir = adjust_path("../models")
        
        os.makedirs(model_dir, exist_ok=True)
        df = pd.read_csv(data_path)
        with open(features_path, 'r') as f:
            selected_features = json.load(f)
        X = df[selected_features]
        y = df['diagnosis']
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_scaled, y)
        
        joblib.dump(model, os.path.join(model_dir, "cancer_prediction_model.pkl"))
        joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
        
        logger.info("✅ Training complete (Root execution).")
    else:
        train_and_save_model()

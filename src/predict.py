"""
Inference Module
================

Isolates inference logic from the UI so the prediction engine 
can later be reused in APIs, batch prediction scripts, or 
alternative frontends. Encapsulates the ML pipeline (scaler + model) 
and feature medians into a single unified interface.
"""

import os
import json
import logging
from typing import Dict, Any

import pandas as pd
import joblib

from src import config

logger = logging.getLogger(__name__)


class CancerPredictor:
    """
    End-to-end inference pipeline for predicting breast cancer malignancy.
    """
    
    def __init__(self, model_dir: str = config.MODEL_DIR):
        """
        Initialize the predictor by loading all necessary ML artifacts.
        
        Args:
            model_dir: Path to directory containing saved pipeline and medians.
        """
        self.pipeline_path = os.path.join(model_dir, config.PIPELINE_FILE)
        self.medians_path = os.path.join(model_dir, config.MEDIANS_FILE)
        
        self.selected_features = config.SELECTED_FEATURES
        self._load_artifacts()

    def _load_artifacts(self):
        """Load the trained pipeline and feature medians."""
        try:
            self.pipeline = joblib.load(self.pipeline_path)
            
            with open(self.medians_path, 'r') as f:
                self.medians = json.load(f)
                
            logger.info("Successfully loaded ML artifacts (Pipeline and Medians).")
        except FileNotFoundError as e:
            logger.error(f"Missing artifact: {e}. Please run the training script first.")
            raise
        except Exception as e:
            logger.error(f"Failed to load ML artifacts: {e}")
            raise

    def predict(self, patient_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict malignancy from patient feature data.
        
        Args:
            patient_data: Dictionary of feature names and values. Missing 
                          features will be filled with dataset medians.
                          
        Returns:
            Dictionary containing prediction code, text diagnosis, and probabilities.
        """
        # Start with medians as fallback
        full_data = self.medians.copy()
        
        # Update with provided data
        for k, v in patient_data.items():
            if k in full_data and v is not None:
                full_data[k] = float(v)
                
        # Convert to DataFrame ensuring correct column order
        df_new = pd.DataFrame([full_data])[self.selected_features]
        
        # Predict using the full pipeline (scaling is handled internally)
        pred_code = int(self.pipeline.predict(df_new)[0])
        probabilities = self.pipeline.predict_proba(df_new)[0]
        
        diagnosis = config.POSITIVE_LABEL if pred_code == 1 else config.NEGATIVE_LABEL
        confidence = float(probabilities[pred_code] * 100)
        
        return {
            'prediction_code': pred_code,
            'diagnosis': "Malignant" if diagnosis == "M" else "Benign",
            'confidence_pct': round(confidence, 2),
            'probabilities': {
                'Benign': round(float(probabilities[0] * 100), 2),
                'Malignant': round(float(probabilities[1] * 100), 2)
            },
            'inputs_used': {k: float(v) for k, v in full_data.items()}
        }

if __name__ == "__main__":
    # Smoke test to verify module loading and artifact paths
    logging.basicConfig(level=logging.INFO)
    try:
        predictor = CancerPredictor(model_dir=config.MODEL_DIR)
        print("Predictor loaded successfully.")
    except Exception as e:
        print(f"Failed to load predictor: {e}")


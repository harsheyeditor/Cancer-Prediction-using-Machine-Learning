"""
Inference Module - Cancer Prediction Project
==================================================

Reusable class for predicting cancer diagnosis from raw patient data.
This encapsulates the model, scaler, and feature selection into a single
pipeline that can be imported by any UI (Gradio, Flask, FastAPI).

Author: Cancer Prediction ML Project
"""

import os
import json
import logging
from typing import Dict, Any

import pandas as pd
import joblib

logger = logging.getLogger(__name__)


class CancerPredictor:
    """
    End-to-end inference pipeline for predicting breast cancer malignancy.
    """
    
    def __init__(self, model_dir: str = "../models", data_dir: str = "../data"):
        """
        Initialize the predictor by loading all necessary ML artifacts.
        
        Args:
            model_dir: Path to directory containing saved models
            data_dir: Path to directory containing processed data files
        """
        self.model_path = os.path.join(model_dir, "cancer_prediction_model.pkl")
        self.scaler_path = os.path.join(model_dir, "scaler.pkl")
        self.features_path = os.path.join(data_dir, "processed", "selected_features.json")
        self.raw_data_path = os.path.join(data_dir, "raw", "breast_cancer.csv")
        
        self._load_artifacts()
        self._load_medians()

    def _load_artifacts(self):
        """Load the trained model, scaler, and feature list."""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            
            with open(self.features_path, 'r') as f:
                self.selected_features = json.load(f)
                
            logger.info("Successfully loaded ML artifacts.")
        except Exception as e:
            logger.error(f"Failed to load ML artifacts: {e}")
            raise

    def _load_medians(self):
        """Load median values for features not provided by the user."""
        try:
            raw_df = pd.read_csv(self.raw_data_path)
            self.medians = raw_df[self.selected_features].median().to_dict()
        except Exception as e:
            logger.warning(f"Could not load medians from raw data: {e}")
            # Fallback medians (rough estimates)
            self.medians = {feat: 10.0 for feat in self.selected_features}

    def predict(self, patient_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict malignancy from patient feature data.
        
        Args:
            patient_data: Dictionary of feature names and values. Missing 
                          features will be filled with dataset medians.
                          
        Returns:
            Dictionary containing prediction code, text diagnosis, and probabilities.
        """
        # Start with medians
        full_data = self.medians.copy()
        
        # Update with provided data
        for k, v in patient_data.items():
            if k in full_data:
                full_data[k] = v
                
        # Convert to DataFrame
        df_new = pd.DataFrame([full_data])[self.selected_features]
        
        # Scale
        X_scaled = pd.DataFrame(
            self.scaler.transform(df_new),
            columns=df_new.columns
        )
        
        # Predict
        pred_code = int(self.model.predict(X_scaled)[0])
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        diagnosis = "Malignant" if pred_code == 1 else "Benign"
        confidence = float(probabilities[pred_code] * 100)
        
        return {
            'prediction_code': pred_code,
            'diagnosis': diagnosis,
            'confidence_pct': round(confidence, 2),
            'probabilities': {
                'Benign': round(float(probabilities[0] * 100), 2),
                'Malignant': round(float(probabilities[1] * 100), 2)
            },
            'inputs_used': {k: float(v) for k, v in full_data.items()}
        }

# For quick testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    predictor = CancerPredictor(model_dir="models", data_dir="data")
    
    # Test with empty data (will use all medians)
    print("Testing with medians:")
    result = predictor.predict({})
    print(f"Result: {result['diagnosis']} ({result['confidence_pct']}% confidence)")

import unittest
import os
from src.predict import CancerPredictor
from src import config

class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.model_dir = config.MODEL_DIR
        self.pipeline_path = os.path.join(self.model_dir, config.PIPELINE_FILE)
        self.medians_path = os.path.join(self.model_dir, config.MEDIANS_FILE)

    def test_predictor_initialization(self):
        # Only run this test if the model has been trained and artifacts exist
        if os.path.exists(self.pipeline_path) and os.path.exists(self.medians_path):
            predictor = CancerPredictor(model_dir=self.model_dir)
            self.assertIsNotNone(predictor.pipeline)
            self.assertIsNotNone(predictor.medians)
        else:
            self.skipTest("ML artifacts not found. Run train.py first.")

    def test_predict_with_medians(self):
        if os.path.exists(self.pipeline_path) and os.path.exists(self.medians_path):
            predictor = CancerPredictor(model_dir=self.model_dir)
            result = predictor.predict({}) # Provide empty dict to fallback to medians
            
            self.assertIn("diagnosis", result)
            self.assertIn("confidence_pct", result)
            self.assertIn(result["diagnosis"], ["Malignant", "Benign"])
        else:
            self.skipTest("ML artifacts not found. Run train.py first.")

if __name__ == "__main__":
    unittest.main()

import unittest
from src.data_loader import load_from_sklearn

class TestDataLoader(unittest.TestCase):
    def test_load_from_sklearn(self):
        df = load_from_sklearn()
        
        # Check basic properties of the dataset
        self.assertFalse(df.empty)
        self.assertIn("diagnosis", df.columns)
        self.assertIn("id", df.columns)
        
        # Ensure mapping was successful (M and B instead of 0 and 1)
        valid_diagnoses = {"M", "B"}
        actual_diagnoses = set(df["diagnosis"].unique())
        self.assertTrue(actual_diagnoses.issubset(valid_diagnoses))

if __name__ == "__main__":
    unittest.main()

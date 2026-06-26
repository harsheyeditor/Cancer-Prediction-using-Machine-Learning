import unittest
import pandas as pd
from src.preprocessing import drop_unnecessary_columns, handle_missing_values

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "id": [1, 2, 3],
            "Unnamed: 32": [None, None, None],
            "feature_1": [10.0, None, 30.0],
            "diagnosis": ["M", "B", "M"]
        })

    def test_drop_unnecessary_columns(self):
        cleaned_df = drop_unnecessary_columns(self.df)
        self.assertNotIn("id", cleaned_df.columns)
        self.assertNotIn("Unnamed: 32", cleaned_df.columns)
        self.assertIn("feature_1", cleaned_df.columns)

    def test_handle_missing_values(self):
        imputed_df = handle_missing_values(self.df, strategy="median")
        self.assertFalse(imputed_df["feature_1"].isnull().any())
        self.assertEqual(imputed_df["feature_1"].iloc[1], 20.0) # Median of 10 and 30

if __name__ == "__main__":
    unittest.main()

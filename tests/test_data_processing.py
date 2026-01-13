"""Unit tests for data processing functions"""
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDataProcessing(unittest.TestCase):
    """Test cases for data processing module"""

    def setUp(self):
        """Set up test fixtures"""
        self.data_processing_path = "src/data_processing.py"

    def test_data_processing_file_exists(self):
        """Test that data processing module exists"""
        self.assertTrue(
            os.path.exists(self.data_processing_path),
            "data_processing.py not found",
        )

    def test_data_directories_exist(self):
        """Test that data directories exist"""
        data_dirs = ["data/raw", "data/processed"]
        for data_dir in data_dirs:
            self.assertTrue(os.path.exists(data_dir), f"Directory {data_dir} not found")

    def test_feature_engineering_exists(self):
        """Test that feature engineering module exists"""
        feature_eng_path = "src/feature_engineering.py"
        self.assertTrue(
            os.path.exists(feature_eng_path), "feature_engineering.py not found"
        )


if __name__ == "__main__":
    unittest.main()

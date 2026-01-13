"""Unit tests for ML models"""
import unittest
import pickle
import os


class TestModels(unittest.TestCase):
    """Test cases for trained models"""

    def setUp(self):
        """Set up test fixtures"""
        self.models_dir = "models"

    def test_model_files_exist(self):
        """Test that all required model files exist"""
        required_models = [
            "drug_interaction_interaction.pkl",
            "drug_interaction_severity.pkl",
            "advanced_ensemble.pkl",
            "advanced_gradient_boosting.pkl",
            "advanced_neural_network.pkl",
            "advanced_svm.pkl",
        ]
        for model_file in required_models:
            model_path = os.path.join(self.models_dir, model_file)
            self.assertTrue(
                os.path.exists(model_path), f"Model file {model_file} not found"
            )

    def test_models_loadable(self):
        """Test that models can be loaded successfully"""
        model_files = [
            "drug_interaction_interaction.pkl",
            "drug_interaction_severity.pkl",
        ]
        for model_file in model_files:
            model_path = os.path.join(self.models_dir, model_file)
            try:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                self.assertIsNotNone(model)
            except Exception as e:
                self.fail(f"Failed to load model {model_file}: {str(e)}")


if __name__ == "__main__":
    unittest.main()

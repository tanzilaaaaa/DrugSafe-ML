"""Unit tests for Flask web interface"""
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestWebInterface(unittest.TestCase):
    """Test cases for Flask endpoints"""

    def setUp(self):
        """Set up test fixtures"""
        self.app_path = "src/web_interface.py"

    def test_web_interface_file_exists(self):
        """Test that web interface file exists"""
        self.assertTrue(
            os.path.exists(self.app_path), "web_interface.py not found"
        )

    def test_templates_exist(self):
        """Test that required templates exist"""
        template_files = ["templates/index.html"]
        for template in template_files:
            self.assertTrue(os.path.exists(template), f"Template {template} not found")

    def test_static_files_exist(self):
        """Test that static files exist"""
        static_files = ["static/css/style.css", "static/js/main.js"]
        for static_file in static_files:
            self.assertTrue(
                os.path.exists(static_file), f"Static file {static_file} not found"
            )


if __name__ == "__main__":
    unittest.main()

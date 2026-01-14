"""Utility functions for the application"""


def format_drug_name(drug_name):
    """Format drug name for consistency"""
    return drug_name.strip().lower()


def validate_drug_input(drug_name):
    """Validate drug name input"""
    if not drug_name or not isinstance(drug_name, str):
        return False
    return len(drug_name.strip()) > 0


def parse_drug_list(drug_string):
    """Parse comma-separated drug list"""
    if not drug_string:
        return []
    drugs = [drug.strip() for drug in drug_string.split(",")]
    return [drug for drug in drugs if validate_drug_input(drug)]


def format_severity(severity_score):
    """Format severity score to readable level"""
    if severity_score < 0.33:
        return "Low"
    elif severity_score < 0.66:
        return "Moderate"
    else:
        return "High"

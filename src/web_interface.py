"""
Web Interface for Drug Interaction Checker
Flask-based web application with REST API
"""

from flask import Flask, render_template, request, jsonify, send_file, redirect
import json
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing import DrugDataProcessor
from models import DrugInteractionModels
from advanced_models import AdvancedDrugModels
from feature_engineering import AdvancedFeatureEngineer

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.config['SECRET_KEY'] = 'drug-interaction-checker-2024'

# Global variables for models
processor = None
models = None
advanced_models = None
feature_engineer = None
model_loaded = False

def initialize_models():
    """Initialize all models and processors"""
    global processor, models, advanced_models, feature_engineer, model_loaded
    
    try:
        processor = DrugDataProcessor()
        models = DrugInteractionModels()
        advanced_models = AdvancedDrugModels()
        feature_engineer = AdvancedFeatureEngineer()
        
        # Load data and train models if not already done
        df = processor.load_data()
        df = processor.clean_data()
        df = processor.encode_features()
        
        # Train basic models
        X_train, X_test, y_int_train, y_int_test, y_sev_train, y_sev_test = processor.split_data()
        models.train_interaction_models(X_train, y_int_train, X_test, y_int_test)
        models.train_severity_models(X_train, y_sev_train, X_test, y_sev_test)
        
        model_loaded = True
        print("Models initialized successfully")
        
    except Exception as e:
        print(f"Error initializing models: {e}")
        model_loaded = False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_interaction():
    """API endpoint for drug interaction prediction"""
    if not model_loaded:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        data = request.get_json()
        drug1 = data.get('drug1', '').strip()
        drug2 = data.get('drug2', '').strip()
        
        if not drug1 or not drug2:
            return jsonify({'error': 'Both drug names are required'}), 400
        
        # Encode drugs
        drug1_encoded, drug1_class_encoded = processor.encode_drug_for_prediction(drug1)
        drug2_encoded, drug2_class_encoded = processor.encode_drug_for_prediction(drug2)
        
        # Predict interaction
        interaction_pred, interaction_prob = models.predict_interaction(
            drug1_encoded, drug2_encoded, drug1_class_encoded, drug2_class_encoded
        )
        
        result = {
            'drug1': drug1,
            'drug2': drug2,
            'interaction': bool(interaction_pred),
            'interaction_confidence': float(max(interaction_prob)),
            'timestamp': datetime.now().isoformat()
        }
        
        # Predict severity if interaction exists
        if interaction_pred and models.severity_model:
            severity_pred, severity_label, severity_prob = models.predict_severity(
                drug1_encoded, drug2_encoded, drug1_class_encoded, drug2_class_encoded
            )
            
            result.update({
                'severity': severity_label,
                'severity_confidence': float(max(severity_prob))
            })
            
            # Get alternatives for high severity
            if severity_label == 'High':
                result['alternatives'] = {
                    'drug1_alternatives': processor.get_drug_alternatives(drug1),
                    'drug2_alternatives': processor.get_drug_alternatives(drug2)
                }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """API endpoint for batch prediction"""
    if not model_loaded:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        data = request.get_json()
        drug_pairs = data.get('drug_pairs', [])
        
        if not drug_pairs:
            return jsonify({'error': 'Drug pairs list is required'}), 400
        
        results = []
        
        for pair in drug_pairs:
            drug1 = pair.get('drug1', '').strip()
            drug2 = pair.get('drug2', '').strip()
            
            if not drug1 or not drug2:
                results.append({'error': 'Invalid drug pair', 'pair': pair})
                continue
            
            try:
                # Encode and predict
                drug1_encoded, drug1_class_encoded = processor.encode_drug_for_prediction(drug1)
                drug2_encoded, drug2_class_encoded = processor.encode_drug_for_prediction(drug2)
                
                interaction_pred, interaction_prob = models.predict_interaction(
                    drug1_encoded, drug2_encoded, drug1_class_encoded, drug2_class_encoded
                )
                
                result = {
                    'drug1': drug1,
                    'drug2': drug2,
                    'interaction': bool(interaction_pred),
                    'interaction_confidence': float(max(interaction_prob))
                }
                
                if interaction_pred and models.severity_model:
                    severity_pred, severity_label, severity_prob = models.predict_severity(
                        drug1_encoded, drug2_encoded, drug1_class_encoded, drug2_class_encoded
                    )
                    result.update({
                        'severity': severity_label,
                        'severity_confidence': float(max(severity_prob))
                    })
                
                results.append(result)
                
            except Exception as e:
                results.append({'error': str(e), 'pair': pair})
        
        return jsonify({
            'results': results,
            'total_pairs': len(drug_pairs),
            'successful_predictions': len([r for r in results if 'error' not in r]),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_info')
def model_info():
    """Get information about loaded models"""
    if not model_loaded:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        info = {
            'models_loaded': model_loaded,
            'available_models': {
                'interaction_models': list(models.models.get('interaction', {}).keys()) if models.models else [],
                'severity_models': list(models.models.get('severity', {}).keys()) if models.models else []
            },
            'dataset_info': {
                'total_records': len(processor.df) if processor.df is not None else 0,
                'interaction_rate': float(processor.df['interaction'].mean()) if processor.df is not None else 0,
                'unique_drugs': len(set(processor.df['drug1'].tolist() + processor.df['drug2'].tolist())) if processor.df is not None else 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/drug_list')
def get_drug_list():
    """Get list of available drugs"""
    if not model_loaded:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        if processor.df is not None:
            drugs = sorted(set(processor.df['drug1'].tolist() + processor.df['drug2'].tolist()))
            drug_classes = sorted(set(processor.df['drug1_class'].tolist() + processor.df['drug2_class'].tolist()))
            
            return jsonify({
                'drugs': drugs,
                'drug_classes': drug_classes,
                'total_drugs': len(drugs),
                'total_classes': len(drug_classes)
            })
        else:
            return jsonify({'error': 'No data available'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics')
def get_statistics():
    """Get dataset and model statistics"""
    if not model_loaded:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        stats = {}
        
        if processor.df is not None:
            df = processor.df
            
            # Dataset statistics
            stats['dataset'] = {
                'total_records': len(df),
                'interaction_rate': float(df['interaction'].mean()),
                'severity_distribution': df['severity'].value_counts().to_dict(),
                'drug_class_distribution': df['drug1_class'].value_counts().to_dict(),
                'most_common_drugs': df['drug1'].value_counts().head(10).to_dict()
            }
        
        # Model performance
        if models.models:
            stats['model_performance'] = {}
            
            if 'interaction' in models.models:
                interaction_perf = {}
                for model_name, metrics in models.models['interaction'].items():
                    interaction_perf[model_name] = {
                        'accuracy': float(metrics['accuracy']),
                        'f1_score': float(metrics['f1_score']),
                        'precision': float(metrics['precision']),
                        'recall': float(metrics['recall'])
                    }
                stats['model_performance']['interaction'] = interaction_perf
            
            if 'severity' in models.models:
                severity_perf = {}
                for model_name, metrics in models.models['severity'].items():
                    severity_perf[model_name] = {
                        'accuracy': float(metrics['accuracy']),
                        'f1_score': float(metrics['f1_score']),
                        'precision': float(metrics['precision']),
                        'recall': float(metrics['recall'])
                    }
                stats['model_performance']['severity'] = severity_perf
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    """Handle CSV file upload for batch processing"""
    if not model_loaded:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith(('.csv', '.txt')):
            return jsonify({'error': 'Invalid file type. Please upload a CSV or TXT file'}), 400
        
        # Read and parse CSV content
        content = file.read().decode('utf-8')
        lines = content.strip().split('\n')
        
        drug_pairs = []
        invalid_lines = []
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line:
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 2 and parts[0] and parts[1]:
                    drug_pairs.append({
                        'drug1': parts[0],
                        'drug2': parts[1]
                    })
                else:
                    invalid_lines.append(i)
        
        if not drug_pairs:
            return jsonify({'error': 'No valid drug pairs found in the file'}), 400
        
        return jsonify({
            'success': True,
            'drug_pairs': drug_pairs,
            'total_pairs': len(drug_pairs),
            'invalid_lines': invalid_lines,
            'message': f'Successfully parsed {len(drug_pairs)} drug pairs'
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/api/download_sample_csv')
def download_sample_csv():
    """Download a sample CSV file for batch processing"""
    sample_data = [
        'drug1,drug2',
        'Warfarin,Aspirin',
        'Metformin,Lisinopril', 
        'Atorvastatin,Omeprazole',
        'Levothyroxine,Gabapentin',
        'Ibuprofen,Warfarin',
        'Simvastatin,Amlodipine',
        'Hydrochlorothiazide,Losartan'
    ]
    
    # Create CSV content
    csv_content = '\n'.join(sample_data)
    
    # Create response
    response = app.response_class(
        csv_content,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=drug_pairs_sample.csv'}
    )
    
    return response

@app.route('/dashboard')
def dashboard():
    """Dashboard page - redirect to main page for now"""
    return redirect('/')

@app.route('/api')
def api_documentation():
    """API documentation page - return JSON documentation"""
    docs = {
        "title": "Drug Interaction Checker API",
        "version": "1.0",
        "description": "Educational ML-powered drug interaction checker API",
        "endpoints": {
            "/api/predict": {
                "method": "POST",
                "description": "Predict drug interaction between two drugs",
                "parameters": {
                    "drug1": "First drug name (string)",
                    "drug2": "Second drug name (string)"
                },
                "response": {
                    "interaction": "boolean",
                    "interaction_confidence": "float",
                    "severity": "string (if interaction exists)",
                    "severity_confidence": "float (if interaction exists)",
                    "alternatives": "object (if high severity)"
                }
            },
            "/api/batch_predict": {
                "method": "POST", 
                "description": "Batch prediction for multiple drug pairs",
                "parameters": {
                    "drug_pairs": "array of {drug1, drug2} objects"
                }
            },
            "/api/model_info": {
                "method": "GET",
                "description": "Get information about loaded models"
            },
            "/api/drug_list": {
                "method": "GET", 
                "description": "Get list of available drugs"
            },
            "/api/statistics": {
                "method": "GET",
                "description": "Get dataset and model statistics"
            }
        },
        "disclaimer": "This API is for educational purposes only. Do not use for real medical decisions."
    }
    return jsonify(docs)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize models
    print("Initializing models...")
    initialize_models()
    
    # Run the app on port 5001 to avoid conflicts with AirPlay
    print("Starting Flask application on port 5001...")
    app.run(debug=True, host='0.0.0.0', port=5001)
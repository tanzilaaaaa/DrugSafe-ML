"""
Enhanced Main Application for Drug Interaction Checker
Educational ML project with advanced features and comprehensive pipeline
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing import DrugDataProcessor
from models import DrugInteractionModels
from advanced_models import AdvancedDrugModels
from feature_engineering import AdvancedFeatureEngineer
from evaluation import ModelEvaluator

class EnhancedDrugInteractionChecker:
    def __init__(self):
        self.processor = DrugDataProcessor()
        self.models = DrugInteractionModels()
        self.advanced_models = AdvancedDrugModels()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.evaluator = ModelEvaluator()
        self.data_loaded = False
        self.models_trained = False
        self.advanced_trained = False
        
    def load_and_process_data(self, enhanced_features=True):
        """
        Load and preprocess the drug interaction data with enhanced features
        """
        print("=" * 80)
        print("ENHANCED DRUG INTERACTION CHECKER - ADVANCED ML PROJECT")
        print("=" * 80)
        print("\n1. Loading and Processing Data...")
        
        # Load data (creates sample data for educational purposes)
        df = self.processor.load_data()
        
        # Clean data
        df = self.processor.clean_data()
        
        # Enhanced feature engineering
        if enhanced_features:
            print("\n--- Applying Advanced Feature Engineering ---")
            df = self.feature_engineer.feature_engineering_pipeline(df)
        
        # Encode features
        df = self.processor.encode_features()
        
        print(f"\nEnhanced Dataset Info:")
        print(f"- Total records: {len(df)}")
        print(f"- Total features: {len(df.columns)}")
        print(f"- Interaction rate: {df['interaction'].mean():.2%}")
        
        # Display data distribution
        self.evaluator.plot_data_distribution(df)
        
        self.data_loaded = True
        return df
    
    def train_basic_models(self):
        """
        Train basic machine learning models
        """
        if not self.data_loaded:
            raise ValueError("Data must be loaded first")
            
        print("\n2. Training Basic Machine Learning Models...")
        
        # Split data
        X_train, X_test, y_int_train, y_int_test, y_sev_train, y_sev_test = self.processor.split_data()
        
        print(f"\nTraining set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        print(f"Number of features: {X_train.shape[1]}")
        
        # Train interaction prediction models
        print("\n--- Training Interaction Prediction Models ---")
        interaction_results = self.models.train_interaction_models(
            X_train, y_int_train, X_test, y_int_test
        )
        
        # Train severity classification models
        print("\n--- Training Severity Classification Models ---")
        severity_results = self.models.train_severity_models(
            X_train, y_sev_train, X_test, y_sev_test
        )
        
        # Create evaluation summaries
        print("\n--- Basic Model Performance Summary ---")
        self.evaluator.create_evaluation_summary(interaction_results)
        
        if severity_results:
            print("\n--- Severity Classification Summary ---")
            self.evaluator.create_evaluation_summary(severity_results)
        
        # Plot model comparisons
        self.evaluator.plot_model_comparison(interaction_results, 'f1_score')
        
        self.models_trained = True
        return interaction_results, severity_results
    
    def train_advanced_models(self, use_feature_selection=True, balance_data=True):
        """
        Train advanced machine learning models with hyperparameter tuning
        """
        if not self.models_trained:
            raise ValueError("Basic models must be trained first")
        
        print("\n3. Training Advanced Machine Learning Models...")
        
        # Get enhanced features
        X_train, X_test, y_int_train, y_int_test, y_sev_train, y_sev_test = self.processor.split_data()
        
        # Apply feature selection if requested
        if use_feature_selection:
            print("\n--- Applying Feature Selection ---")
            X_train_selected = self.feature_engineer.select_features(X_train, y_int_train, method='selectkbest', k=15)
            X_test_selected = self.feature_engineer.feature_selector.transform(X_test)
            
            # Convert back to DataFrame
            selected_features = self.feature_engineer.selected_features
            X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
            X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
        else:
            X_train_selected, X_test_selected = X_train, X_test
        
        # Balance dataset if requested
        if balance_data:
            print("\n--- Balancing Dataset ---")
            X_train_balanced, y_int_train_balanced = self.feature_engineer.balance_dataset(
                X_train_selected, y_int_train, method='smote'
            )
        else:
            X_train_balanced, y_int_train_balanced = X_train_selected, y_int_train
        
        # Train advanced models
        advanced_results = self.advanced_models.train_advanced_models(
            X_train_balanced, y_int_train_balanced, X_test_selected, y_int_test
        )
        
        # Create ensemble model
        if len(advanced_results) >= 2:
            ensemble_model = self.advanced_models.create_ensemble_model(advanced_results)
            if ensemble_model:
                ensemble_results = self.advanced_models.train_ensemble(
                    X_train_balanced, y_int_train_balanced, X_test_selected, y_int_test
                )
                advanced_results['Ensemble'] = ensemble_results
        
        # Feature importance analysis
        if hasattr(X_train_selected, 'columns'):
            feature_names = X_train_selected.columns.tolist()
            importance_df = self.advanced_models.feature_importance_analysis(feature_names)
        
        # Cross-validation analysis
        cv_results = self.advanced_models.cross_validation_analysis(X_train_balanced, y_int_train_balanced)
        
        self.advanced_trained = True
        return advanced_results
    
    def comprehensive_model_evaluation(self):
        """
        Perform comprehensive evaluation of all models
        """
        print("\n4. Comprehensive Model Evaluation...")
        
        # Compare all models
        all_results = {}
        
        # Add basic models
        if self.models.models and 'interaction' in self.models.models:
            for name, metrics in self.models.models['interaction'].items():
                all_results[f"Basic_{name}"] = metrics
        
        # Add advanced models
        if self.advanced_models.advanced_models:
            for name, metrics in self.advanced_models.advanced_models.items():
                all_results[f"Advanced_{name}"] = metrics
        
        # Create comprehensive comparison
        if all_results:
            print("\n--- Comprehensive Model Comparison ---")
            self.evaluator.create_evaluation_summary(all_results)
            self.evaluator.plot_model_comparison(all_results, 'f1_score', 'plots/comprehensive_model_comparison.png')
        
        return all_results
    
    def predict_drug_interaction(self, drug1, drug2, use_advanced=False):
        """
        Predict interaction between two drugs using basic or advanced models
        """
        if not self.models_trained:
            raise ValueError("Models must be trained first")
        
        if use_advanced and not self.advanced_trained:
            print("Advanced models not available, using basic models")
            use_advanced = False
        
        print(f"\n5. Predicting Interaction: {drug1} + {drug2}")
        print(f"Using {'Advanced' if use_advanced else 'Basic'} Models")
        print("-" * 60)
        
        try:
            # Get encoded features using proper encoding
            drug1_encoded, drug1_class_encoded = self.processor.encode_drug_for_prediction(drug1)
            drug2_encoded, drug2_class_encoded = self.processor.encode_drug_for_prediction(drug2)
            
            # Choose model to use
            if use_advanced and self.advanced_models.ensemble_model:
                # Use ensemble model
                features = np.array([[drug1_encoded, drug2_encoded, drug1_class_encoded, drug2_class_encoded]])
                interaction_pred = self.advanced_models.ensemble_model.predict(features)[0]
                interaction_prob = self.advanced_models.ensemble_model.predict_proba(features)[0]
                model_used = "Advanced Ensemble"
            else:
                # Use basic model
                interaction_pred, interaction_prob = self.models.predict_interaction(
                    drug1_encoded, drug2_encoded, drug1_class_encoded, drug2_class_encoded
                )
                model_used = "Basic Decision Tree"
            
            print(f"Model Used: {model_used}")
            print(f"Interaction Prediction: {'YES' if interaction_pred else 'NO'}")
            print(f"Confidence: {max(interaction_prob):.2%}")
            
            # If interaction predicted, classify severity
            if interaction_pred and self.models.severity_model:
                severity_pred, severity_label, severity_prob = self.models.predict_severity(
                    drug1_encoded, drug2_encoded, drug1_class_encoded, drug2_class_encoded
                )
                
                print(f"Severity Level: {severity_label}")
                print(f"Severity Confidence: {max(severity_prob):.2%}")
                
                # Suggest alternatives for high severity
                if severity_label == 'High':
                    print(f"\n⚠️  HIGH SEVERITY INTERACTION DETECTED!")
                    print("Suggested alternatives:")
                    alt1 = self.processor.get_drug_alternatives(drug1)
                    alt2 = self.processor.get_drug_alternatives(drug2)
                    print(f"- Instead of {drug1}: {', '.join(alt1)}")
                    print(f"- Instead of {drug2}: {', '.join(alt2)}")
            
            return {
                'interaction': bool(interaction_pred),
                'interaction_confidence': float(max(interaction_prob)),
                'severity': severity_label if interaction_pred else 'None',
                'severity_confidence': float(max(severity_prob)) if interaction_pred else 0.0,
                'model_used': model_used
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
    
    def run_comprehensive_demo(self):
        """
        Run comprehensive demo with all features
        """
        try:
            # Load and process data with enhanced features
            df = self.load_and_process_data(enhanced_features=True)
            
            # Train basic models
            basic_interaction_results, basic_severity_results = self.train_basic_models()
            
            # Train advanced models
            advanced_results = self.train_advanced_models(use_feature_selection=True, balance_data=True)
            
            # Comprehensive evaluation
            all_results = self.comprehensive_model_evaluation()
            
            # Save models
            print("\n--- Saving Models ---")
            self.models.save_models()
            self.advanced_models.save_advanced_models()
            
            # Demo predictions with both basic and advanced models
            print("\n" + "=" * 80)
            print("COMPREHENSIVE DEMO PREDICTIONS")
            print("=" * 80)
            
            test_pairs = [
                ('Warfarin', 'Aspirin'),
                ('Metformin', 'Lisinopril'),
                ('Atorvastatin', 'Omeprazole'),
                ('Levothyroxine', 'Gabapentin'),
                ('Ibuprofen', 'Warfarin')
            ]
            
            for drug1, drug2 in test_pairs:
                print(f"\n{'='*60}")
                print(f"Testing: {drug1} + {drug2}")
                print(f"{'='*60}")
                
                # Basic model prediction
                basic_result = self.predict_drug_interaction(drug1, drug2, use_advanced=False)
                
                # Advanced model prediction
                advanced_result = self.predict_drug_interaction(drug1, drug2, use_advanced=True)
                
                # Compare results
                if basic_result and advanced_result:
                    print(f"\n📊 Model Comparison:")
                    print(f"Basic Model - Interaction: {basic_result['interaction']}, Confidence: {basic_result['interaction_confidence']:.2%}")
                    print(f"Advanced Model - Interaction: {advanced_result['interaction']}, Confidence: {advanced_result['interaction_confidence']:.2%}")
                    
                    if basic_result['interaction'] != advanced_result['interaction']:
                        print("⚠️  Models disagree on interaction prediction!")
                
                print()
            
            # Generate final report
            self.generate_project_report(all_results)
            
            print("\n" + "=" * 80)
            print("⚠️  EDUCATIONAL DISCLAIMER")
            print("=" * 80)
            print("This enhanced project is for educational purposes only.")
            print("Features include:")
            print("- Advanced ML models (SVM, Neural Networks, Ensemble)")
            print("- Feature engineering and selection")
            print("- Hyperparameter tuning with cross-validation")
            print("- Data balancing techniques")
            print("- Comprehensive model evaluation")
            print("- Web interface with REST API")
            print("\nDO NOT use for real medical decisions.")
            print("Always consult healthcare professionals.")
            print("=" * 80)
            
        except Exception as e:
            print(f"Error running comprehensive demo: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_project_report(self, all_results):
        """
        Generate comprehensive project report
        """
        print("\n" + "=" * 80)
        print("PROJECT COMPLETION REPORT")
        print("=" * 80)
        
        # Find best performing model
        if all_results:
            best_model = max(all_results.items(), key=lambda x: x[1]['f1_score'])
            
            print(f"\n🏆 Best Performing Model: {best_model[0]}")
            print(f"   - F1 Score: {best_model[1]['f1_score']:.4f}")
            print(f"   - Accuracy: {best_model[1]['accuracy']:.4f}")
            print(f"   - Precision: {best_model[1]['precision']:.4f}")
            print(f"   - Recall: {best_model[1]['recall']:.4f}")
        
        # Project statistics
        print(f"\n📈 Project Statistics:")
        print(f"   - Total Models Trained: {len(all_results) if all_results else 0}")
        print(f"   - Dataset Size: {len(self.processor.df) if self.processor.df is not None else 0}")
        print(f"   - Features Created: {len(self.processor.df.columns) if self.processor.df is not None else 0}")
        print(f"   - Plots Generated: Multiple visualization files in plots/")
        
        # Technical features implemented
        print(f"\n🔧 Technical Features Implemented:")
        features = [
            "✅ Basic ML Models (Logistic Regression, Decision Tree, Random Forest, Naive Bayes)",
            "✅ Advanced ML Models (SVM, Neural Networks, Gradient Boosting, AdaBoost)",
            "✅ Ensemble Methods (Voting Classifier)",
            "✅ Feature Engineering (Interaction features, embeddings, statistical features)",
            "✅ Feature Selection (SelectKBest, RFE)",
            "✅ Data Balancing (SMOTE, Under-sampling)",
            "✅ Hyperparameter Tuning (Grid Search with Cross-Validation)",
            "✅ Comprehensive Model Evaluation",
            "✅ Web Interface with REST API",
            "✅ Interactive Dashboard",
            "✅ Batch Processing Capabilities",
            "✅ Model Persistence and Loading",
            "✅ Comprehensive Documentation"
        ]
        
        for feature in features:
            print(f"   {feature}")
        
        print(f"\n🎯 Project Objectives Met:")
        objectives = [
            "✅ Drug-drug interaction detection",
            "✅ Severity classification",
            "✅ Alternative drug suggestions",
            "✅ Multiple ML algorithms implementation",
            "✅ Model comparison and evaluation",
            "✅ Feature engineering pipeline",
            "✅ Web-based user interface",
            "✅ API for integration",
            "✅ Educational documentation",
            "✅ Real-world applicability (with disclaimers)"
        ]
        
        for objective in objectives:
            print(f"   {objective}")
        
        print(f"\n📁 Generated Files:")
        files = [
            "- Source code: src/ directory with modular architecture",
            "- Models: models/ directory with trained model files",
            "- Visualizations: plots/ directory with analysis charts",
            "- Web interface: templates/ and static/ directories",
            "- Documentation: README.md, requirements.txt, LICENSE",
            "- Jupyter notebook: notebooks/eda_analysis.ipynb"
        ]
        
        for file_info in files:
            print(f"   {file_info}")
        
        print(f"\n🚀 How to Use:")
        usage = [
            "1. Command Line: python3 src/main.py",
            "2. Web Interface: python3 src/web_interface.py (then visit http://localhost:5000)",
            "3. Jupyter Notebook: jupyter notebook notebooks/eda_analysis.ipynb",
            "4. API Integration: Use REST endpoints for programmatic access"
        ]
        
        for usage_info in usage:
            print(f"   {usage_info}")

def main():
    """
    Main function to run the enhanced drug interaction checker
    """
    checker = EnhancedDrugInteractionChecker()
    
    # Run the comprehensive demo
    checker.run_comprehensive_demo()

if __name__ == "__main__":
    main()
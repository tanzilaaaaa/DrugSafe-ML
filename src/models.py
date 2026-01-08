"""
Machine Learning Models for Drug Interaction Checker
Implements various ML algorithms for interaction prediction and severity classification
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

class DrugInteractionModels:
    def __init__(self):
        self.interaction_model = None
        self.severity_model = None
        self.models = {}
        
    def train_interaction_models(self, X_train, y_train, X_test, y_test):
        """
        Train multiple models for interaction prediction
        """
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Naive Bayes': GaussianNB()
        }
        
        results = {}
        
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            results[name] = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            print(f"\n{name} Results:")
            print(f"Accuracy: {results[name]['accuracy']:.4f}")
            print(f"Precision: {results[name]['precision']:.4f}")
            print(f"Recall: {results[name]['recall']:.4f}")
            print(f"F1-Score: {results[name]['f1_score']:.4f}")
        
        # Select best model based on F1-score
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        self.interaction_model = results[best_model_name]['model']
        
        print(f"\nBest model for interaction prediction: {best_model_name}")
        
        self.models['interaction'] = results
        return results
    
    def train_severity_models(self, X_train, y_train, X_test, y_test):
        """
        Train models for severity classification (only for interacting drugs)
        """
        # Filter data for only interacting drugs (severity > 0)
        interaction_mask_train = y_train > 0
        interaction_mask_test = y_test > 0
        
        if not any(interaction_mask_train) or not any(interaction_mask_test):
            print("No interacting drugs found for severity classification")
            return {}
        
        X_train_int = X_train[interaction_mask_train]
        y_train_int = y_train[interaction_mask_train]
        X_test_int = X_test[interaction_mask_test]
        y_test_int = y_test[interaction_mask_test]
        
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=8),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=50),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                # Train model
                model.fit(X_train_int, y_train_int)
                
                # Make predictions
                y_pred = model.predict(X_test_int)
                
                # Calculate metrics
                results[name] = {
                    'model': model,
                    'accuracy': accuracy_score(y_test_int, y_pred),
                    'precision': precision_score(y_test_int, y_pred, average='weighted'),
                    'recall': recall_score(y_test_int, y_pred, average='weighted'),
                    'f1_score': f1_score(y_test_int, y_pred, average='weighted'),
                    'confusion_matrix': confusion_matrix(y_test_int, y_pred)
                }
                
                print(f"\n{name} Severity Results:")
                print(f"Accuracy: {results[name]['accuracy']:.4f}")
                print(f"Precision: {results[name]['precision']:.4f}")
                print(f"Recall: {results[name]['recall']:.4f}")
                print(f"F1-Score: {results[name]['f1_score']:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        if results:
            # Select best model based on F1-score
            best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
            self.severity_model = results[best_model_name]['model']
            print(f"\nBest model for severity classification: {best_model_name}")
        
        self.models['severity'] = results
        return results
    
    def predict_interaction(self, drug1_encoded, drug2_encoded, drug1_class_encoded, drug2_class_encoded):
        """
        Predict if two drugs interact
        """
        if self.interaction_model is None:
            raise ValueError("Interaction model not trained yet")
        
        # Combine features in the same order as training
        features = np.array([[drug1_encoded, drug2_encoded, drug1_class_encoded, drug2_class_encoded]])
        
        # Predict interaction
        interaction_prob = self.interaction_model.predict_proba(features)[0]
        interaction_pred = self.interaction_model.predict(features)[0]
        
        return interaction_pred, interaction_prob
    
    def predict_severity(self, drug1_encoded, drug2_encoded, drug1_class_encoded, drug2_class_encoded):
        """
        Predict severity of drug interaction
        """
        if self.severity_model is None:
            raise ValueError("Severity model not trained yet")
        
        # Combine features in the same order as training
        features = np.array([[drug1_encoded, drug2_encoded, drug1_class_encoded, drug2_class_encoded]])
        
        # Predict severity
        severity_prob = self.severity_model.predict_proba(features)[0]
        severity_pred = self.severity_model.predict(features)[0]
        
        severity_map = {0: 'None', 1: 'Low', 2: 'Moderate', 3: 'High'}
        severity_label = severity_map.get(severity_pred, 'Unknown')
        
        return severity_pred, severity_label, severity_prob
    
    def save_models(self, filepath_prefix='models/drug_interaction'):
        """
        Save trained models to disk
        """
        if self.interaction_model:
            joblib.dump(self.interaction_model, f'{filepath_prefix}_interaction.pkl')
        if self.severity_model:
            joblib.dump(self.severity_model, f'{filepath_prefix}_severity.pkl')
        print("Models saved successfully")
    
    def load_models(self, filepath_prefix='models/drug_interaction'):
        """
        Load trained models from disk
        """
        try:
            self.interaction_model = joblib.load(f'{filepath_prefix}_interaction.pkl')
            self.severity_model = joblib.load(f'{filepath_prefix}_severity.pkl')
            print("Models loaded successfully")
        except FileNotFoundError:
            print("Model files not found. Please train models first.")
    
    def get_feature_importance(self, model_type='interaction'):
        """
        Get feature importance for tree-based models
        """
        model = self.interaction_model if model_type == 'interaction' else self.severity_model
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        else:
            print(f"Feature importance not available for {type(model).__name__}")
            return None
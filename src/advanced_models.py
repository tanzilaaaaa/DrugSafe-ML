"""
Advanced Machine Learning Models for Drug Interaction Checker
Implements ensemble methods, neural networks, and advanced techniques
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

class AdvancedDrugModels:
    def __init__(self):
        self.advanced_models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.best_params = {}
        
    def create_advanced_models(self):
        """
        Create advanced ML models with hyperparameter tuning
        """
        models = {
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Neural Network': MLPClassifier(random_state=42, max_iter=1000),
            'AdaBoost': AdaBoostClassifier(random_state=42)
        }
        
        # Hyperparameter grids
        param_grids = {
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'Neural Network': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01]
            },
            'AdaBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0]
            }
        }
        
        return models, param_grids
    
    def train_advanced_models(self, X_train, y_train, X_test, y_test, cv_folds=5):
        """
        Train advanced models with hyperparameter tuning
        """
        models, param_grids = self.create_advanced_models()
        results = {}
        
        print("\n--- Training Advanced Models with Hyperparameter Tuning ---")
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Create pipeline with scaling for neural networks and SVM
            if name in ['Neural Network', 'SVM']:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])
                # Adjust parameter names for pipeline
                param_grid = {f'model__{k}': v for k, v in param_grids[name].items()}
            else:
                pipeline = model
                param_grid = param_grids[name]
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv_folds, 
                scoring='f1_weighted', n_jobs=-1, verbose=0
            )
            
            try:
                grid_search.fit(X_train, y_train)
                
                # Best model predictions
                y_pred = grid_search.predict(X_test)
                y_pred_proba = grid_search.predict_proba(X_test)
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                results[name] = {
                    'model': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1_score': f1_score(y_test, y_pred, average='weighted'),
                    'cv_score': grid_search.best_score_,
                    'roc_auc': roc_auc_score(y_test, y_pred_proba[:, 1]) if len(np.unique(y_test)) == 2 else None
                }
                
                print(f"{name} - Best CV F1: {grid_search.best_score_:.4f}")
                print(f"{name} - Test F1: {results[name]['f1_score']:.4f}")
                
                self.best_params[name] = grid_search.best_params_
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        self.advanced_models = results
        return results
    
    def create_ensemble_model(self, base_models):
        """
        Create ensemble model using voting classifier
        """
        if len(base_models) < 2:
            print("Need at least 2 models for ensemble")
            return None
        
        # Select top 3 models based on F1 score
        sorted_models = sorted(base_models.items(), 
                             key=lambda x: x[1]['f1_score'], reverse=True)[:3]
        
        estimators = [(name, model_info['model']) for name, model_info in sorted_models]
        
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use probability-based voting
        )
        
        print(f"\nCreated ensemble with models: {[name for name, _ in estimators]}")
        return self.ensemble_model
    
    def train_ensemble(self, X_train, y_train, X_test, y_test):
        """
        Train the ensemble model
        """
        if self.ensemble_model is None:
            print("No ensemble model created")
            return None
        
        print("\n--- Training Ensemble Model ---")
        self.ensemble_model.fit(X_train, y_train)
        
        # Evaluate ensemble
        y_pred = self.ensemble_model.predict(X_test)
        y_pred_proba = self.ensemble_model.predict_proba(X_test)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        ensemble_results = {
            'model': self.ensemble_model,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba[:, 1]) if len(np.unique(y_test)) == 2 else None
        }
        
        print(f"Ensemble F1 Score: {ensemble_results['f1_score']:.4f}")
        return ensemble_results
    
    def feature_importance_analysis(self, feature_names):
        """
        Analyze feature importance across models
        """
        importance_data = {}
        
        for name, model_info in self.advanced_models.items():
            model = model_info['model']
            
            # Extract feature importance
            if hasattr(model, 'feature_importances_'):
                importance_data[name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficients
                importance_data[name] = np.abs(model.coef_[0])
            elif hasattr(model, 'named_steps') and hasattr(model.named_steps['model'], 'feature_importances_'):
                # For pipelines
                importance_data[name] = model.named_steps['model'].feature_importances_
        
        # Create DataFrame for analysis
        if importance_data:
            importance_df = pd.DataFrame(importance_data, index=feature_names)
            importance_df['Average'] = importance_df.mean(axis=1)
            importance_df = importance_df.sort_values('Average', ascending=False)
            
            print("\n--- Feature Importance Analysis ---")
            print(importance_df.round(4))
            
            return importance_df
        
        return None
    
    def cross_validation_analysis(self, X, y, cv_folds=5):
        """
        Perform cross-validation analysis on all models
        """
        cv_results = {}
        
        print(f"\n--- {cv_folds}-Fold Cross-Validation Analysis ---")
        
        for name, model_info in self.advanced_models.items():
            model = model_info['model']
            
            try:
                cv_scores = cross_val_score(model, X, y, cv=cv_folds, 
                                          scoring='f1_weighted', n_jobs=-1)
                
                cv_results[name] = {
                    'mean_score': cv_scores.mean(),
                    'std_score': cv_scores.std(),
                    'scores': cv_scores
                }
                
                print(f"{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                print(f"Error in CV for {name}: {e}")
                continue
        
        return cv_results
    
    def save_advanced_models(self, filepath_prefix='models/advanced'):
        """
        Save all advanced models
        """
        for name, model_info in self.advanced_models.items():
            safe_name = name.replace(' ', '_').lower()
            joblib.dump(model_info['model'], f'{filepath_prefix}_{safe_name}.pkl')
        
        if self.ensemble_model:
            joblib.dump(self.ensemble_model, f'{filepath_prefix}_ensemble.pkl')
        
        # Save best parameters
        joblib.dump(self.best_params, f'{filepath_prefix}_best_params.pkl')
        
        print("Advanced models saved successfully")
    
    def load_advanced_models(self, filepath_prefix='models/advanced'):
        """
        Load advanced models
        """
        try:
            self.best_params = joblib.load(f'{filepath_prefix}_best_params.pkl')
            self.ensemble_model = joblib.load(f'{filepath_prefix}_ensemble.pkl')
            print("Advanced models loaded successfully")
        except FileNotFoundError:
            print("Advanced model files not found. Please train models first.")
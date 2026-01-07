"""
Model Evaluation and Visualization Module
Provides comprehensive evaluation metrics and visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, title="Confusion Matrix"):
        """
        Plot confusion matrix heatmap
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
        
        return cm
    
    def plot_model_comparison(self, results_dict, metric='f1_score', save_path='plots/model_comparison.png'):
        """
        Compare multiple models using bar plot
        """
        models = list(results_dict.keys())
        scores = [results_dict[model][metric] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Model comparison plot saved to {save_path}")
    
    def plot_roc_curve(self, y_true, y_pred_proba, title="ROC Curve"):
        """
        Plot ROC curve for binary classification
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
        
        return roc_auc
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, title="Precision-Recall Curve"):
        """
        Plot Precision-Recall curve
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
        avg_precision = average_precision_score(y_true, y_pred_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.show()
        
        return avg_precision
    
    def generate_classification_report(self, y_true, y_pred, target_names=None):
        """
        Generate detailed classification report
        """
        report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
        
        # Convert to DataFrame for better visualization
        df_report = pd.DataFrame(report).transpose()
        
        print("Classification Report:")
        print("=" * 50)
        print(classification_report(y_true, y_pred, target_names=target_names))
        
        return df_report
    
    def plot_feature_importance(self, feature_importance, feature_names, title="Feature Importance"):
        """
        Plot feature importance
        """
        if feature_importance is None:
            print("Feature importance not available")
            return
        
        # Sort features by importance
        indices = np.argsort(feature_importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.bar(range(len(feature_importance)), feature_importance[indices])
        plt.xticks(range(len(feature_importance)), 
                  [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()
    
    def evaluate_model_performance(self, model, X_test, y_test, model_name="Model"):
        """
        Comprehensive model evaluation
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Generate classification report
        print(f"\n{model_name} Performance:")
        print("=" * 50)
        self.generate_classification_report(y_test, y_pred)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred, title=f"{model_name} - Confusion Matrix")
        
        # Plot ROC curve for binary classification
        if y_pred_proba is not None and len(np.unique(y_test)) == 2:
            self.plot_roc_curve(y_test, y_pred_proba, title=f"{model_name} - ROC Curve")
            self.plot_precision_recall_curve(y_test, y_pred_proba, 
                                           title=f"{model_name} - Precision-Recall Curve")
        
        # Feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            feature_names = [f'Feature_{i}' for i in range(len(model.feature_importances_))]
            self.plot_feature_importance(model.feature_importances_, feature_names,
                                       title=f"{model_name} - Feature Importance")
    
    def create_evaluation_summary(self, results_dict):
        """
        Create summary table of all model results
        """
        summary_data = []
        
        for model_name, metrics in results_dict.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        print("\nModel Performance Summary:")
        print("=" * 60)
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    def save_evaluation_plots(self, output_dir='plots/'):
        """
        Save all evaluation plots to directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        print(f"Plots saved to {output_dir}")
        
    def plot_data_distribution(self, df, target_col='interaction', save_path='plots/data_distribution.png'):
        """
        Plot distribution of target variable
        """
        plt.figure(figsize=(12, 4))
        
        # Target distribution
        plt.subplot(1, 2, 1)
        df[target_col].value_counts().plot(kind='bar', color=['lightblue', 'lightcoral'])
        plt.title(f'Distribution of {target_col.title()}')
        plt.xlabel(target_col.title())
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        
        # Severity distribution (if available)
        if 'severity' in df.columns:
            plt.subplot(1, 2, 2)
            df['severity'].value_counts().plot(kind='bar', 
                                             color=['green', 'yellow', 'orange', 'red'])
            plt.title('Distribution of Severity')
            plt.xlabel('Severity Level')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Data distribution plot saved to {save_path}")
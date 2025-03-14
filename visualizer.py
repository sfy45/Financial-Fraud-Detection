import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve
import shap

class Visualizer:
    def __init__(self):
        pass
    
    def plot_class_distribution(self, df, target_col='Class'):
        """Plot the distribution of fraud vs non-fraud transactions"""
        plt.figure(figsize=(10, 6))
        sns.countplot(x=target_col, data=df)
        plt.title('Class Distribution (Fraud vs Non-Fraud)')
        plt.xlabel('Class (0: Normal, 1: Fraud)')
        plt.ylabel('Count')
        
        # Add percentage labels
        total = len(df)
        for p in plt.gca().patches:
            height = p.get_height()
            plt.text(p.get_x() + p.get_width()/2.,
                    height + 3,
                    '{:.2f}%'.format(100 * height/total),
                    ha="center") 
        
        return plt
    
    def plot_feature_distributions(self, df, target_col='Class', n_features=5):
        """Plot distributions of top features by class"""
        # Select numerical columns only
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        num_cols = [col for col in num_cols if col != target_col]
        
        # If there are too many features, select a subset
        if len(num_cols) > n_features:
            num_cols = num_cols[:n_features]
        
        # Create subplots
        fig, axes = plt.subplots(len(num_cols), 1, figsize=(12, 4*len(num_cols)))
        
        # If there's only one feature, axes won't be an array
        if len(num_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(num_cols):
            sns.histplot(data=df, x=col, hue=target_col, bins=50, ax=axes[i], kde=True)
            axes[i].set_title(f'Distribution of {col} by Class')
            
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, df, target_col='Class'):
        """Plot correlation matrix of features"""
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                   linewidths=0.5, vmin=-1, vmax=1)
        plt.title('Feature Correlation Matrix')
        
        return plt
    
    def plot_feature_importance(self, model, feature_names, model_name="Model"):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            # Get feature importances
            importances = model.feature_importances_
            
            # Sort feature importances in descending order
            indices = np.argsort(importances)[::-1]
            
            # Rearrange feature names so they match the sorted feature importances
            names = [feature_names[i] for i in indices]
            
            # Create plot
            plt.figure(figsize=(12, 8))
            plt.title(f"Feature Importance ({model_name})")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), names, rotation=90)
            plt.tight_layout()
            
            return plt
        else:
            print(f"Model {model_name} doesn't have feature_importances_ attribute")
            return None
    
    def plot_roc_curve(self, models_results):
        """Plot ROC curves for multiple models"""
        plt.figure(figsize=(10, 8))
        
        for result in models_results:
            model_name = result['model_name']
            y_test = result['y_test']
            y_pred_proba = result['y_pred_proba']
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = result['auc']
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='best')
        
        return plt
    
    def plot_precision_recall_curve(self, models_results):
        """Plot Precision-Recall curves for multiple models"""
        plt.figure(figsize=(10, 8))
        
        for result in models_results:
            model_name = result['model_name']
            y_test = result['y_test']
            y_pred_proba = result['y_pred_proba']
            
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            
            plt.plot(recall, precision, label=f'{model_name}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        
        return plt
    
    def plot_confusion_matrix(self, cm, model_name="Model"):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        return plt
    
    def plot_shap_values(self, model, X_test, feature_names, model_name="Model"):
        """Plot SHAP values to explain model predictions"""
        # Create explainer
        if model_name == "XGBoost":
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_test)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names)
        
        return plt
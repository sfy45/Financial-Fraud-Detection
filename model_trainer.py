# utils/model_trainer.py (updated)
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
            'XGBoost': XGBClassifier(scale_pos_weight=10, n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        }
        self.neural_net = None
    
    def train_models(self, X_train, y_train):
        """Train multiple machine learning models"""
        trained_models = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
        
        return trained_models
    
    def train_neural_network(self, X_train, y_train, input_dim):
        """Train a neural network model"""
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=64,
            validation_split=0.2,
            verbose=1
        )
        
        self.neural_net = model
        return model, history
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Evaluate model performance with various metrics"""
        if model_name == "Neural Network":
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Detailed classification report
        report = classification_report(y_test, y_pred)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba
        }
        
        return results
    
    def save_model(self, model, file_path):
        """Save the trained model to a file"""
        if isinstance(model, Sequential):
            model.save(file_path)
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
    
    def load_model(self, file_path, model_type='sklearn'):
        """Load a trained model from a file"""
        if model_type == 'keras':
            from tensorflow.keras.models import load_model
            return load_model(file_path)
        else:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
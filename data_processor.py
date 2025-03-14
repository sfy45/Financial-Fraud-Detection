import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn import __version__ as sklearn_version
from packaging import version

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
        # Handle different scikit-learn versions
        if version.parse(sklearn_version) >= version.parse('1.2.0'):
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        else:
            self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
    def load_data(self, file_path):
        """Load the dataset from a CSV file"""
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, df, target_col='Class'):
        """Preprocess the data for model training"""
        # Handle missing values
        df = df.fillna(df.mean())
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        num_features = X.select_dtypes(include=['int64', 'float64']).columns
        
        # Get categorical features if any
        cat_features = X.select_dtypes(include=['object', 'category']).columns
        
        # Create preprocessing pipelines
        if version.parse(sklearn_version) >= version.parse('1.2.0'):
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), num_features),
                    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_features)
                ] if len(cat_features) > 0 else [
                    ('num', StandardScaler(), num_features)
                ]
            )
        else:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), num_features),
                    ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), cat_features)
                ] if len(cat_features) > 0 else [
                    ('num', StandardScaler(), num_features)
                ]
            )
        
        # Fit and transform the training data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
        
        return X_train_resampled, X_test_processed, y_train_resampled, y_test, preprocessor
    
    def engineer_features(self, df):
        """Create new features for fraud detection"""
        # Copy the dataframe to avoid modifying the original
        df_new = df.copy()
        
        # If Time column exists, create time-based features
        if 'Time' in df_new.columns:
            # Convert seconds to hours of the day (assuming Time is in seconds from a reference point)
            df_new['Hour'] = (df_new['Time'] / 3600) % 24
            
            # Flag for transactions during odd hours (midnight to 5 AM)
            df_new['Odd_Hour'] = ((df_new['Hour'] >= 0) & (df_new['Hour'] < 5)).astype(int)
        
        # If Amount column exists, create amount-based features
        if 'Amount' in df_new.columns:
            # Log transform for amount (to handle skewed distribution)
            df_new['Log_Amount'] = np.log1p(df_new['Amount'])
            
            # Flag for high-value transactions (top 5%)
            threshold = df_new['Amount'].quantile(0.95)
            df_new['High_Value'] = (df_new['Amount'] > threshold).astype(int)
        
        # Transaction frequency features (if multiple transactions per account)
        if 'card_id' in df_new.columns:  # Assuming there's a card or account ID
            # Number of transactions per card
            tx_count = df_new.groupby('card_id').size().reset_index(name='Tx_Count')
            df_new = df_new.merge(tx_count, on='card_id', how='left')
            
            # Average transaction amount per card
            avg_amount = df_new.groupby('card_id')['Amount'].mean().reset_index(name='Avg_Amount')
            df_new = df_new.merge(avg_amount, on='card_id', how='left')
            
            # Transaction amount deviation from average
            df_new['Amount_Deviation'] = df_new['Amount'] - df_new['Avg_Amount']
        
        return df_new
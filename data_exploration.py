# pages/data_exploration.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils.data_processor import DataProcessor
from utils.visualizer import Visualizer

def app():
    st.title("Data Exploration")
    
    # Initialize classes
    data_processor = DataProcessor()
    visualizer = Visualizer()
    
    # Load data function
    @st.cache_data
    def load_data():
        # Check if data exists in the data directory
        data_path = "data/creditcard.csv"
        if os.path.exists(data_path):
            return pd.read_csv(data_path)
        else:
            st.warning("Default dataset not found. Please upload a dataset.")
            return None
    
    # Load data
    df = load_data()
    if df is None:
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df.to_csv("data/uploaded_data.csv", index=False)
    
    if df is not None:
        st.write(f"Dataset shape: {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Data overview
        st.header("Data Overview")
        st.write(df.head())
        
        # Data information
        st.header("Data Information")
        buffer = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.write(buffer)
        
        # Statistical summary
        st.header("Statistical Summary")
        st.write(df.describe())
        
        # Class distribution
        st.header("Class Distribution")
        if 'Class' in df.columns:
            fig = visualizer.plot_class_distribution(df)
            st.pyplot(fig)
            
            # Calculate fraud percentage
            fraud_percentage = df['Class'].mean() * 100
            st.write(f"Fraud transactions: {fraud_percentage:.2f}% of the dataset")
        else:
            st.warning("No 'Class' column found in the dataset. Please ensure your target variable is named 'Class'.")
        
        # Feature distributions
        st.header("Feature Distributions")
        num_features = st.slider("Number of features to display", 1, min(10, len(df.columns)-1), 5)
        fig = visualizer.plot_feature_distributions(df, n_features=num_features)
        st.pyplot(fig)
        
        # Correlation matrix
        st.header("Correlation Matrix")
        fig = visualizer.plot_correlation_matrix(df)
        st.pyplot(fig)
        
        # Transaction amount analysis
        if 'Amount' in df.columns:
            st.header("Transaction Amount Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Amount Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=df, x='Amount', bins=50, kde=True, ax=ax)
                st.pyplot(fig)
            
            with col2:
                if 'Class' in df.columns:
                    st.subheader("Amount by Class")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(x='Class', y='Amount', data=df, ax=ax)
                    st.pyplot(fig)
        
        # Time analysis
        if 'Time' in df.columns:
            st.header("Transaction Time Analysis")
            
            # Convert time to hours
            df_time = df.copy()
            df_time['Hour'] = (df_time['Time'] / 3600) % 24
            
            fig, ax = plt.subplots(figsize=(12, 6))
            if 'Class' in df.columns:
                sns.histplot(data=df_time, x='Hour', hue='Class', bins=24, kde=True, ax=ax)
            else:
                sns.histplot(data=df_time, x='Hour', bins=24, kde=True, ax=ax)
            plt.title('Transaction Distribution by Hour of Day')
            plt.xlabel('Hour of Day')
            plt.ylabel('Number of Transactions')
            st.pyplot(fig)
            
        # Feature analysis for fraud detection
        if 'Class' in df.columns:
            st.header("Feature Analysis for Fraud Detection")
            
            # Select top features correlated with fraud
            corr_with_fraud = df.corr()['Class'].sort_values(ascending=False)
            top_features = corr_with_fraud[1:6].index.tolist()  # Skip Class itself
            
            st.subheader("Top Features Correlated with Fraud")
            st.write(corr_with_fraud[1:11])  # Show top 10 correlations
            
            # Plot distributions of top features by fraud/non-fraud
            st.subheader("Distributions of Top Features by Class")
            for feature in top_features:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=df, x=feature, hue='Class', bins=50, kde=True, ax=ax)
                plt.title(f'Distribution of {feature} by Class')
                st.pyplot(fig)

if __name__ == "__main__":
    app()
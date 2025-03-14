# 💳 Financial Fraud Detection System

## Overview

The Financial Fraud Detection System is a comprehensive machine learning application designed to detect fraudulent financial transactions. Built using Python and Streamlit, this system provides an end-to-end solution for data preprocessing, feature engineering, model training, and fraud detection. The application supports multiple machine learning models, including Logistic Regression, Random Forest, and XGBoost, and offers detailed visualizations and performance metrics to help users understand and interpret the results.

## Features

- **Data Upload**: Upload your transaction data in CSV format.
- **Data Preprocessing**: Handle missing values, remove outliers, and normalize data.
- **Feature Engineering**: Create new features based on transaction time, amount, and other attributes.
- **Model Training**: Train multiple machine learning models with customizable parameters.
- **Fraud Detection**: Detect fraudulent transactions using the trained models.
- **Visualizations**: Interactive visualizations including ROC curves, confusion matrices, and feature importance plots.
- **Performance Metrics**: Detailed performance metrics including accuracy, precision, recall, F1 score, and AUC.
- **Download Results**: Download the predictions and high-risk transactions as CSV files.

## 🌐 Demo
🔗 **Live App:** [Financial Fraud Detection System](https://huggingface.co/spaces/sfy45/Financial-Fraud-Detection)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sfy45/financial-fraud-detection.git
   cd financial-fraud-detection
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Upload Data**: Start by uploading your transaction data in CSV format. Ensure your data includes a target column (usually named 'Class') indicating fraud (1) or non-fraud (0).

2. **Preprocess Data**: Handle missing values, remove outliers, and normalize your data.

3. **Feature Engineering**: Create new features to improve the detection of fraudulent transactions.

4. **Train Models**: Select and train machine learning models. Customize model parameters and training options as needed.

5. **View Results**: Analyze the fraud detection results, including visualizations and performance metrics. Download the predictions and high-risk transactions for further analysis.

## Project Structure

```
financial-fraud-detection/
│
├── app.py                  
├── README.md               
├── requirements.txt        
├── data/
    |_.gitkeep                
├── models/
    |_.gitkeep                
├── pages/
    |_data_exploration.py              
└── utils/
    |_data_processor.py
    |_model_trainer.py
    |_visualizer.py                 
```

## Dependencies

- **Streamlit**: For building the web application.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Scikit-learn**: For machine learning models and evaluation metrics.
- **XGBoost**: For the XGBoost classifier.
- **Plotly**: For interactive visualizations.
- **Seaborn**: For statistical data visualization.
- **Imbalanced-learn**: For handling class imbalance using SMOTE.

## Contributing

If you have a contribution to make, feel free to submit issues or pull requests. PRs are more than welcome!


## Contact

For any issues or suggestions, feel free to reach out: 📧 sophiasad1421@gmail.com

---

Thank you for using the Financial Fraud Detection System! I hope it helps you in identifying and preventing fraudulent transactions effectively.

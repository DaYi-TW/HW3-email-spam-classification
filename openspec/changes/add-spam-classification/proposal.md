# Add Spam Email Classification

## Why
This project requires a machine learning-based spam email classification system for cybersecurity coursework (HW3). Email spam detection is a fundamental cybersecurity application that protects users from phishing, malware, and unwanted content. We need to build a baseline classifier using SVM with plans for future enhancements in subsequent phases.

## What Changes
- **Phase 1 (Baseline)**: Implement comprehensive spam classification system with multiple ML models
  - **Data pre-processing**: 
    - Load SMS spam dataset from course repository
    - Data cleaning (remove duplicates, handle missing values, normalize text)
    - Text tokenization (word splitting, lowercasing)
    - Vectorization using TF-IDF and bag-of-words approaches
  
  - **Model training** (implement and compare three algorithms):
    - Logistic Regression classifier
    - Naïve Bayes classifier
    - Support Vector Machine (SVM) classifier
  
  - **Evaluation**:
    - Calculate metrics (accuracy, precision, recall, F1-score) for all models
    - Generate comparison charts (bar charts, confusion matrices, ROC curves)
    - Create performance visualization dashboards
  
  - **Visualization and Streamlit UI**:
    - Build interactive web interface using Streamlit
    - Real-time spam detection interface
    - Model comparison dashboard
    - Performance metrics visualization
    - Interactive prediction with user input
  
  - Model persistence and deployment-ready artifacts

- **Phase 2+**: Placeholder for future enhancements
  - Reserved for advanced feature engineering experiments
  - Reserved for deep learning models (LSTM, BERT)
  - Reserved for ensemble methods and model optimization

## Impact
- **Affected specs**: `email-spam-classifier` (new capability)
- **Affected code**: 
  - Core ML pipeline script for training and evaluation (in `ml/` folder)
  - Streamlit web application for UI (in `ml/` folder)
  - Model artifacts and saved vectorizers (in `ml/models/` folder)
  - All Python scripts and notebooks must be organized within `ml/` directory
- **Dataset dependency**: External CSV file from course GitHub repository
- **Dependencies**: 
  - Core ML: scikit-learn, pandas, numpy
  - Visualization: matplotlib, seaborn
  - UI: streamlit
  - NLP: nltk or spaCy (for advanced tokenization)

## Project Structure
```
ml/
├── spam_classifier.py          # Main ML pipeline script
├── streamlit_app.py           # Streamlit web interface
├── models/                    # Saved model artifacts
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── svm.pkl
│   └── vectorizer.pkl
├── data/                      # Dataset storage
│   └── sms_spam_no_header.csv
├── charts/                    # Generated visualizations
│   ├── accuracy_comparison.png
│   ├── metrics_comparison.png
│   ├── confusion_matrices.png
│   └── roc_curves.png
├── requirements.txt           # Python dependencies
└── README.md                  # Setup and usage documentation
```

## Breaking Changes
None - this is a new capability with no existing dependencies.

## Risks
- Dataset availability: Relies on external GitHub URL (mitigated by using official course repository)
- Class imbalance: SMS dataset may have unequal spam/ham distribution (will monitor during implementation)
- Model persistence: Need to ensure reproducibility with random seed settings

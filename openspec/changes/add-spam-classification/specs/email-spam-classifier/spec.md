# Email Spam Classifier Specification

## ADDED Requirements

### Requirement: Dataset Loading
The system SHALL load SMS spam dataset from the official course repository URL and prepare it for training.

#### Scenario: Successful dataset download
- **WHEN** the dataset URL is accessible
- **THEN** the system downloads and loads the CSV file into memory
- **AND** the dataset contains text messages and spam/ham labels

#### Scenario: Missing column headers
- **WHEN** the CSV file has no headers
- **THEN** the system adds appropriate column names (label, message)
- **AND** the data structure is ready for processing

#### Scenario: Handle missing or malformed data
- **WHEN** the dataset contains null or empty values
- **THEN** the system identifies and handles missing data appropriately
- **AND** logs any data quality issues

### Requirement: Data Preprocessing
The system SHALL preprocess text data and labels through cleaning, tokenization, and vectorization to prepare for machine learning model training.

#### Scenario: Text cleaning
- **WHEN** raw text messages are loaded
- **THEN** the system converts text to lowercase
- **AND** removes special characters, numbers, and punctuation
- **AND** removes extra whitespace and duplicates
- **AND** logs data cleaning statistics

#### Scenario: Text tokenization
- **WHEN** cleaned text is ready for processing
- **THEN** the system tokenizes text into individual words
- **AND** removes stop words (common words like 'the', 'is', 'a')
- **AND** optionally applies stemming or lemmatization
- **AND** maintains token consistency across the pipeline

#### Scenario: Binary label encoding
- **WHEN** labels are in text format (spam, ham)
- **THEN** the system encodes them to binary values (1 for spam, 0 for ham)
- **AND** maintains consistent encoding throughout the pipeline

#### Scenario: Train-test split
- **WHEN** preprocessing the dataset
- **THEN** the system splits data into training (80%) and testing (20%) sets
- **AND** maintains stratification to preserve class distribution
- **AND** sets random seed for reproducibility

### Requirement: Feature Vectorization
The system SHALL convert preprocessed and tokenized text messages into numerical feature vectors suitable for machine learning using multiple vectorization approaches.

#### Scenario: TF-IDF vectorization
- **WHEN** extracting features from tokenized text messages
- **THEN** the system applies TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- **AND** fits the vectorizer only on training data to prevent data leakage
- **AND** transforms both training and testing data using the fitted vectorizer
- **AND** documents vectorization parameters (max_features, ngram_range)

#### Scenario: Bag-of-words vectorization
- **WHEN** comparing feature extraction methods
- **THEN** the system provides bag-of-words (CountVectorizer) as an alternative
- **AND** maintains consistent vocabulary between training and testing
- **AND** enables comparison between TF-IDF and bag-of-words performance

#### Scenario: Feature consistency
- **WHEN** processing new messages for prediction
- **THEN** the system uses the same fitted vectorizer from training
- **AND** produces feature vectors with consistent dimensionality
- **AND** handles out-of-vocabulary words appropriately

### Requirement: Multi-Model Training
The system SHALL train and compare three different machine learning classifiers: Logistic Regression, Naïve Bayes, and Support Vector Machine.

#### Scenario: Logistic Regression training
- **WHEN** training data features and labels are prepared
- **THEN** the system initializes a Logistic Regression classifier
- **AND** sets random_state for reproducible results
- **AND** fits the model on training features and labels
- **AND** logs training completion time and parameters

#### Scenario: Naïve Bayes training
- **WHEN** training data features and labels are prepared
- **THEN** the system initializes a Multinomial Naïve Bayes classifier
- **AND** fits the model on training features and labels
- **AND** logs training completion time
- **AND** handles zero probabilities appropriately

#### Scenario: Support Vector Machine training
- **WHEN** training data features and labels are prepared
- **THEN** the system initializes an SVM classifier with linear kernel
- **AND** sets random_state for reproducible results
- **AND** fits the model on training features and labels
- **AND** logs training completion time and parameters

#### Scenario: Model parameters documentation
- **WHEN** any model is trained
- **THEN** the system records all hyperparameters used
- **AND** stores model metadata (type, parameters, training time)
- **AND** makes parameters available for future reference and comparison

### Requirement: Model Evaluation with Metrics and Charts
The system SHALL evaluate all trained models using standard classification metrics and generate comprehensive visualizations for comparison.

#### Scenario: Performance metrics calculation
- **WHEN** each model makes predictions on the test set
- **THEN** the system calculates accuracy, precision, recall, and F1-score
- **AND** generates a confusion matrix showing true/false positives and negatives
- **AND** calculates ROC-AUC scores
- **AND** records training time for each model
- **AND** displays results in a human-readable format

#### Scenario: Classification report
- **WHEN** evaluation is complete for all models
- **THEN** the system produces detailed classification reports
- **AND** shows per-class metrics (spam and ham separately)
- **AND** includes support (number of samples) for each class
- **AND** creates a comparison table across all three models

#### Scenario: Evaluation visualization - Bar charts
- **WHEN** metrics are calculated for all models
- **THEN** the system generates bar charts comparing accuracy across models
- **AND** creates grouped bar charts for precision, recall, and F1-score
- **AND** generates training time comparison chart
- **AND** saves all charts as image files

#### Scenario: Evaluation visualization - Confusion matrices
- **WHEN** predictions are complete
- **THEN** the system visualizes confusion matrices as heatmaps
- **AND** displays confusion matrices for all three models side-by-side
- **AND** includes percentage annotations in the heatmaps

#### Scenario: Evaluation visualization - ROC curves
- **WHEN** calculating model performance
- **THEN** the system plots ROC curves for all three models on the same chart
- **AND** displays AUC scores in the legend
- **AND** highlights the best-performing model
- **AND** saves the ROC comparison chart

#### Scenario: Performance thresholds
- **WHEN** any model is evaluated
- **THEN** the system achieves at least 90% accuracy as a baseline target
- **AND** logs a warning if performance is below threshold
- **AND** recommends the best model based on overall metrics

### Requirement: Model Persistence
The system SHALL save all trained models and vectorizers for reuse without retraining.

#### Scenario: Save trained artifacts
- **WHEN** model training and evaluation are complete
- **THEN** the system saves all three trained models (Logistic Regression, Naïve Bayes, SVM) to `ml/models/` directory
- **AND** saves the fitted TF-IDF vectorizer to `ml/models/vectorizer.pkl`
- **AND** uses a consistent serialization format (joblib or pickle)
- **AND** stores model metadata (accuracy, parameters, training time)

#### Scenario: Load and predict with saved model
- **WHEN** loading a previously saved model
- **THEN** the system successfully loads both model and vectorizer from `ml/models/`
- **AND** can make predictions on new text messages
- **AND** produces identical results to the original trained model
- **AND** supports loading any of the three trained models

### Requirement: Prediction Interface
The system SHALL provide a simple interface for classifying new messages as spam or ham.

#### Scenario: Single message prediction
- **WHEN** given a new text message
- **THEN** the system preprocesses and vectorizes the text
- **AND** applies the trained model to predict spam (1) or ham (0)
- **AND** returns the prediction with confidence score if available

#### Scenario: Batch prediction
- **WHEN** given multiple messages to classify
- **THEN** the system processes all messages efficiently
- **AND** returns predictions for each message
- **AND** maintains consistent preprocessing across all inputs

### Requirement: Streamlit Web UI
The system SHALL provide an interactive web interface using Streamlit for real-time spam detection, model comparison, and performance visualization.

#### Scenario: Spam detection interface
- **WHEN** a user opens the Streamlit application
- **THEN** the system displays a text input area for entering messages
- **AND** provides a dropdown to select model (Logistic Regression, Naïve Bayes, or SVM)
- **AND** includes a "Classify" button to trigger prediction
- **AND** displays prediction result (Spam/Ham) with confidence score
- **AND** provides example messages for quick testing

#### Scenario: Real-time prediction
- **WHEN** a user enters text and clicks "Classify"
- **THEN** the system loads the selected model and vectorizer
- **AND** preprocesses and vectorizes the input text
- **AND** makes a prediction in real-time
- **AND** displays the result with prediction probability
- **AND** shows visual indicators (colors, icons) for spam vs ham

#### Scenario: Model comparison dashboard
- **WHEN** a user navigates to the comparison page
- **THEN** the system displays a comprehensive comparison table with all metrics
- **AND** embeds interactive charts showing accuracy, precision, recall, F1-score
- **AND** displays confusion matrices side-by-side for all models
- **AND** shows ROC curves comparison
- **AND** highlights the best-performing model with recommendations

#### Scenario: Performance visualization dashboard
- **WHEN** viewing the visualization page
- **THEN** the system displays metric cards for key statistics
- **AND** shows dataset statistics (total samples, spam/ham distribution)
- **AND** visualizes top TF-IDF terms or feature importance
- **AND** provides download buttons for evaluation reports
- **AND** includes interactive charts for exploration

#### Scenario: UI navigation and layout
- **WHEN** the Streamlit app is running
- **THEN** the system provides a sidebar for page navigation
- **AND** maintains consistent layout and styling across pages
- **AND** includes app header with title and description
- **AND** responds quickly to user interactions
- **AND** handles errors gracefully with informative messages

### Requirement: Phase 2+ Extensibility
The system SHALL be structured to support future enhancement phases without major refactoring.

#### Scenario: Algorithm extensibility
- **WHEN** implementing Phase 2 with additional models
- **THEN** the system allows easy addition of new classifiers
- **AND** maintains the same preprocessing and evaluation pipeline
- **AND** enables comparison across all models in the Streamlit UI

#### Scenario: Feature engineering extensions
- **WHEN** adding new feature extraction methods
- **THEN** the system supports plugging in alternative vectorizers
- **AND** maintains backward compatibility with saved Phase 1 models
- **AND** allows A/B testing between different feature approaches

## Notes

**Dataset Source**: https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv

**Project Structure**: All code must be organized in the `ml/` directory:
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
└── README.md                  # Documentation
```

**Phase 1 Focus**: This specification includes comprehensive data preprocessing (cleaning, tokenization, vectorization), three ML models (Logistic Regression, Naïve Bayes, SVM), detailed evaluation with charts, and an interactive Streamlit UI.

**Future Phases**: Phase 2+ will explore advanced feature engineering (word embeddings), deep learning models (LSTM, BERT), ensemble methods, and hyperparameter optimization.

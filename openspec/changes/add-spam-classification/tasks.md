# Implementation Tasks

## 1. Phase 1 - Multi-Model Spam Classifier with Streamlit UI

### 1.1 Project Setup
- [ ] 1.1.1 Create `ml/` directory structure
- [ ] 1.1.2 Create subdirectories: `ml/models/`, `ml/data/`, `ml/charts/`
- [ ] 1.1.3 Create `ml/requirements.txt` with all dependencies
- [ ] 1.1.4 Create `ml/spam_classifier.py` for main ML pipeline
- [ ] 1.1.5 Create `ml/streamlit_app.py` for web interface

### 1.2 Data Loading
- [ ] 1.2.1 Implement function to download CSV from GitHub URL
- [ ] 1.2.2 Save dataset to `ml/data/` directory
- [ ] 1.2.3 Load dataset into pandas DataFrame
- [ ] 1.2.4 Verify data structure and handle missing values
- [ ] 1.2.5 Remove duplicate entries
- [ ] 1.2.6 Perform initial data exploration and statistics

### 1.3 Data Preprocessing and Cleaning
- [ ] 1.3.1 Add column headers (label, message) to dataset
- [ ] 1.3.2 Convert text to lowercase for consistency
- [ ] 1.3.3 Remove special characters, numbers, and punctuation
- [ ] 1.3.4 Remove extra whitespace
- [ ] 1.3.5 Encode labels (spam/ham) to binary (1/0)
- [ ] 1.3.6 Handle class imbalance (check distribution)
- [ ] 1.3.7 Split dataset into training and testing sets (80/20)
- [ ] 1.3.8 Verify train-test split stratification

### 1.4 Text Tokenization
- [ ] 1.4.1 Implement tokenization function (word splitting)
- [ ] 1.4.2 Remove stop words (common words like 'the', 'is', 'a')
- [ ] 1.4.3 Apply stemming or lemmatization (optional)
- [ ] 1.4.4 Document tokenization choices and rationale

### 1.5 Feature Vectorization
- [ ] 1.5.1 Implement TF-IDF vectorization
- [ ] 1.5.2 Implement bag-of-words (CountVectorizer) as alternative
- [ ] 1.5.3 Fit vectorizers on training text data only
- [ ] 1.5.4 Transform both training and testing data to feature vectors
- [ ] 1.5.5 Document feature extraction parameters (max_features, ngram_range, etc.)
- [ ] 1.5.6 Compare TF-IDF vs bag-of-words feature quality

### 1.6 Model Training - Logistic Regression
- [ ] 1.6.1 Import LogisticRegression from scikit-learn
- [ ] 1.6.2 Initialize Logistic Regression with parameters
- [ ] 1.6.3 Set random_state for reproducibility
- [ ] 1.6.4 Train model on training features and labels
- [ ] 1.6.5 Log training time and model parameters

### 1.7 Model Training - Naïve Bayes
- [ ] 1.7.1 Import MultinomialNB from scikit-learn
- [ ] 1.7.2 Initialize Naïve Bayes classifier
- [ ] 1.7.3 Train model on training features and labels
- [ ] 1.7.4 Log training time and model parameters

### 1.8 Model Training - Support Vector Machine
- [ ] 1.8.1 Import SVC (Support Vector Classifier) from scikit-learn
- [ ] 1.8.2 Initialize SVM with linear kernel
- [ ] 1.8.3 Set random_state for reproducibility
- [ ] 1.8.4 Train model on training features and labels
- [ ] 1.8.5 Log training time and model parameters

### 1.9 Model Evaluation - Metrics Calculation
- [ ] 1.9.1 Make predictions on test set for all three models
- [ ] 1.9.2 Calculate accuracy score for each model
- [ ] 1.9.3 Generate classification report (precision, recall, F1-score) for each
- [ ] 1.9.4 Create confusion matrix for each model
- [ ] 1.9.5 Calculate ROC-AUC scores
- [ ] 1.9.6 Create model comparison table

### 1.10 Evaluation Visualization - Charts
- [ ] 1.10.1 Create bar chart comparing accuracy across models
- [ ] 1.10.2 Create grouped bar chart for precision, recall, F1-score comparison
- [ ] 1.10.3 Visualize confusion matrices as heatmaps for all models
- [ ] 1.10.4 Plot ROC curves for all three models
- [ ] 1.10.5 Create training time comparison chart
- [ ] 1.10.6 Save all charts to `ml/charts/` directory

### 1.11 Model Persistence
- [ ] 1.11.1 Save all three trained models to `ml/models/` directory using joblib or pickle
- [ ] 1.11.2 Save fitted vectorizer to `ml/models/vectorizer.pkl`
- [ ] 1.11.3 Create model registry with metadata (accuracy, parameters, timestamp)
- [ ] 1.11.4 Document model file locations and naming conventions
- [ ] 1.11.5 Test loading saved models from `ml/models/` and making predictions

### 1.12 Streamlit UI - Setup
- [ ] 1.12.1 Create `ml/streamlit_app.py` for the Streamlit application
- [ ] 1.12.2 Set up Streamlit page configuration (title, icon, layout)
- [ ] 1.12.3 Create sidebar for navigation
- [ ] 1.12.4 Add app header and description
- [ ] 1.12.5 Configure paths to load models from `ml/models/` directory

### 1.13 Streamlit UI - Spam Detection Interface
- [ ] 1.13.1 Create text input area for user message
- [ ] 1.13.2 Add model selector dropdown (LR, NB, SVM)
- [ ] 1.13.3 Implement "Classify" button
- [ ] 1.13.4 Load selected model and vectorizer on demand
- [ ] 1.13.5 Display prediction result (Spam/Ham) with confidence
- [ ] 1.13.6 Add example spam/ham messages for testing
- [ ] 1.13.7 Show prediction probability scores

### 1.14 Streamlit UI - Model Comparison Dashboard
- [ ] 1.14.1 Create dedicated page for model comparison
- [ ] 1.14.2 Display comparison table with all metrics
- [ ] 1.14.3 Embed accuracy bar chart
- [ ] 1.14.4 Embed precision/recall/F1 comparison chart
- [ ] 1.14.5 Display confusion matrices side-by-side
- [ ] 1.14.6 Show ROC curves comparison
- [ ] 1.14.7 Add model recommendation based on metrics

### 1.15 Streamlit UI - Performance Visualization
- [ ] 1.15.1 Create metrics cards (st.metric) for key statistics
- [ ] 1.15.2 Add interactive charts using plotly (optional enhancement)
- [ ] 1.15.3 Display dataset statistics (total samples, spam/ham ratio)
- [ ] 1.15.4 Show feature importance or top TF-IDF terms
- [ ] 1.15.5 Add download button for evaluation reports

### 1.16 Documentation
- [ ] 1.16.1 Add comments explaining each step in all scripts
- [ ] 1.16.2 Create `ml/README.md` with setup and usage instructions
- [ ] 1.16.3 Document model performance results
- [ ] 1.16.4 Include example predictions with sample texts
- [ ] 1.16.5 Document Streamlit app features and navigation
- [ ] 1.16.6 Create `ml/requirements.txt` with all dependencies
- [ ] 1.16.7 Add instructions for running from `ml/` directory
- [ ] 1.16.8 Note observations about dataset and model behavior

## 2. Phase 2 - Future Enhancements
- [ ] 2.1 (Placeholder) Plan advanced feature engineering (n-grams, word embeddings)
- [ ] 2.2 (Placeholder) Plan deep learning models (LSTM, BERT)
- [ ] 2.3 (Placeholder) Plan ensemble methods and hyperparameter tuning

## 3. Phase 3+ - Additional Phases
- [ ] 3.1 (Placeholder) Reserved for future phases

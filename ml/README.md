# ML folder — Spam Classifier

This folder contains the Phase 1 implementation for the Spam Email/SMS classifier.

Contents
- `spam_classifier.py` — Training script that downloads the SMS spam dataset, preprocesses text, trains Logistic Regression, Naive Bayes, and SVM, saves models to `ml/models/` and charts to `ml/charts/`.
- `streamlit_app.py` — Small Streamlit app to load saved models and provide an interactive classification UI.
- `models/` — Saved model artifacts (created after running training).
- `charts/` — Evaluation charts (created after running training).

Quickstart
1. Activate the conda env created for this project:

```powershell
conda activate spam-classifier
```

2. Train models (if not already trained):

```powershell
python ml\spam_classifier.py
```

3. Run the Streamlit UI:

```powershell
streamlit run ml\streamlit_app.py
```

Notes
- The Streamlit app expects saved artifacts in `ml/models/`: `vectorizer.pkl` and the model `.pkl` files.
- The project uses scikit-learn, pandas, numpy, matplotlib, seaborn, streamlit, nltk, and joblib.
# Spam Email Classifier - Phase 1

A comprehensive spam email classification system using machine learning for Cybersecurity HW3.

## 📋 Project Overview

This project implements a multi-model spam email classification system that:
- Trains and compares three ML algorithms (Logistic Regression, Naïve Bayes, SVM)
- Provides comprehensive evaluation metrics and visualizations
- Deploys an interactive web interface for real-time predictions

## 🏗️ Project Structure

```
ml/
├── spam_classifier.py          # Main ML pipeline script
├── streamlit_app.py           # Streamlit web interface
├── models/                    # Saved model artifacts
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── support_vector_machine.pkl
│   └── vectorizer.pkl
├── data/                      # Dataset storage
│   └── sms_spam_no_header.csv
├── charts/                    # Generated visualizations
│   ├── accuracy_comparison.png
│   ├── metrics_comparison.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── training_time.png
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd ml
pip install -r requirements.txt
```

### 2. Train Models

Run the main classifier script to:
- Download the SMS spam dataset
- Preprocess and clean the data
- Train all three models
- Generate evaluation metrics
- Create visualization charts
- Save trained models

```bash
python spam_classifier.py
```

This will take a few minutes to complete. You'll see:
- Data loading and exploration statistics
- Preprocessing steps
- Training progress for each model
- Evaluation metrics and comparison table
- Example predictions

### 3. Launch Web Interface

Start the Streamlit web application:

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Features

### Data Preprocessing
- **Text Cleaning**: Lowercase conversion, special character removal, whitespace normalization
- **Tokenization**: Word splitting with stop word removal
- **Vectorization**: TF-IDF with bigrams (max 3000 features)
- **Label Encoding**: Binary encoding (spam=1, ham=0)
- **Train-Test Split**: 80/20 split with stratification

### Machine Learning Models

#### 1. Logistic Regression
- Fast training and prediction
- Provides probability estimates
- Good baseline performance

#### 2. Naïve Bayes (Multinomial)
- Probabilistic classifier
- Efficient for text data
- Handles high-dimensional features well

#### 3. Support Vector Machine (Linear)
- Finds optimal decision boundary
- Effective in high-dimensional spaces
- Robust to overfitting

### Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: Spam detection accuracy
- **Recall**: Spam catch rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: True/false positives and negatives
- **Training Time**: Model efficiency comparison

### Visualizations
1. **Accuracy Comparison**: Bar chart showing accuracy across models
2. **Metrics Comparison**: Grouped bar chart for precision, recall, F1-score
3. **Confusion Matrices**: Heatmaps for all three models
4. **ROC Curves**: Comparison of model discrimination ability
5. **Training Time**: Efficiency comparison

### Streamlit Web Interface

#### 🏠 Home Page
- Text input for messages
- Model selector dropdown
- Real-time spam/ham prediction
- Confidence scores and probabilities
- Example messages for testing

#### 📊 Model Comparison Dashboard
- Performance metrics table
- Interactive visualizations
- Side-by-side model comparison
- Test all models simultaneously

#### ℹ️ About Page
- Project overview
- Technical details
- Model explanations
- Future enhancements

## 📈 Dataset

- **Name**: SMS Spam Collection Dataset
- **Source**: [PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity)
- **Format**: Tab-separated values (TSV)
- **Size**: ~5,500 messages
- **Labels**: spam / ham
- **Language**: English

## 🔧 Technical Stack

- **Python 3.x**: Primary programming language
- **scikit-learn**: ML algorithms and preprocessing
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **matplotlib**: Visualization
- **seaborn**: Statistical visualization
- **streamlit**: Web interface
- **joblib**: Model persistence
- **requests**: HTTP requests for dataset download

## 💡 Usage Examples

### Training Models

```bash
# Navigate to ml directory
cd ml

# Run the classifier
python spam_classifier.py
```

**Expected Output:**
```
============================================================
SPAM EMAIL CLASSIFIER - PHASE 1
Multi-Model Baseline Implementation
============================================================

Downloading dataset from https://...
✓ Dataset saved to data/sms_spam_no_header.csv

============================================================
DATA LOADING AND EXPLORATION
============================================================

✓ Dataset loaded successfully
  Total samples: 5572
  Features: ['label', 'message']
  ...

[Training completes with metrics and visualizations]

✓ PHASE 1 IMPLEMENTATION COMPLETE!
```

### Using Streamlit App

```bash
# Start the app
streamlit run streamlit_app.py

# App opens at http://localhost:8501
```

### Programmatic Usage

```python
import joblib

# Load models
lr_model = joblib.load('models/logistic_regression.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# Predict
message = "Congratulations! You won a prize!"
message_vec = vectorizer.transform([message.lower()])
prediction = lr_model.predict(message_vec)[0]

print("SPAM" if prediction == 1 else "HAM")
```

## 📊 Expected Performance

Based on Phase 1 baseline implementation:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~96% | ~95% | ~90% | ~92% |
| Naïve Bayes | ~97% | ~98% | ~85% | ~91% |
| SVM | ~98% | ~99% | ~90% | ~94% |

*Note: Actual results may vary depending on random seed and data split.*

## 🛠️ Troubleshooting

### Issue: Import errors
**Solution**: Install all dependencies
```bash
pip install -r requirements.txt
```

### Issue: Models not found in Streamlit app
**Solution**: Train models first
```bash
python spam_classifier.py
```

### Issue: Dataset download fails
**Solution**: Check internet connection or download manually
```bash
# Download from:
https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/master/Chapter03/datasets/sms_spam_no_header.csv

# Save to: ml/data/sms_spam_no_header.csv
```

### Issue: Streamlit port already in use
**Solution**: Use a different port
```bash
streamlit run streamlit_app.py --server.port 8502
```

## 🔮 Future Enhancements (Phase 2+)

- **Advanced Feature Engineering**
  - Word embeddings (Word2Vec, GloVe)
  - Character-level n-grams
  - Sentiment analysis features

- **Deep Learning Models**
  - LSTM networks
  - BERT transformers
  - Attention mechanisms

- **Ensemble Methods**
  - Voting classifiers
  - Stacking
  - Boosting (XGBoost, LightGBM)

- **Hyperparameter Optimization**
  - Grid search
  - Random search
  - Bayesian optimization

- **Deployment**
  - REST API
  - Docker containerization
  - Cloud deployment (Azure/AWS)

## 📝 Notes

- All models use `random_state=42` for reproducibility
- TF-IDF vectorizer uses English stop words
- 80/20 train-test split with stratification
- Models are saved in pickle format using joblib
- Charts are saved as PNG images at 300 DPI

## 📚 References

- [scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Hands-On AI for Cybersecurity Book](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity)
- [SMS Spam Collection Dataset](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/)

## 👨‍💻 Author

**Cybersecurity HW3 Project**  
Internet of Things (IoT) Course  
October 23, 2025

---

## 📄 License

This project is for educational purposes as part of coursework.

---

**Need help?** Check the troubleshooting section above or review the code comments in `spam_classifier.py` and `streamlit_app.py`.

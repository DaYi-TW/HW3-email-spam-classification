"""
Spam Email Classifier - Phase 1: Multi-Model Baseline
======================================================

This script implements a comprehensive spam classification system using:
- Logistic Regression
- Naïve Bayes
- Support Vector Machine (SVM)

Features:
- Data preprocessing (cleaning, tokenization, vectorization)
- Multi-model training and comparison
- Comprehensive evaluation with metrics and visualizations
- Model persistence for deployment

Author: Cybersecurity HW3
Date: October 23, 2025
"""

import os
import time
import warnings
import requests
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Dataset URL
DATASET_URL = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"

# Paths (store artifacts under ml/ to match project layout)
DATA_DIR = os.path.join('ml', 'data')
MODELS_DIR = os.path.join('ml', 'models')
CHARTS_DIR = os.path.join('ml', 'charts')


def download_dataset(url, save_path):
    """
    Download dataset from URL and save to disk.
    
    Args:
        url (str): URL to download from
        save_path (str): Path to save the file
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Downloading dataset from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Dataset saved to {save_path}")
        return True
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        return False


def load_and_explore_data(file_path):
    """
    Load dataset and perform initial exploration.
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print("\n" + "="*60)
    print("DATA LOADING AND EXPLORATION")
    print("="*60)
    
    # Load data with appropriate column names
    # File uses quoted CSV with comma separator (no header)
    df = pd.read_csv(file_path, header=None, names=['label', 'message'], sep=',', quotechar='"', encoding='utf-8')
    
    print(f"\n✓ Dataset loaded successfully")
    print(f"  Total samples: {len(df)}")
    print(f"  Features: {df.columns.tolist()}")
    
    # Check for missing values
    missing = df.isnull().sum()
    print(f"\n  Missing values:\n{missing}")
    
    # Remove duplicates
    original_len = len(df)
    df = df.drop_duplicates()
    duplicates_removed = original_len - len(df)
    print(f"  Duplicates removed: {duplicates_removed}")
    
    # Class distribution
    print(f"\n  Class distribution:")
    class_counts = df['label'].value_counts()
    for label, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"    {label}: {count} ({percentage:.1f}%)")
    
    return df


def preprocess_text(df):
    """
    Preprocess text data: cleaning, tokenization, encoding.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)
    
    # Ensure message column is string and handle missing values
    if 'message' not in df.columns:
        raise KeyError("Expected column 'message' in dataframe")

    # Coerce to string and replace NaNs with empty string
    df['message'] = df['message'].astype(str).fillna('')

    # Convert to lowercase
    df['message'] = df['message'].str.lower()
    print("✓ Converted text to lowercase")

    # Remove special characters, numbers, punctuation (keeping spaces)
    df['message'] = df['message'].str.replace(r'[^a-z\s]', '', regex=True)
    print("✓ Removed special characters and numbers")

    # Remove extra whitespace
    df['message'] = df['message'].str.replace(r'\s+', ' ', regex=True).str.strip()
    print("✓ Removed extra whitespace")

    # Drop rows with empty messages after cleaning
    before_drop = len(df)
    df = df[df['message'].str.len() > 0].copy()
    dropped = before_drop - len(df)
    if dropped > 0:
        print(f"  ⚠ Dropped {dropped} empty messages after preprocessing")

    # Normalize and encode labels: spam=1, ham=0
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['label_encoded'] = df['label'].map({'spam': 1, 'ham': 0})

    # Drop rows where label mapping failed
    before_label_drop = len(df)
    df = df.dropna(subset=['label_encoded']).copy()
    label_dropped = before_label_drop - len(df)
    if label_dropped > 0:
        print(f"  ⚠ Dropped {label_dropped} rows with unknown labels")

    print("✓ Encoded labels (spam=1, ham=0)")
    
    print(f"\n  Sample preprocessed messages:")
    for i in range(min(3, len(df))):
        print(f"    [{df.iloc[i]['label']}] {df.iloc[i]['message'][:60]}...")
    
    return df


def create_features(X_train, X_test, vectorizer_type='tfidf'):
    """
    Convert text to numerical features using vectorization.
    
    Args:
        X_train: Training text data
        X_test: Testing text data
        vectorizer_type (str): 'tfidf' or 'count'
    
    Returns:
        tuple: (X_train_vec, X_test_vec, vectorizer)
    """
    print("\n" + "="*60)
    print(f"FEATURE EXTRACTION ({vectorizer_type.upper()})")
    print("="*60)
    
    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words='english'
        )
    else:
        vectorizer = CountVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words='english'
        )
    
    # Fit on training data only
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"✓ Vectorization complete")
    print(f"  Training features shape: {X_train_vec.shape}")
    print(f"  Testing features shape: {X_test_vec.shape}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    return X_train_vec, X_test_vec, vectorizer


def train_model(model, model_name, X_train, y_train):
    """
    Train a machine learning model.
    
    Args:
        model: sklearn model instance
        model_name (str): Name of the model
        X_train: Training features
        y_train: Training labels
    
    Returns:
        tuple: (trained_model, training_time)
    """
    print(f"\n  Training {model_name}...")
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"  ✓ {model_name} trained in {training_time:.2f} seconds")
    
    return model, training_time


def evaluate_model(model, model_name, X_test, y_test):
    """
    Evaluate a trained model and return metrics.
    
    Args:
        model: Trained sklearn model
        model_name (str): Name of the model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    # Calculate ROC-AUC if probability predictions available
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        metrics['fpr'], metrics['tpr'], _ = roc_curve(y_test, y_pred_proba)
    
    return metrics


def print_evaluation_summary(results, training_times):
    """
    Print a summary of all model evaluations.
    
    Args:
        results (dict): Dictionary of model results
        training_times (dict): Dictionary of training times
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION SUMMARY")
    print("="*60)
    
    # Create comparison table
    print(f"\n{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Time(s)':<10}")
    print("-" * 95)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f} {training_times[model_name]:<10.2f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
    print(f"\n✓ Best performing model: {best_model[0]} (F1-Score: {best_model[1]['f1_score']:.4f})")
    
    # Check threshold
    for model_name, metrics in results.items():
        if metrics['accuracy'] < 0.90:
            print(f"\n⚠ Warning: {model_name} accuracy ({metrics['accuracy']:.4f}) below 90% threshold")


def create_visualizations(results, training_times, save_dir):
    """
    Create and save all visualization charts.
    
    Args:
        results (dict): Dictionary of model results
        training_times (dict): Dictionary of training times
        save_dir (str): Directory to save charts
    """
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Accuracy Comparison Bar Chart
    print("\n  Creating accuracy comparison chart...")
    fig, ax = plt.subplots(figsize=(10, 6))
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    
    bars = ax.bar(models, accuracies, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.9, color='red', linestyle='--', label='90% Threshold')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_comparison.png'), dpi=300)
    print(f"  ✓ Saved: {save_dir}/accuracy_comparison.png")
    plt.close()
    
    # 2. Metrics Comparison (Grouped Bar Chart)
    print("  Creating metrics comparison chart...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(['accuracy', 'precision', 'recall', 'f1_score']):
        values = [results[m][metric] for m in models]
        ax.bar(x + i*width, values, width, label=metrics_names[i])
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=300)
    print(f"  ✓ Saved: {save_dir}/metrics_comparison.png")
    plt.close()
    
    # 3. Confusion Matrices
    print("  Creating confusion matrices...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (model_name, metrics) in enumerate(results.items()):
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'), dpi=300)
    print(f"  ✓ Saved: {save_dir}/confusion_matrices.png")
    plt.close()
    
    # 4. ROC Curves
    print("  Creating ROC curves...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    for idx, (model_name, metrics) in enumerate(results.items()):
        if 'fpr' in metrics and 'tpr' in metrics:
            ax.plot(metrics['fpr'], metrics['tpr'], 
                   label=f"{model_name} (AUC = {metrics['roc_auc']:.4f})",
                   color=colors[idx], linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=300)
    print(f"  ✓ Saved: {save_dir}/roc_curves.png")
    plt.close()
    
    # 5. Training Time Comparison
    print("  Creating training time chart...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    times = [training_times[m] for m in models]
    bars = ax.bar(models, times, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Model Training Time Comparison', fontsize=14, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_time.png'), dpi=300)
    print(f"  ✓ Saved: {save_dir}/training_time.png")
    plt.close()
    
    print("\n✓ All visualizations created successfully!")


def save_models(models, vectorizer, save_dir):
    """
    Save trained models and vectorizer to disk.
    
    Args:
        models (dict): Dictionary of trained models
        vectorizer: Fitted vectorizer
        save_dir (str): Directory to save models
    """
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save each model
    for model_name, model in models.items():
        filename = model_name.lower().replace(' ', '_') + '.pkl'
        filepath = os.path.join(save_dir, filename)
        joblib.dump(model, filepath)
        print(f"  ✓ Saved: {filepath}")
    
    # Save vectorizer
    vectorizer_path = os.path.join(save_dir, 'vectorizer.pkl')
    joblib.dump(vectorizer, vectorizer_path)
    print(f"  ✓ Saved: {vectorizer_path}")
    
    print("\n✓ All models saved successfully!")


def main():
    """
    Main execution function.
    """
    print("\n" + "="*60)
    print("SPAM EMAIL CLASSIFIER - PHASE 1")
    print("Multi-Model Baseline Implementation")
    print("="*60)
    
    # Step 1: Download and load data
    dataset_path = os.path.join(DATA_DIR, 'sms_spam_no_header.csv')
    if not os.path.exists(dataset_path):
        os.makedirs(DATA_DIR, exist_ok=True)
        if not download_dataset(DATASET_URL, dataset_path):
            print("Failed to download dataset. Exiting...")
            return
    
    df = load_and_explore_data(dataset_path)
    
    # Step 2: Preprocess data
    df = preprocess_text(df)
    
    # Step 3: Split data
    print("\n" + "="*60)
    print("TRAIN-TEST SPLIT")
    print("="*60)
    
    X = df['message']
    y = df['label_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"✓ Data split complete")
    print(f"  Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Testing samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    print(f"  Training class distribution:\n{y_train.value_counts()}")
    
    # Step 4: Create features
    X_train_vec, X_test_vec, vectorizer = create_features(X_train, X_test, vectorizer_type='tfidf')
    
    # Step 5: Train models
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'Support Vector Machine': SVC(kernel='linear', random_state=RANDOM_STATE, probability=True)
    }
    
    trained_models = {}
    training_times = {}
    
    for model_name, model in models.items():
        trained_model, train_time = train_model(model, model_name, X_train_vec, y_train)
        trained_models[model_name] = trained_model
        training_times[model_name] = train_time
    
    # Step 6: Evaluate models
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    results = {}
    for model_name, model in trained_models.items():
        print(f"\n  Evaluating {model_name}...")
        metrics = evaluate_model(model, model_name, X_test_vec, y_test)
        results[model_name] = metrics
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Step 7: Print summary
    print_evaluation_summary(results, training_times)
    
    # Step 8: Create visualizations
    create_visualizations(results, training_times, CHARTS_DIR)
    
    # Step 9: Save models
    save_models(trained_models, vectorizer, MODELS_DIR)
    
    # Step 10: Test prediction example
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    
    test_messages = [
        "Congratulations! You've won a free iPhone. Click here to claim now!",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your account will be suspended. Verify your identity immediately!",
        "Thanks for the meeting notes. I'll review them tonight."
    ]
    
    print("\nTesting predictions with saved models:")
    for msg in test_messages:
        msg_vec = vectorizer.transform([msg.lower()])
        
        print(f"\nMessage: \"{msg}\"")
        for model_name, model in trained_models.items():
            pred = model.predict(msg_vec)[0]
            pred_label = "SPAM" if pred == 1 else "HAM"
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(msg_vec)[0]
                confidence = proba[pred] * 100
                print(f"  {model_name}: {pred_label} (confidence: {confidence:.1f}%)")
            else:
                print(f"  {model_name}: {pred_label}")
    
    print("\n" + "="*60)
    print("✓ PHASE 1 IMPLEMENTATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review generated charts in the 'charts/' directory")
    print("2. Check saved models in the 'models/' directory")
    print("3. Run the Streamlit app: streamlit run streamlit_app.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

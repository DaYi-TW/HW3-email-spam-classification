# Project Context

## Purpose
Cybersecurity coursework project (HW3) focused on building spam email classification system using machine learning techniques. The goal is to develop, evaluate, and iterate on ML models for detecting spam emails, progressing through multiple phases from baseline to advanced implementations.

## Tech Stack
- **Python 3.x** - Primary programming language
- **scikit-learn** - Machine learning library (Logistic Regression, Naïve Bayes, SVM)
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization and chart generation
- **seaborn** - Statistical data visualization
- **streamlit** - Interactive web UI for model deployment
- **nltk or spaCy** - Natural language processing (tokenization, optional)

## Project Conventions

### Code Style
- Follow PEP 8 Python style guide
- Use descriptive variable names (e.g., `X_train`, `y_test`, `spam_classifier`)
- Add docstrings to functions and classes
- Keep functions focused and under 50 lines where possible
- Use type hints for function signatures

### Architecture Patterns
- **Phased development**: Implement features in phases (Phase 1: multi-model baseline with UI, Phase 2+: enhancements)
- **ML folder organization**: All code in `ml/` directory with subdirectories for models, data, and charts
- **Data pipeline**: Load → Clean → Tokenize → Vectorize → Train → Evaluate → Visualize → Save
- **Multi-model comparison**: Train and compare Logistic Regression, Naïve Bayes, and SVM
- **Interactive UI**: Streamlit web application for real-time predictions and model comparison
- **Reproducibility**: Set random seeds, save models and vectorizers for consistent results

### Testing Strategy
- **Manual validation**: Test with sample emails and verify classification accuracy
- **Metrics-based evaluation**: Track accuracy, precision, recall, F1-score
- **Train-test split**: Use standard 80/20 or similar split for validation
- **Future**: Add unit tests for preprocessing and feature extraction functions as code grows

### Git Workflow
- Work on master branch for coursework
- Commit after each phase completion
- Use descriptive commit messages: "Add Phase 1 baseline SVM classifier"

## Domain Context
**Spam Email Classification**
- **Spam vs Ham**: Binary classification problem (spam = unwanted, ham = legitimate)
- **SMS dataset**: Using SMS spam dataset as proxy for email spam (similar characteristics)
- **Feature engineering**: Text needs to be converted to numerical features (TF-IDF, bag-of-words, etc.)
- **Class imbalance**: Spam datasets often have more ham than spam; may need balancing techniques
- **Evaluation priority**: Balance precision (avoid false positives blocking legitimate emails) and recall (catch actual spam)

**Machine Learning Approach**
- **Phase 1**: Multi-model comparison (Logistic Regression, Naïve Bayes, SVM)
- **Evaluation**: Comprehensive metrics with visualizations (accuracy, precision, recall, F1, ROC curves)
- **Deployment**: Streamlit web interface for interactive spam detection
- **Future phases**: Deep learning models (LSTM, BERT), ensemble methods, hyperparameter optimization
- **Iterative improvement**: Each phase builds on previous learnings

## Important Constraints
- Academic project with deadline constraints
- Dataset source must be from official course repository
- Model must be reproducible (set random seeds)
- Keep implementations simple and understandable for educational purposes

## External Dependencies
- **Dataset source**: https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv
- **scikit-learn documentation**: For ML algorithm references
- No cloud services or external APIs required

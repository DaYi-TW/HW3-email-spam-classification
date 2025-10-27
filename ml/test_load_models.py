import os
import joblib
import numpy as np

MODELS_DIR = os.path.join('ml', 'models')


def load_vectorizer():
    path = os.path.join(MODELS_DIR, 'vectorizer.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)


def load_models():
    models = {}
    for fname in os.listdir(MODELS_DIR):
        if fname.endswith('.pkl') and fname != 'vectorizer.pkl':
            name = fname.replace('.pkl', '')
            models[name] = joblib.load(os.path.join(MODELS_DIR, fname))
    return models


def main():
    print('Loading artifacts from ml/models...')
    vec = load_vectorizer()
    models = load_models()
    print(f'Loaded vectorizer and {len(models)} models: {list(models.keys())}')

    texts = [
        "Congratulations! You've won a free iPhone. Click here to claim now!",
        "Hey, are we still meeting for lunch tomorrow?",
    ]

    X = vec.transform([t.lower() for t in texts])

    for name, model in models.items():
        print('\nModel:', name)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            preds = model.predict(X)
            for t, p, pred in zip(texts, proba, preds):
                print(f'  Text: {t}\n    Pred: {pred} (prob: {p})')
        else:
            preds = model.predict(X)
            for t, pred in zip(texts, preds):
                print(f'  Text: {t}\n    Pred: {pred}')


if __name__ == '__main__':
    main()

import os
import re
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Constants
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models')
CHARTS_DIR = os.path.join(BASE_DIR, 'charts')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'sms_spam_no_header.csv')


def safe_load_models():
    """Load models and vectorizer from ml/models."""
    models = {}
    vectorizer = None
    if not os.path.exists(MODELS_DIR):
        return models, None

    # known filenames (from training script)
    mapping = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Support Vector Machine': 'support_vector_machine.pkl'
    }

    for name, fname in mapping.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)

    vec_path = os.path.join(MODELS_DIR, 'vectorizer.pkl')
    if os.path.exists(vec_path):
        vectorizer = joblib.load(vec_path)

    return models, vectorizer


@st.cache_data
def load_and_preprocess_dataset(limit=None):
    """Load the SMS dataset and apply the same light preprocessing used in training."""
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()

    # file uses quoted CSV with comma separator
    df = pd.read_csv(DATA_PATH, header=None, names=['label', 'message'], sep=',', quotechar='"', encoding='utf-8')

    # basic cleaning to match training
    df['message'] = df['message'].astype(str).fillna('')
    df['message'] = df['message'].str.lower()
    df['message'] = df['message'].str.replace(r'[^a-z\s]', '', regex=True)
    df['message'] = df['message'].str.replace(r'\s+', ' ', regex=True).str.strip()
    df = df[df['message'].str.len() > 0].copy()
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['label_encoded'] = df['label'].map({'spam': 1, 'ham': 0})
    df = df.dropna(subset=['label_encoded'])

    if limit is not None:
        return df.sample(n=min(limit, len(df)), random_state=42)

    return df


@st.cache_data
def compute_performance(_models, _vectorizer, sample_size=2000):
    """Compute accuracy/precision/recall/f1 for each model using a held-out sample of the dataset."""
    if not _models or _vectorizer is None:
        return {}

    df = load_and_preprocess_dataset(limit=sample_size)
    if df.empty:
        return {}

    X = _vectorizer.transform(df['message'])
    y = df['label_encoded'].astype(int).values

    perf = {}
    for name, model in _models.items():
        try:
            y_pred = model.predict(X)
            perf[name] = {
                'accuracy': float(accuracy_score(y, y_pred)),
                'precision': float(precision_score(y, y_pred, zero_division=0)),
                'recall': float(recall_score(y, y_pred, zero_division=0)),
                'f1': float(f1_score(y, y_pred, zero_division=0))
            }
        except Exception as e:
            perf[name] = {'error': str(e)}

    return perf


def predict(message, model, vectorizer):
    msg = str(message).lower()
    vec = vectorizer.transform([msg])
    pred = model.predict(vec)[0]
    proba = None
    conf = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(vec)[0]
        conf = float(proba[pred])
    return int(pred), conf, proba


def main():
    st.set_page_config(page_title='Spam Classifier', layout='wide', page_icon='📧')
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .spam-box {
        padding: 20px;
        border-radius: 10px;
        font-size: 1.3rem;
        font-weight: 600;
        text-align: center;
        margin: 15px 0;
        background-color: #ffebee;
        color: #c62828;
        border: 3px solid #c62828;
    }
    .ham-box {
        padding: 20px;
        border-radius: 10px;
        font-size: 1.3rem;
        font-weight: 600;
        text-align: center;
        margin: 15px 0;
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 3px solid #2e7d32;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1565c0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">📧 垃圾郵件分類器</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">機器學習垃圾郵件/簡訊分類系統</p>', unsafe_allow_html=True)

    models, vectorizer = safe_load_models()

    if not models or vectorizer is None:
        st.error('⚠️ 模型檔案未找到！請先執行訓練腳本: `python ml\\spam_classifier.py`')
        return

    perf = compute_performance(models, vectorizer, sample_size=2000)

    left, right = st.columns([2.5, 1.5])

    with left:
        st.markdown('###  輸入訊息')
        
        # Initialize session state
        if 'message_text' not in st.session_state:
            st.session_state['message_text'] = ''

        # 範例訊息
        with st.expander("💡 試試範例訊息"):
            st.markdown("**🚨 垃圾郵件範例**")
            col_spam1, col_spam2 = st.columns(2)

            with col_spam1:
                if st.button("🎁 中獎通知", use_container_width=True, key='spam1'):
                    st.session_state.message_text = "Dear user,\nYou've been selected as our lucky winner! Click below to claim your $1,000 Amazon gift card now.\n\n[Claim Reward Now]\n\nHurry! This offer expires in 24 hours.\n\nNote: You must complete the survey to receive your reward."
                if st.button("⚠️ 假銀行警告", use_container_width=True, key='spam2'):
                    st.session_state.message_text = "Dear Customer,\nWe have detected unusual activity in your bank account. Please verify your information immediately to restore access.\n\nClick here to confirm your account: [Fake Bank Link]\n\nThank you for your prompt attention.\n\n— Security Department"

            with col_spam2:
                if st.button("💰 投資詐騙", use_container_width=True, key='spam3'):
                    st.session_state.message_text = "Hi there,\nYou can now make $10,000 per week from the comfort of your home!\n\nOur automated crypto trading bot guarantees 99% accuracy. Start with just $250 today.\n\n[Join Now]"
                if st.button("💼 工作詐騙", use_container_width=True, key='spam4'):
                    st.session_state.message_text = "We're hiring remote workers now!\nNo experience needed — just an internet connection.\n\nClick here to apply: [Suspicious Link]\n\nLimited positions available!"

            st.markdown("---")
            st.markdown("**✅ 正常郵件範例**")
            col_ham1, col_ham2 = st.columns(2)

            with col_ham1:
                if st.button("📅 會議提醒", use_container_width=True, key='ham1'):
                    st.session_state.message_text = "Hi team,\nJust a quick reminder that our Project Alpha sync is scheduled for tomorrow at 10 AM in Meeting Room B.\n\nAgenda:\n- Review current milestones\n- Discuss upcoming release\n- Assign QA testing tasks\n\nThanks,\nSarah"
                if st.button("📦 訂單確認", use_container_width=True, key='ham2'):
                    st.session_state.message_text = "Dear Mr. Lee,\nThank you for shopping with us!\n\nYour order #48327 has been successfully placed.\nEstimated delivery date: November 2, 2025\n\nYou can track your shipment [here].\n\nBest regards,\nThe Store Team"

            with col_ham2:
                if st.button("🦷 預約通知", use_container_width=True, key='ham3'):
                    st.session_state.message_text = "Hello,\nThis is to confirm your dental checkup appointment with Dr. Chen on Monday, October 30 at 3:00 PM.\n\nLocation: SmileCare Clinic, 2nd Floor, Main Building\n\nPlease arrive 10 minutes early.\n\n— SmileCare Clinic Team"
                if st.button("🧾 HR 通知", use_container_width=True, key='ham4'):
                    st.session_state.message_text = "Dear all,\nWe've updated our annual leave policy effective January 2026.\n\nKey changes include:\n- Unused leave can now be carried over up to 10 days.\n- Leave requests must be submitted at least one week in advance.\n\nPlease review the updated policy on the HR portal.\n\nRegards,\nHR Department"
        
        st.markdown('')
        
        # text_area 使用 value 而不是 key 綁定
        message = st.text_area(
            '請輸入郵件或簡訊內容:',
            height=200,
            placeholder='在此輸入或貼上訊息內容...',
            label_visibility='collapsed',
            value=st.session_state.message_text
        )
        
        st.markdown('---')
        classify_btn = st.button('🔍 開始分類', type='primary', use_container_width=True)

        if classify_btn:
            message = st.session_state.test_msg
            if not message:
                st.warning('⚠️ 請輸入訊息內容')
            else:
                st.markdown('---')
                st.markdown('### 🎯 分類結果')
                selected = st.session_state.get('selected_model', 'All models')
                if selected == 'All models':
                    cols = st.columns(len(models))
                    for i, (name, model) in enumerate(models.items()):
                        with cols[i]:
                            pred, conf, proba = predict(message, model, vectorizer)
                            label = '🚨 SPAM' if pred == 1 else '✅ HAM'
                            box_class = 'spam-box' if pred == 1 else 'ham-box'
                            st.markdown(f'<div class="{box_class}">{label}</div>', unsafe_allow_html=True)
                            st.markdown(f'**{name}**')
                            if conf is not None:
                                st.metric('信心度', f'{conf*100:.1f}%')
                else:
                    model = models[selected]
                    pred, conf, proba = predict(message, model, vectorizer)
                    label = '🚨 SPAM' if pred == 1 else '✅ HAM'
                    box_class = 'spam-box' if pred == 1 else 'ham-box'
                    st.markdown(f'<div class="{box_class}">{label}</div>', unsafe_allow_html=True)
                    
                    if conf is not None:
                        st.markdown(f'### 信心度: {conf*100:.1f}%')
                        st.progress(conf)
                        
                        if proba is not None:
                            col_p1, col_p2 = st.columns(2)
                            with col_p1:
                                st.metric('正常郵件機率', f'{proba[0]*100:.1f}%')
                            with col_p2:
                                st.metric('垃圾郵件機率', f'{proba[1]*100:.1f}%')

    with right:
        st.markdown('### 🤖 選擇模型')
        options = ['All models'] + list(models.keys())
        sel = st.selectbox('模型選擇', options, key='selected_model', label_visibility='collapsed')

        st.markdown('---')
        st.markdown('### 📊 模型效能')
        if sel == 'All models':
            st.info('💡 選擇單一模型以查看詳細效能指標')
            rows = []
            for name in models.keys():
                m = perf.get(name, {})
                if 'error' in m:
                    rows.append({'模型': name, '準確率': 'error'})
                else:
                    rows.append({
                        '模型': name,
                        '準確率': f"{m['accuracy']:.3f}",
                        '精確率': f"{m['precision']:.3f}",
                        '召回率': f"{m['recall']:.3f}",
                        'F1': f"{m['f1']:.3f}"
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            m = perf.get(sel)
            if not m:
                st.warning('效能資料不可用')
            elif 'error' in m:
                st.error(f'計算錯誤: {m["error"]}')
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric('準確率', f"{m['accuracy']:.1%}")
                    st.metric('精確率', f"{m['precision']:.1%}")
                with col2:
                    st.metric('召回率', f"{m['recall']:.1%}")
                    st.metric('F1 分數', f"{m['f1']:.1%}")

        st.markdown('---')
        st.markdown('### 📈 視覺化圖表')
        
        chart_files = {
            'accuracy_comparison.png': '準確率比較',
            'metrics_comparison.png': '效能指標比較',
            'confusion_matrices.png': '混淆矩陣',
            'roc_curves.png': 'ROC 曲線',
            'training_time.png': '訓練時間'
        }
        
        chart_found = False
        for fname, title in chart_files.items():
            p = os.path.join(CHARTS_DIR, fname)
            if os.path.exists(p):
                chart_found = True
                with st.expander(f"📊 {title}"):
                    st.image(p, use_column_width=True)
        
        if not chart_found:
            st.info('📊 執行訓練腳本以生成圖表')


if __name__ == '__main__':
    main()

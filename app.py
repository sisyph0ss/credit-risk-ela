import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time

# Sayfa Yapılandırması
st.set_page_config(page_title="Kredi Riski Karar Destek Sistemi", page_icon="🏦", layout="wide")

# Modern Tasarım ve Renkler için Özel CSS
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .risk-low {
        color: #2e7d32;
        background-color: #c8e6c9;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #2e7d32;
    }
    .risk-medium {
        color: #f57c00;
        background-color: #ffe0b2;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #f57c00;
    }
    .risk-high {
        color: #c62828;
        background-color: #ffcdd2;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #c62828;
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    </style>
""", unsafe_allow_html=True)

# 1. Veri Yükleme ve Ön İşleme
@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_csv("credit_risk_dataset.csv")
    except FileNotFoundError:
        st.error("Hata: 'credit_risk_dataset.csv' dosyası bulunamadı. Lütfen dosyanın aynı dizinde olduğundan emin olun.")
        st.stop()

    # Aykırı değerleri temizleme (NaN değerleri koruyarak filtreleme yapıyoruz)
    df = df[(df['person_age'] <= 100) | (df['person_age'].isna())]
    df = df[(df['person_emp_length'] <= 60) | (df['person_emp_length'].isna())]

    # Eksik veri doldurma (Imputation)
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].mean())

    return df

# 2. Model Eğitimi ve Karşılaştırma
@st.cache_resource
def train_and_evaluate_models(df):
    # Kategorik verileri dönüştürme (One-Hot Encoding)
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Özellikler ve Hedef
    X = df_encoded.drop('loan_status', axis=1)
    y = df_encoded['loan_status']

    # Eğitim ve Test Ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Modelleri Tanımlama
    models = {
        "Lojistik Regresyon": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metriklerin hesaplanması (Hocanın belirttiği öncelik sırası)
        roc_auc = roc_auc_score(y_test, y_prob)
        rec = recall_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        
        results[name] = {
            "ROC-AUC": roc_auc,
            "Recall": rec,
            "Precision": prec,
            "F1-Score": f1,
            "Accuracy": acc
        }
        trained_models[name] = model

    # Çok Kriterli Karar Verme (MCDM) - Ağırlıklı Toplam Yöntemi
    # Hocanın belirttiği öncelik sırasına göre ağırlıklar:
    # 1. ROC-AUC (Ağırlık: 5)
    # 2. Recall (Ağırlık: 4)
    # 3. Precision (Ağırlık: 3)
    # 4. F1-Score (Ağırlık: 2)
    # 5. Accuracy (Ağırlık: 1)
    weights = {
        "ROC-AUC": 5.0,
        "Recall": 4.0,
        "Precision": 3.0,
        "F1-Score": 2.0,
        "Accuracy": 1.0
    }
    
    model_scores = {}
    for name, metrics in results.items():
        # Ağırlıklı toplam skoru hesapla
        total_score = (
            weights["ROC-AUC"] * metrics["ROC-AUC"] +
            weights["Recall"] * metrics["Recall"] +
            weights["Precision"] * metrics["Precision"] +
            weights["F1-Score"] * metrics["F1-Score"] +
            weights["Accuracy"] * metrics["Accuracy"]
        )
        # 100 üzerinden normalize et (Toplam ağırlık: 5+4+3+2+1 = 15)
        normalized_score = (total_score / 15.0) * 100
        results[name]["Total-Score"] = normalized_score
        model_scores[name] = normalized_score

    # En yüksek toplam puanı alan modeli seç
    best_model_name = max(model_scores, key=model_scores.get)
    best_model = trained_models[best_model_name]
    feature_columns = X.columns.tolist()

    return best_model, best_model_name, results, feature_columns



# Başlık ve Açıklama
st.title("🏦 Akıllı Kredi Karar Destek Sistemi")
st.markdown("**Regresyon Modelleri İle Tahmin Edici Analiz Ve Karşılaştırmalı Karar Destek Yaklaşımı**")
st.write("Bu uygulama, müşteri verilerini analiz ederek kredi onay sürecini otomatize eder ve riski öngörür.")

with st.spinner('Veri seti yükleniyor ve modeller eğitiliyor. Lütfen bekleyin...'):
    df = load_and_preprocess_data()
    best_model, best_model_name, model_results, feature_columns = train_and_evaluate_models(df)

# Ekranda model performanslarını gösterme
st.sidebar.header("📊 Model Performansları")
st.sidebar.info("Modeller test seti üzerinde değerlendirilmiş ve belirlenen kriter öncelik sırasına göre **Ağırlıklı Çok Kriterli Karar Verme (MCDM)** yöntemiyle puanlanmıştır. En iyi model bu toplam skora göre seçilmiştir.")


for model_name, metrics in model_results.items():
    score_val = metrics["Total-Score"]
    if model_name == best_model_name:
        st.sidebar.success(f"🏆 {model_name} (Seçilen) \n\n **Toplam Skor: %{score_val:.2f}**")
    else:
        st.sidebar.subheader(f"🔹 {model_name}")
        st.sidebar.markdown(f"**Toplam Skor: %{score_val:.2f}**")
        
    st.sidebar.markdown(f"""
    1. **ROC-AUC (Ağırlık: 5):** `{metrics['ROC-AUC']:.4f}`
    2. **Recall (Ağırlık: 4):** `%{metrics['Recall']*100:.2f}`
    3. **Precision (Ağırlık: 3):** `%{metrics['Precision']*100:.2f}`
    4. **F1-Score (Ağırlık: 2):** `%{metrics['F1-Score']*100:.2f}`
    5. **Accuracy (Ağırlık: 1):** `%{metrics['Accuracy']*100:.2f}`
    """)
    st.sidebar.markdown("---")

st.sidebar.markdown(f"**Karar Verici Motor:** {best_model_name}")



# Kullanıcı Arayüzü - Veri Girişi
st.header("👤 Müşteri Bilgileri")

col1, col2, col3 = st.columns(3)

with col1:
    person_age = st.number_input("Yaş", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Yıllık Gelir ($)", min_value=0, value=50000, step=1000)
    person_emp_length = st.number_input("Çalışma Süresi (Yıl)", min_value=0.0, max_value=60.0, value=5.0)
    cb_person_cred_hist_length = st.number_input("Kredi Geçmişi Uzunluğu (Yıl)", min_value=0, max_value=50, value=5)

with col2:
    loan_amnt = st.number_input("Kredi Miktarı ($)", min_value=500, value=10000, step=500)
    loan_int_rate = st.number_input("Faiz Oranı (%)", min_value=1.0, max_value=30.0, value=10.0, step=0.1)
    loan_percent_income = loan_amnt / person_income if person_income > 0 else 0.0
    st.info(f"Gelir/Kredi Oranı: **{loan_percent_income:.2f}**")

with col3:
    person_home_ownership = st.selectbox("Ev Sahipliği Durumu", options=df['person_home_ownership'].unique())
    loan_intent = st.selectbox("Kredi Amacı", options=df['loan_intent'].unique())
    loan_grade = st.selectbox("Kredi Notu (Grade)", options=sorted(df['loan_grade'].unique()))
    cb_person_default_on_file = st.selectbox("Geçmişte Temerrüde Düştü mü?", options=['N', 'Y'])

# Analiz Butonu
st.markdown("---")
analyze_button = st.button("🔍 Riski Analiz Et ve Karar Ver", use_container_width=True)

if analyze_button:
    # Girdiyi DataFrame'e çevirme
    input_data = {
        'person_age': person_age,
        'person_income': person_income,
        'person_emp_length': person_emp_length,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'person_home_ownership': person_home_ownership,
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'cb_person_default_on_file': cb_person_default_on_file
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Categorical variable encoding with same schema
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols)
    
    # Modelin beklediği kolonları oluştur ve eksik kolonları 0 ile doldur
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
            
    # Model eğitimindeki sütun sırasını eşle
    input_encoded = input_encoded[feature_columns]
    
    # Tahmin
    with st.spinner("Analiz ediliyor..."):
        time.sleep(1) # Daha iyi kullanıcı deneyimi için küçük bir bekleme
        risk_probability = best_model.predict_proba(input_encoded)[0][1] * 100 # Sınıf 1 (Temerrüt/Risk) olasılığı
        
    st.markdown("---")
    st.header("🎯 Analiz Sonucu")
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric(label="Hesaplanan Risk Olasılığı", value=f"% {risk_probability:.2f}")
        
    with res_col2:
        # Karar Destek Mekanizması Kuralları
        if risk_probability < 25:
            st.markdown("""
                <div class="risk-low">
                    <h2>✅ KREDİ ONAYLANDI</h2>
                    <p style="font-size: 18px;"><b>Risk Durumu: Düşük Risk</b></p>
                    <p>Müşterinin temerrüde düşme olasılığı çok düşüktür. Kredi otomatik olarak onaylanabilir.</p>
                </div>
            """, unsafe_allow_html=True)
            if risk_probability > 0:
                st.balloons()
        elif 25 <= risk_probability <= 60:
            st.markdown("""
                <div class="risk-medium">
                    <h2>⚠️ EK TEMİNAT / MANUEL İNCELEME</h2>
                    <p style="font-size: 18px;"><b>Risk Durumu: Orta Risk</b></p>
                    <p>Sınırda risk faktörleri tespit edildi. Kredi tahsisi için manuel analist onayı ve/veya ek teminat talep edilmelidir.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="risk-high">
                    <h2>❌ KREDİ REDDEDİLDİ</h2>
                    <p style="font-size: 18px;"><b>Risk Durumu: Yüksek Risk</b></p>
                    <p>Müşterinin geri ödememe olasılığı tolere edilebilir seviyenin üzerindedir. Kredi talebi reddedilmelidir.</p>
                </div>
            """, unsafe_allow_html=True)

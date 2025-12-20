
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ==============================
# KONFIGURASI HALAMAN
# ==============================
st.set_page_config(
    page_title="Phishing Detector EAS",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# LOAD MODEL
# ==============================
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

# ==============================
# FITUR (HARUS SAMA DENGAN TRAIN)
# ==============================
features = [
    'google_index',
    'web_traffic',
    'domain_age',
    'ratio_extHyperlinks',
    'ratio_intHyperlinks',
    'nb_hyperlinks',
    'page_rank',
    'safe_anchor',
    'domain_registration_length',
    'links_in_tags'
]

# ==============================
# HEADER
# ==============================
st.title("🛡️ Phishing Detection System")
st.caption("Model: Random Forest + InfoGain (RAW Features)")
st.markdown("---")

tab1, tab2 = st.tabs(["🧮 Input Manual", "📂 Upload CSV"])

# =====================================================
# TAB 1 — INPUT MANUAL (REALTIME)
# =====================================================
with tab1:
    st.subheader("🔢 Masukkan Nilai Fitur")

    col1, col2 = st.columns(2)
    values = {}

    for i, feat in enumerate(features):
        with col1 if i < 5 else col2:
            values[feat] = st.number_input(
                label=feat,
                value=0.0,
                step=0.1
            )

    input_df = pd.DataFrame([values])

    if st.button("🔍 Prediksi"):
        proba = model.predict_proba(input_df)[0, 1]
        pred = int(proba >= 0.5)

        st.markdown("---")
        if pred == 1:
            st.error("🚨 **HASIL: PHISHING**")
        else:
            st.success("✅ **HASIL: LEGIT (AMAN)**")

        st.metric(
            label="Confidence (Phishing)",
            value=f"{proba * 100:.2f}%"
        )

# =====================================================
# TAB 2 — CSV (REALTIME PREDICTION)
# =====================================================
with tab2:
    st.subheader("📊 Prediksi Massal dari CSV")

    uploaded_file = st.file_uploader(
        "Upload file CSV (harus mengandung 10 fitur)",
        type=["csv"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if not all(f in df.columns for f in features):
            st.error("❌ Kolom CSV tidak sesuai.")
            st.write("Kolom yang dibutuhkan:", features)
        else:
            st.success("✅ File valid. Prediksi diproses secara realtime.")

            X = df[features].copy()

            # === PREDIKSI LANGSUNG (REALTIME) ===
            proba = model.predict_proba(X)[:, 1]
            pred = (proba >= 0.5).astype(int)

            df_result = df.copy()
            df_result["Prediction"] = np.where(pred == 1, "Phishing", "Legit")
            df_result["Confidence (%)"] = (proba * 100).round(2)
            
            # =========================
            # EVALUASI JIKA ADA LABEL
            # =========================
            if "status" in df.columns:
                y_true = (df["status"] == "phishing").astype(int).values

                accuracy = (y_true == pred).mean()
                phishing_acc = ((pred[y_true == 1] == 1).mean() * 100) if (y_true == 1).any() else 0
                legit_acc = ((pred[y_true == 0] == 0).mean() * 100) if (y_true == 0).any() else 0

                st.markdown("### 📈 Evaluasi Model (CSV Berlabel)")

                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy", f"{accuracy*100:.2f}%")
                c2.metric("Recall Phishing", f"{phishing_acc:.2f}%")
                c3.metric("Recall Legit", f"{legit_acc:.2f}%")


            # === RINGKASAN ===
            colA, colB, colC = st.columns(3)
            colA.metric("Total Data", len(df_result))
            colB.metric("Phishing", int((pred == 1).sum()))
            colC.metric("Legit", int((pred == 0).sum()))

            st.markdown("### 🔍 Preview Hasil")
            st.dataframe(df_result.head(20), use_container_width=True)

            # === DOWNLOAD ===
            csv = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Hasil Prediksi",
                csv,
                file_name="hasil_prediksi_phishing.csv",
                mime="text/csv"
            )

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.info("""
**Pengenalan Pola (C-081)**  

**Dosen Pengampu:**  
1. Fetty Tri Anggraeny, S.Kom., M.Kom.  
2. Dina Zatusiva Haq  

**Anggota Kelompok:**  
1. Indah Ayu Putri Mashita Cahyani (22081010048)  
2. Dela Puspita Lasminingrum (22081010209)  
3. Angie Nurshabrina Putri (22081010254)  
""")

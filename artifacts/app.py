import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- KONFIGURASI ---
st.set_page_config(page_title="Phishing Detector UTS", layout="wide")

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    # Pastikan file ini namanya 'model.pkl' di folder kamu
    return joblib.load('model.pkl')

model = load_model()

# Daftar 10 fitur terbaik dari Eksperimen A
features = [
    'google_index', 'web_traffic', 'domain_age', 'ratio_extHyperlinks', 
    'ratio_intHyperlinks', 'nb_hyperlinks', 'page_rank', 'safe_anchor', 
    'domain_registration_length', 'links_in_tags'
]

# --- TAMPILAN ---
st.title("🛡️ Phishing Detection ")
st.write(f"Menggunakan 10 Fitur Terbaik (InfoGain RAW) - Akurasi: 94.4%")

tab1, tab2 = st.tabs(["Input Manual", "Upload CSV"])

with tab1:
    st.subheader("Masukkan Nilai Fitur")
    
    # Membuat 2 kolom agar tampilan input rapi
    col1, col2 = st.columns(2)
    
    inputs = []
    for i, feat in enumerate(features):
        with col1 if i < 5 else col2:
            val = st.number_input(f"{feat}", value=0.0, step=0.1)
            inputs.append(val)
    
    if st.button("Prediksi Satuan"):
        # Proses Prediksi
        input_array = np.array([inputs])
        prediction = model.predict(input_array)
        probability = model.predict_proba(input_array)
        
        st.divider()
        if prediction[0] == 1:
            st.error(f"HASIL: **PHISHING** 🚩")
        else:
            st.success(f"HASIL: **LEGIT (AMAN)** ✅")
        st.write(f"Tingkat Keyakinan: {np.max(probability)*100:.2f}%")

with tab2:
    st.subheader("Prediksi Massal via CSV")
    uploaded_file = st.file_uploader("Upload CSV (Pastikan ada 10 kolom fitur)", type="csv")
    
    if uploaded_file:
        df_test = pd.read_csv(uploaded_file)
        
        # Cek apakah 10 kolom ada di CSV
        if all(col in df_test.columns for col in features):
            st.write("Data Berhasil Dimuat:")
            st.dataframe(df_test[features].head())
            
            if st.button("Proses Seluruh Data"):
                preds = model.predict(df_test[features])
                probs = model.predict_proba(df_test[features]).max(axis=1)
                
                df_test['Status'] = ["Phishing" if p == 1 else "Legit" for p in preds]
                df_test['Confidence'] = probs
                
                st.success("Selesai!")
                st.dataframe(df_test)
                
                # Fitur Download Hasil
                csv = df_test.to_csv(index=False).encode('utf-8')
                st.download_button("Download Hasil (.csv)", data=csv, file_name="hasil_prediksi_phishing.csv")
        else:
            st.error(f"CSV harus memiliki kolom: {', '.join(features)}")


            
st.divider()
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
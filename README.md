
# 🛡️ Phishing Detection System
### Information Gain (Top-K) + Random Forest

Sistem deteksi website phishing berbasis **Machine Learning** yang mengombinasikan seleksi fitur cerdas dengan klasifikasi presisi tinggi.

## 🌟 Fitur Utama
* **Feature Selection:** Menggunakan *Information Gain (Top-K)* untuk memilih fitur paling berpengaruh.
* **Core Engine:** Menggunakan algoritma *Random Forest* dengan fitur numerik RAW (tanpa diskretisasi).
* **Workflow Lengkap:** Mendukung training, prediksi massal via CSV, dan visualisasi interaktif.
* **Dashboard:** Interface ramah pengguna berbasis *Streamlit*.

---

## 📁 Struktur Folder
```text
phising_detection/
├─ README.md
├─ requirements.txt
├─ train.py               # Script training model
├─ predict.py             # Script inferensi/prediksi
├─ data/
│  └─ dataset phising raw.csv
└─ artifacts/              # Dibuat otomatis setelah training
   ├─ model.pkl            # Model Random Forest terlatuh
   ├─ selected_features.json
   ├─ feature_schema.json
   └─ config.json

```

---

## ⚙️ Instalasi

Disarankan menggunakan **Python 3.9** atau yang lebih baru.

1. Clone repositori ini atau download source code.
2. Instal dependensi yang diperlukan:
```bash
pip install -r requirements.txt

```



---

## 🚀 Cara Penggunaan

### 1. Training Model

Melatih model menggunakan fitur RAW dengan seleksi fitur berdasarkan skor Information Gain tertinggi.

```bash
python train.py --data "data/dataset phising raw.csv" --mode raw --top_k 10

```

**Parameter:**

* `--data` : Path lokasi dataset CSV.
* `--mode` : Gunakan `raw` (default untuk proyek ini).
* `--top_k` : Jumlah fitur terbaik yang ingin diambil.

**Output:**

* Metrik evaluasi (Accuracy, Precision, Recall, F1-score).
* Artefak model tersimpan di folder `artifacts/`.

### 2. Prediksi Data (CSV)

Gunakan model yang sudah dilatih untuk mengklasifikasi data baru dalam jumlah besar.

```bash
python predict.py --input "data/dataset phising raw.csv" --output "predictions.csv"

```

**Output:**

* File `predictions.csv` dengan kolom tambahan:
* `Prediction` : Hasil klasifikasi (Phishing / Legit).
* `Confidence` : Tingkat keyakinan model (0-1).


* Jika dataset input memiliki kolom `status`, sistem akan otomatis menampilkan laporan evaluasi performa.

---

## 🧾 Format Input & Fitur

Dataset harus berisi fitur numerik (`int` / `float`). Label opsional pada kolom `status` dengan nilai `legitimate` atau `phishing`.

**Contoh 10 Fitur Teratas (Top-K):**

1. `google_index`
2. `web_traffic`
3. `domain_age`
4. `ratio_extHyperlinks`
5. `ratio_intHyperlinks`
6. `nb_hyperlinks`
7. `page_rank`
8. `safe_anchor`
9. `domain_registration_length`
10. `links_in_tags`

> [!TIP]
> Daftar lengkap fitur yang terpilih dapat Anda cek pada file: `artifacts/selected_features.json`.

---

## 📊 Visualisasi Dashboard

Aplikasi ini dilengkapi dengan dashboard **Streamlit** untuk pengalaman yang lebih interaktif:

* Input fitur secara manual untuk cek URL tunggal.
* Upload CSV dan lihat prediksi secara real-time.
* Visualisasi grafik perbandingan Phishing vs Legit.

**Jalankan Streamlit:**

```bash
python -m streamlit run artifacts/app.py

```

---

```



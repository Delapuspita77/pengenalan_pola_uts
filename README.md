# Model Deteksi Phishing — InfoGain (Top-K) + Random Forest

Deteksi phishing dengan **seleksi fitur Information Gain (Top-K)** dan **Random Forest** sebagai **klasifier**. Siap dipakai untuk memprediksi CSV berisi **fitur numerik**.

---

## Struktur Folder

model deteksi phising/
├─ requirements.txt
├─ train.py
├─ predict.py
├─ data/
│ └─ dataset phising raw.csv
└─ artifacts/ # dibuat otomatis setelah training
├─ model.pkl
├─ kbins.pkl # hanya jika mode=discretized
├─ selected_features.json
├─ feature_schema.json
└─ config.json
---

## Instalasi

```bash
pip install -r requirements.txt

Disarankan Python 3.9+.

Cara Pakai Cepat
1) Training (membangun model)
python train.py --data "data/dataset phising raw.csv" --mode discretized --top_k 10


--mode: discretized (default) atau raw

Output: metrik holdout di terminal, artefak tersimpan di artifacts/

2) Prediksi (memakai model)
python predict.py --input "data/dataset phising raw.csv" --output "predictions.csv"


Output: predictions.csv berisi kolom proba_phishing (0..1) dan prediction (0/1)

Jika input punya kolom status, metrik evaluasi otomatis dicetak

Format Input

CSV berisi fitur numerik (int/float)

Label opsional: kolom status ∈ {legitimate, phishing}

Bila model dilatih dengan --mode discretized, beberapa fitur model berupa disc_*.

Script membangun disc_* otomatis dari kolom mentah (contoh umum):
page_rank, domain_age, ratio_intHyperlinks, links_in_tags, ratio_digits_url

Daftar fitur final yang dipakai model ada di artifacts/selected_features.json

Troubleshooting

FileNotFoundError (input CSV) → cek path. Uji cepat:

python predict.py --input "data/dataset phising raw.csv" --output "predictions.csv"


Kolom tidak ditemukan (mode discretized) → pastikan kolom mentah untuk membentuk disc_* tersedia

Metrik prediksi > metrik holdout → wajar bila memprediksi seluruh dataset (optimistis); patokan generalisasi = metrik holdout dari train.py

Lisensi

Untuk penggunaan akademik/riset. Mohon atribusi ke pemilik dataset asli.

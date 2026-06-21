# Deteksi & Koreksi Typo Bahasa Indonesia (Traditional ML)

Sistem untuk **mendeteksi** apakah sebuah kata/kalimat mengandung typo dan
**mengoreksinya**, dibangun **sepenuhnya dengan Machine Learning klasik** —
tanpa deep learning (tanpa neural network, transformer, RNN, LSTM, atau BERT).

## Pendekatan singkat
- **Deteksi level-KATA** — klasifikasi *kata valid* vs *typo* memakai TF-IDF
  n-gram karakter + model klasik (Logistic Regression, Complement Naive Bayes,
  Linear SVM, XGBoost — dibandingkan, diambil yang terbaik).
- **Deteksi level-KALIMAT** — klasifikasi *kalimat benar* vs *mengandung typo*
  memakai TF-IDF n-gram kata + karakter. Mampu menangkap *real-word error*.
- **Koreksi (non-DL)** — kandidat dicari dari kosakata benar dengan kombinasi
  *edit distance* (Damerau-Levenshtein), kemiripan *cosine* TF-IDF n-gram
  karakter, frekuensi kata, dan pola tetangga keyboard QWERTY.

## Struktur file
| File | Keterangan |
|------|------------|
| `typo_detection_correction.ipynb` | Notebook utama: EDA → fitur → pelatihan → evaluasi → simpan model → menu. |
| `typo_lib.py` | Pustaka inti: tokenizer, edit distance, kelas `TypoCorrector` (deteksi & koreksi). |
| `typo_app.py` | Aplikasi **menu terminal** untuk pengujian interaktif. |
| `streamlit_app.py` | Aplikasi **web (Streamlit)** untuk deteksi & koreksi interaktif. |
| `requirements.txt` | Daftar dependensi. |
| `models/` | Artefak hasil pelatihan (model, vectorizer, kamus) — dibuat oleh notebook. |
| `dataset/` | 18 file CSV (pasangan `kalimat_awal` ↔ `kalimat_salah`). |

## Cara menjalankan
1. **Pasang dependensi**
   ```bash
   pip install -r requirements.txt
   ```
2. **Latih model** — buka & jalankan seluruh sel notebook:
   ```bash
   jupyter notebook typo_detection_correction.ipynb
   ```
   Notebook akan membuat folder `models/` berisi artefak terlatih.
3. **Uji lewat menu terminal**
   ```bash
   python typo_app.py
   ```
   Masukkan kata/kalimat → sistem menampilkan status typo, confidence, dan saran koreksi.

## Menjalankan aplikasi web (Streamlit)
Pastikan folder `models/` sudah ada (langkah 2). Lalu:
```bash
streamlit run streamlit_app.py
```
Buka URL yang muncul (biasanya `http://localhost:8501`). Aplikasi menyediakan
kotak input, tombol contoh, status deteksi + confidence, dan saran koreksi top-k.

## Deploy ke Streamlit Community Cloud (gratis)
1. **Siapkan repository GitHub** (sertakan `streamlit_app.py`, `typo_lib.py`,
   `requirements.txt`, dan folder **`models/`** — model wajib ikut di-commit):
   ```bash
   git init
   git add streamlit_app.py typo_lib.py requirements.txt models/ .streamlit/ README.md
   git commit -m "Deploy aplikasi deteksi & koreksi typo"
   git branch -M main
   git remote add origin https://github.com/<user>/<repo>.git
   git push -u origin main
   ```
2. Buka <https://share.streamlit.io> → **New app** → pilih repo & branch `main`.
3. **Main file path**: `streamlit_app.py` → **Deploy**.

Catatan:
- `scikit-learn` dipin ke `==1.7.2` di `requirements.txt` agar artefak `models/`
  cocok dengan versi di cloud (tanpa peringatan versi).
- Untuk build cloud lebih ramping, blok `pandas/matplotlib/notebook/jupyter` di
  `requirements.txt` boleh dihapus (hanya dipakai untuk training di notebook,
  bukan oleh aplikasi web).
- File `dataset/` **tidak** dibutuhkan oleh aplikasi web (boleh tidak di-commit).

## Contoh
```
Input : Buni apa?
Status: ADA TYPO terdeteksi
Koreksi: Bunyi apa?

Input : Bunyi apa?
Status: TIDAK ADA typo  -> Tidak ada koreksi yang dibutuhkan.
```

## Keterbatasan (didokumentasikan di notebook)
- **Real-word error** (typo yang kebetulan kata valid, mis. *belajar* → *belanja*)
  sulit dideteksi dari kata tunggal; karena itu ditambahkan **model level-kalimat**.
- Koreksi terbatas pada kata yang pernah muncul pada `kalimat_awal` (kamus dataset).

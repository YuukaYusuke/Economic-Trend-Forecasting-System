# Economic Trend Forecasting System

Aplikasi Streamlit untuk analisis dan prediksi arah nilai tukar menggunakan LSTM.

## Kebutuhan sistem

- Python 3.10–3.11 (disarankan 3.11 untuk Streamlit Cloud)
- Windows / macOS / Linux
- Koneksi internet (API kurs live)

## Instalasi lokal

```bash
cd "Economic Trend Forecasting System"
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Rahasia (API key)

**Jangan** menaruh API key di kode sumber.

- Salin `.env.example` → `.env` dan isi `ALPHAVANTAGE_API_KEY`, **atau**
- Salin `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml` (lokal saja)

Tanpa key Alpha Vantage, aplikasi tetap jalan memakai **Frankfurter** (bulk/realtime); halaman prediksi pasangan tunggal bisa terbatas.

## Latih semua model (wajib sebelum deploy)

```bash
python train_models.py
python train_models.py --force   # latih ulang
```

Model disimpan di `models/`. Lihat `models/README.md`.

## Menjalankan aplikasi

```bash
streamlit run app.py
```

Bahasa default: **Basa Jawa (aksara Jawa)**. Pilih bahasa lain di sidebar.

---

## Deploy ke Streamlit Community Cloud

### 1. Push ke GitHub

Pastikan **tidak** ikut ter-commit:

- `.env`, `.streamlit/secrets.toml`
- API key di kode (sudah dihapus dari repo)

File `.gitignore` sudah disiapkan.

### 2. Buat app di [share.streamlit.io](https://share.streamlit.io)

| Pengaturan | Nilai |
|------------|--------|
| Main file | `app.py` |
| Python | 3.11 (`runtime.txt`) |
| Dependencies | `requirements.txt` |

### 3. Secrets (App settings → Secrets)

Salin dari `.streamlit/secrets.toml.example`:

```toml
ALPHAVANTAGE_API_KEY = "kunci_anda"

[app]
allow_runtime_training = false
```

### 4. Sertakan model di repo

```bash
python train_models.py
git add models/*.keras models/*_meta.pkl
git commit -m "Add trained LSTM models for deploy"
git push
```

Tanpa folder `models/` di repo dan `allow_runtime_training = false`, fitur prediksi menampilkan pesan agar melatih lokal lalu deploy ulang.

### 5. Rekomendasi keamanan

- Rotasi API key jika pernah ter-expose di commit lama
- `enableXsrfProtection` aktif di `.streamlit/config.toml`
- Disclaimer investasi sudah di UI (`disclaimer` i18n)

### Memori Cloud

TensorFlow membutuhkan RAM cukup besar. Jika build gagal, kurangi jumlah model yang di-commit atau upgrade plan Streamlit.

---

## Struktur folder

| Path | Fungsi |
|------|--------|
| `app.py` | Entry point Streamlit |
| `pages/` | Halaman multipage |
| `modules/` | LSTM, API, i18n, deploy |
| `locales/` | Terjemahan (13 bahasa + JV) |
| `data/dataset.csv` | Data historis |
| `models/` | Model LSTM (hasil `train_models.py`) |
| `.streamlit/` | Config & contoh secrets |

## Fitur

- Dashboard eksekutif, dark mode, 13 bahasa (default JV)
- Visualisasi Plotly, prediksi + keyakinan, export CSV/PDF
- Evaluasi LSTM vs MA, realtime heatmap, bandingkan & simulasi

## Disclaimer

Bukan saran investasi. Prediksi bersifat indikatif.

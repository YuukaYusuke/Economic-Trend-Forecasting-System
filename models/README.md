# Folder model LSTM

File di sini dihasilkan oleh:

```bash
python train_models.py
```

## Deploy Streamlit Cloud

1. Latih model **di komputer lokal** (lebih cepat).
2. Commit file `*.keras` dan `*_meta.pkl` ke repo **atau** gunakan [Git LFS](https://git-lfs.github.com/) jika ukuran besar.
3. Di Cloud, set `allow_runtime_training = false` di Secrets (lihat `.streamlit/secrets.toml.example`).
4. Tanpa file model di repo, halaman prediksi menampilkan pesan error — aplikasi tidak akan melatih otomatis di server.

Ukuran perkiraan: ~1–5 MB per mata uang (tergantung arsitektur).

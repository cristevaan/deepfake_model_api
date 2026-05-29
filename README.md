# Deepfake model API

API ini merupakan **model AI / ML** yang mampu mendeteksi anomali deepfake berupa **klasifikasi** pada data berupa **gambar atau foto**. Output dari model ini terdiri dari prediksi kelas, nilai confidence, probabilitas tiap kelas, dan penjelasan dari **_generative_ AI (Groq / OpenAI)**.

## Fitur

- Upload gambar melalui API
- Prediksi Real / Fake
- Confidence score
- Probability tiap kelas
- Penjelasan otomatis menggunakan Generative AI
- Dokumentasi API otomatis (Swagger UI)

## Tech Stack

- **Framework**: FastAPI
- **ML Model**: TensorFlow/Keras
- **Generative AI**: Groq API
- **Server Runtime**: Uvicorn
- **Deployment**: Railway

## Direktori

```
deepfake_model_api/
├── main.py             # main code
├── Procfile            # file configurasi server (Railway)
├── README.md
├── requirements.txt    # pip dependencies
└── model/
    ├── label.txt       # label kelas model deepfake
    └── model.keras     # model deepfake
```

## Prerequisites

- Python 3.10+
- pip (Python Package Manager)
- API Key dari [Groq](https://console.groq.com) untuk generative AI

## Setup Server Lokal

```bash
# 1. Setup virtual environment
python3 -m venv venv        # Windows: py -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# 2. Install dependencies
# Masuk atau arahkan ke dalam direktori proyek
pip install -r requirements.txt

# 3. Setup environment variabel
# Buat file .env secara manual
# Edit file .env dan isi:
# GROQ_API_KEY=<API_KEY_DARI_console.groq.com>

# 4. Run server
uvicorn main:app
# uvicorn main:app --reload (restart otomatis saat ada perubahan)
# uvicorn main:app --host 127.0.0.1 --port 5000 (menentukan host dan port custom)

# 5. Akses API Documentation (Swagger UI)
http://127.0.0.1:8000/docs # atau http://localhost:8000/docs
```

## Setup Server dengan Railway

1. Push direktori ke GitHub / Fork Repositories ini
2. Buat project baru di Railway lalu Deploy from GitHub
3. Buat variabel baru dengan nama `GROQ_API_KEY` dan isi dengan API Key dari [Groq](https://console.groq.com)
4. Railway otomatis membaca `Procfile` untuk menjalankan server

## Endpoint

| Method |    Path    | Deskripsi                             |
| ------ | :--------: | ------------------------------------- |
| POST   | `/predict` | Prediksi Gambar dengan Model & Gen AI |

## Response Example

```json
{
  "filename": "gambar2.jpg",
  "prediction": "Real",
  "confidence": 93.81,
  "probabilities": {
    "Fake": 6.19,
    "Real": 93.81
  },
  "raw_score": 0.9380651712417603,
  "explanation": "Model memprediksi gambar ini asli dengan tingkat kepercayaan 93.81%. Hasil ini menunjukkan kemungkinan besar foto tersebut tidak dimanipulasi. Namun, untuk memastikan, Anda dapat memeriksa sumber asli foto atau menggunakan alat forensik lain."
}
```

# 🎵 Music Genre Classifier

End-to-end music genre classification using **MFCC features**, a **CNN model**, and a **FastAPI inference API**.

---

## What It Does
- Extracts MFCC features from audio
- Predicts music genre using a trained CNN
- Exposes a REST API via FastAPI
- Returns top-K genre probabilities

---

## Structure
```text
app/ # FastAPI inference
models/ # Trained model + labels
training/ # Training scripts
samples/ # Sample audio
```

---

## ▶️ Run Locally
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Swagger UI: http://127.0.0.1:8000/docs

## API
POST /predict
Upload an audio file (.wav, .mp3, .flac, .ogg)

{
  "genre": "jazz",
  "confidence": 0.56,
  "top_k": [
    { "genre": "jazz", "prob": 0.56 },
    { "genre": "classical", "prob": 0.31 },
    { "genre": "rock", "prob": 0.06 }
  ]
}
## Tech
Python · TensorFlow/Keras · Librosa · FastAPI · Uvicorn

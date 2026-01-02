from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile
import os

from app.features import extract_mfcc
from app.inference import predict

app = FastAPI(title="Music Genre Classifier")

@app.get("/")
def root():
    return {"status": "Music Genre Classifier API is running"}

@app.post("/predict")
async def predict_genre(file: UploadFile = File(...)):
    # basic file type check (optional but good practice)
    if not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    # save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # extract features
        mfccs = extract_mfcc(tmp_path)

        # guard against short / invalid audio
        if mfccs.shape[0] == 0:
            raise HTTPException(
                status_code=400,
                detail="Audio too short or invalid for feature extraction"
            )

        # run inference
        result = predict(mfccs)
        return result

    finally:
        # always clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
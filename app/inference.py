import json
import numpy as np
from tensorflow import keras

MODEL_PATH = "models/cnn_genre.keras"
MAPPING_PATH = "models/mapping.json"

model = keras.models.load_model(MODEL_PATH)

with open(MAPPING_PATH, "r") as fp:
    mapping = json.load(fp)

def predict(mfccs, top_k=3):
    preds = model.predict(mfccs)
    mean_pred = np.mean(preds, axis=0)

    indices = np.argsort(mean_pred)[::-1][:top_k]

    return {
        "genre": mapping[indices[0]],
        "confidence": float(mean_pred[indices[0]]),
        "top_k": [
            {"genre": mapping[i], "prob": float(mean_pred[i])}
            for i in indices
        ]
    }
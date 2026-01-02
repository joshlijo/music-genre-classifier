from app.inference import predict
from app.features import extract_mfcc

mfccs = extract_mfcc("samples/wmggw.mp3")
print(predict(mfccs))
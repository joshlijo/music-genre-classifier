import librosa
import numpy as np
import math

SAMPLE_RATE = 22050
TRACK_DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def extract_mfcc(file_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors = math.ceil(samples_per_segment / hop_length)

    mfccs = []

    for d in range(num_segments):
        start = samples_per_segment * d
        finish = start + samples_per_segment

        mfcc = librosa.feature.mfcc(
            y=signal[start:finish],
            sr=sr,
            n_mfcc=num_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        mfcc = mfcc.T

        if len(mfcc) == num_mfcc_vectors:
            mfccs.append(mfcc)

    mfccs = np.array(mfccs)
    mfccs = mfccs[..., np.newaxis]

    return mfccs
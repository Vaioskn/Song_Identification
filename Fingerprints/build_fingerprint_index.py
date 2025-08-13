import os
import csv
import pickle
import hashlib

import numpy as np
import librosa
from scipy.ndimage import maximum_filter
from tqdm import tqdm

AUDIO_DIR = "audio_1000"
METADATA_CSV = os.path.join(AUDIO_DIR, "metadata.csv")
OUT_INDEX = "fingerprints.pkl"

PEAK_NEIGHBORHOOD_SIZE = 20
FAN_VALUE = 50
WINDOW_SIZE = 4096
OVERLAP_RATIO = 0.5
MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200

ENERGY_THRESHOLD_RATIO = 0.3

DEFAULT_FS = 22050

def stable_hash(f1, f2, dt):
    key = f"{f1}|{f2}|{dt}".encode("utf-8")
    digest = hashlib.sha1(key).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)

def get_anchors(S):
    footprint = np.ones((PEAK_NEIGHBORHOOD_SIZE, PEAK_NEIGHBORHOOD_SIZE))
    local_max = maximum_filter(S, footprint=footprint) == S
    background = (S == 0)
    eroded_bg = maximum_filter(background, footprint=footprint)
    peaks = local_max & ~eroded_bg
    return np.argwhere(peaks)

def fingerprint(y, sr=DEFAULT_FS):
    hop_length = int(WINDOW_SIZE * OVERLAP_RATIO)
    S = np.abs(librosa.stft(y, n_fft=WINDOW_SIZE, hop_length=hop_length))
    
    energy = np.sum(S**2, axis=0)
    thresh = np.median(energy) * ENERGY_THRESHOLD_RATIO

    anchors = get_anchors(S)
    anchors = [ (f,t) for f,t in anchors if energy[t] >= thresh ]

    hashes = []
    for i, (f1, t1) in enumerate(anchors):
        for j in range(1, FAN_VALUE):
            if i + j < len(anchors):
                f2, t2 = anchors[i + j]
                dt = t2 - t1
                if MIN_HASH_TIME_DELTA <= dt <= MAX_HASH_TIME_DELTA:
                    h = stable_hash(f1, f2, dt)
                    hashes.append((h, t1))
    return hashes

def fingerprint_file(path):
    y, sr = librosa.load(path, sr=DEFAULT_FS, mono=True)
    return fingerprint(y, sr)

def build_index():
    index = {}
    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Fingerprinting"):
            tid = int(row["track_id"])
            mp3_path = os.path.join(AUDIO_DIR, f"{tid}.mp3")
            if not os.path.isfile(mp3_path):
                tqdm.write(f"Missing file: {mp3_path}")
                continue
            try:
                for h, offs in fingerprint_file(mp3_path):
                    index.setdefault(h, []).append((tid, offs))
            except Exception as e:
                tqdm.write(f"Error on {tid}.mp3: {e}")
    with open(OUT_INDEX, "wb") as out:
        pickle.dump(index, out)
    print(f"Built index with {len(index)} unique hashes -> {OUT_INDEX}")

if __name__ == "__main__":
    build_index()
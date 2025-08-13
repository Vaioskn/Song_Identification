import os
import csv
import time
import pickle
import hashlib
from itertools import product

import numpy as np
import librosa
from scipy.ndimage import maximum_filter
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

def stable_hash(f1, f2, dt):
    key = f"{f1}|{f2}|{dt}".encode("utf-8")
    digest = hashlib.sha1(key).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def get_anchors(S, peak_size):
    footprint = np.ones((peak_size, peak_size))
    local_max = maximum_filter(S, footprint=footprint) == S
    background = (S == 0)
    eroded_bg = maximum_filter(background, footprint=footprint)
    return np.argwhere(local_max & ~eroded_bg)


def fingerprint(y, params):
    hop_length = int(params['WINDOW_SIZE'] * params['OVERLAP_RATIO'])
    S = np.abs(librosa.stft(y,
                             n_fft=params['WINDOW_SIZE'],
                             hop_length=hop_length))
    
    energy = np.sum(S**2, axis=0)
    threshold = np.median(energy) * params['ENERGY_THRESHOLD_RATIO']

    anchors = get_anchors(S, params['PEAK_NEIGHBORHOOD_SIZE'])
    anchors = [(f, t) for f, t in anchors if energy[t] >= threshold]

    hashes = []
    for i, (f1, t1) in enumerate(anchors):
        for j in range(1, params['FAN_VALUE']):
            if i + j < len(anchors):
                f2, t2 = anchors[i + j]
                dt = t2 - t1
                if params['MIN_HASH_TIME_DELTA'] <= dt <= params['MAX_HASH_TIME_DELTA']:
                    h = stable_hash(f1, f2, dt)
                    hashes.append((h, t1))
    return hashes


def build_index(audio_dir, metadata_csv, params):
    index = {}
    with open(metadata_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Indexing"):
            track_id = int(row['track_id'])
            path = os.path.join(audio_dir, f"{track_id}.mp3")
            if not os.path.isfile(path):
                continue
            y, sr = librosa.load(path, sr=params.get('DEFAULT_FS', 22050), mono=True)
            for h, offs in fingerprint(y, params):
                index.setdefault(h, []).append((track_id, offs))
    return index


def evaluate_index(audio_dir, metadata_csv, index, params, clip_duration=5.0, noise_level=0.0, top_n=1):
    with open(metadata_csv, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    hits = 0
    total = len(rows)
    t_start = time.time()

    for row in tqdm(rows, desc="Evaluating"):
        tid = int(row['track_id'])
        path = os.path.join(audio_dir, f"{tid}.mp3")
        y, sr = librosa.load(path, sr=params.get('DEFAULT_FS', 22050), mono=True)
        dur = len(y) / sr
        
        if dur > clip_duration:
            start = np.random.uniform(0, dur - clip_duration)
            clip = y[int(start * sr):int((start + clip_duration) * sr)]
        else:
            clip = y
            
        if noise_level > 0:
            noise = np.random.randn(len(clip))
            clip += noise_level * noise / np.std(noise)


        votes = {}
        for h, offs in fingerprint(clip, params):
            for track_id, db_offs in index.get(h, []):
                dt = db_offs - offs
                votes[(track_id, dt)] = votes.get((track_id, dt), 0) + 1
                
        track_scores = {}
        for (track_id, _), v in votes.items():
            track_scores[track_id] = max(track_scores.get(track_id, 0), v)
            
        top = sorted(track_scores.items(), key=lambda x: x[1], reverse=True)
        preds = [tid_score for tid_score, _ in top[:top_n]]

        if preds and tid in preds:
            hits += 1

    latency = (time.time() - t_start) / total
    accuracy = hits / total
    return accuracy, latency


if __name__ == '__main__':
    
    AUDIO_DIR = "audio_1000"
    METADATA_CSV = os.path.join(AUDIO_DIR, 'metadata.csv')
    
    base_params = {
        'WINDOW_SIZE': 4096,
        'OVERLAP_RATIO': 0.5,
        'MIN_HASH_TIME_DELTA': 0,
        'MAX_HASH_TIME_DELTA': 200,
        'DEFAULT_FS': 22050,
    }

    grid = {
        'PEAK_NEIGHBORHOOD_SIZE': [20, 30, 40],
        'FAN_VALUE': [15, 30, 50],
        'ENERGY_THRESHOLD_RATIO': [0.3, 0.5, 0.7],
    }
    top_n = 1

    results = []


    print("Starting grid search over {} configurations...".format(
        np.prod([len(v) for v in grid.values()])))

    for pns, fan, eth in product(
            grid['PEAK_NEIGHBORHOOD_SIZE'],
            grid['FAN_VALUE'],
            grid['ENERGY_THRESHOLD_RATIO']):
        params = dict(base_params)
        params.update({
            'PEAK_NEIGHBORHOOD_SIZE': pns,
            'FAN_VALUE': fan,
            'ENERGY_THRESHOLD_RATIO': eth,
        })
        label = f"PNS={pns}_FAN={fan}_ETH={eth}"
        print(f"\n--- Testing {label} ---")
        
        t0 = time.time()
        idx = build_index(AUDIO_DIR, METADATA_CSV, params)
        print(f"Index built ({len(idx)} hashes) in {time.time() - t0:.1f}s")
        
        acc, lat = evaluate_index(AUDIO_DIR, METADATA_CSV, idx, params, top_n=top_n)
        print(f"-> Top-1 Accuracy: {acc:.2%}, Latency: {lat:.3f}s/clip")
        results.append((label, pns, fan, eth, acc, lat))


    best = max(results, key=lambda x: x[4])
    print(f"\nBest config: {best[0]} with Top-1 Accuracy {best[4]:.2%}")


    out_csv = 'grid_search_results.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'PNS', 'FAN', 'ETH', 'accuracy', 'latency'])
        writer.writerows(results)
    print(f"Results saved to {out_csv}")

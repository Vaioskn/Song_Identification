import os
import csv
import pickle
import time
import numpy as np
import librosa
from scipy.ndimage import maximum_filter
from tqdm import tqdm
import matplotlib.pyplot as plt
import hashlib

AUDIO_DIR = "audio_1000"
METADATA_CSV = os.path.join(AUDIO_DIR, "metadata.csv")
INDEX_PATH = "fingerprints.pkl"

DURATIONS = [2.0, 5.0, 10.0]
SNR_DBS = [0, 10, 20]
TOP_NS = [1, 5]

CMC_MAX_RANK = 10

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
    fsz       = PEAK_NEIGHBORHOOD_SIZE
    footprint = np.ones((fsz, fsz))
    local_max = maximum_filter(S, footprint=footprint) == S
    background= (S == 0)
    eroded    = maximum_filter(background, footprint=footprint)
    return np.argwhere(local_max & ~eroded)

def fingerprint(y, sr=DEFAULT_FS):
    hop   = int(WINDOW_SIZE * OVERLAP_RATIO)
    S     = np.abs(librosa.stft(y, n_fft=WINDOW_SIZE, hop_length=hop))
    energy= np.sum(S**2, axis=0)
    thresh= np.median(energy) * ENERGY_THRESHOLD_RATIO

    anchors = [(f, t) for f, t in get_anchors(S) if energy[t] >= thresh]

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

def load_index(path=INDEX_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def recognize_scores(clip, index):
    votes = {}
    for h, offs in fingerprint(clip):
        for tid, db_offs in index.get(h, []):
            dt = db_offs - offs
            votes[(tid, dt)] = votes.get((tid, dt), 0) + 1
    track_scores = {}
    for (tid, _), v in votes.items():
        track_scores[tid] = max(track_scores.get(tid, 0), v)
    return track_scores


def evaluate_top_n(index, clip_dur, snr_db, top_n):
    rows = list(csv.DictReader(open(METADATA_CSV, newline="", encoding="utf-8")))
    hits = 0
    for r in tqdm(rows, desc=f"{clip_dur}s @ {snr_db}dB (Top-{top_n})"):
        tid  = int(r["track_id"])
        path = os.path.join(AUDIO_DIR, f"{tid}.mp3")
        y, sr = librosa.load(path, sr=DEFAULT_FS, mono=True)
        dur = len(y) / sr

        if dur > clip_dur:
            start = np.random.uniform(0, dur - clip_dur)
            i0, i1 = int(start * sr), int((start + clip_dur) * sr)
            clip = y[i0:i1]
        else:
            clip = y

        if snr_db >= 0:
            sig_pow   = np.mean(clip**2)
            noise_pow = sig_pow / (10**(snr_db / 10))
            noise = np.random.randn(len(clip))
            noise = noise * np.sqrt(noise_pow) / np.std(noise)
            clip  = clip + noise

        scores = recognize_scores(clip, index)
        ranked = [t for t, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
        if tid in ranked[:top_n]:
            hits += 1

    return hits / len(rows)

def compute_cmc(index, clip_dur, snr_db, max_rank=CMC_MAX_RANK):
    rows = list(csv.DictReader(open(METADATA_CSV, newline="", encoding="utf-8")))
    cmc_counts = np.zeros(max_rank, dtype=int)

    for r in tqdm(rows, desc=f"CMC {clip_dur}s @ {snr_db}dB"):
        tid  = int(r["track_id"])
        path = os.path.join(AUDIO_DIR, f"{tid}.mp3")
        y, sr = librosa.load(path, sr=DEFAULT_FS, mono=True)
        dur = len(y) / sr

        if dur > clip_dur:
            start = np.random.uniform(0, dur - clip_dur)
            i0, i1 = int(start * sr), int((start + clip_dur) * sr)
            clip = y[i0:i1]
        else:
            clip = y

        if snr_db >= 0:
            sig_pow   = np.mean(clip**2)
            noise_pow = sig_pow / (10**(snr_db / 10))
            noise = np.random.randn(len(clip))
            noise = noise * np.sqrt(noise_pow) / np.std(noise)
            clip  = clip + noise

        scores = recognize_scores(clip, index)
        ranked = [t for t, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
        for k in range(1, max_rank + 1):
            if tid in ranked[:k]:
                cmc_counts[k - 1] += 1

    return cmc_counts / len(rows)


if __name__ == "__main__":
    idx = load_index()

    results = {}
    for top_n in TOP_NS:
        for dur in DURATIONS:
            for snr in SNR_DBS:
                acc = evaluate_top_n(idx, dur, snr, top_n)
                results[(top_n, dur, snr)] = acc
                print(f"→ Top-{top_n}: {dur}s @ {snr}dB → {acc:.2%}")

    for top_n in TOP_NS:
        print(f"\n**Top-{top_n} Accuracy Table**")
        header = ["SNR \\ Dur"] + [f"{d}s" for d in DURATIONS]
        print("| " + " | ".join(header) + " |")
        print("|" + "------|" * len(header))
        for snr in SNR_DBS:
            row = [f"{snr}dB"] + [f"{results[(top_n, d, snr)]*100:5.2f}%" for d in DURATIONS]
            print("| " + " | ".join(row) + " |")

    for top_n in TOP_NS:
        plt.figure()
        for snr in SNR_DBS:
            ys = [results[(top_n, d, snr)] for d in DURATIONS]
            plt.plot(DURATIONS, ys, marker='o', label=f"{snr} dB")
        plt.xlabel("Clip duration (s)")
        plt.ylabel(f"Top-{top_n} accuracy")
        plt.title(f"Accuracy vs. Clip Length (Top-{top_n})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"accuracy_vs_clip_top{top_n}.png", dpi=200)

        plt.figure()
        for dur in DURATIONS:
            ys = [results[(top_n, dur, s)] for s in SNR_DBS]
            plt.plot(SNR_DBS, ys, marker='o', label=f"{dur}s")
        plt.xlabel("Noise level (SNR in dB)")
        plt.ylabel(f"Top-{top_n} accuracy")
        plt.title(f"Accuracy vs. Noise Level (Top-{top_n})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"accuracy_vs_noise_top{top_n}.png", dpi=200)

        data = np.array([[results[(top_n, d, s)] for s in SNR_DBS] for d in DURATIONS])
        fig, ax = plt.subplots()
        im = ax.imshow(data, aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(len(SNR_DBS)))
        ax.set_yticks(np.arange(len(DURATIONS)))
        ax.set_xticklabels([f"{s}dB" for s in SNR_DBS])
        ax.set_yticklabels([f"{d}s" for d in DURATIONS])
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i,j]*100:4.1f}%", ha="center", va="center", color="w")
        ax.set_xlabel("Noise (dB SNR)")
        ax.set_ylabel("Clip duration (s)")
        ax.set_title(f"Top-{top_n} Accuracy Heatmap")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(f"accuracy_heatmap_top{top_n}.png", dpi=200)

    cmc = compute_cmc(idx, clip_dur=5.0, snr_db=0, max_rank=CMC_MAX_RANK)
    plt.figure()
    ranks = np.arange(1, CMC_MAX_RANK + 1)
    plt.plot(ranks, cmc, marker='o')
    plt.xticks(ranks)
    plt.xlabel("Rank n")
    plt.ylabel("Recognition rate")
    plt.title("CMC Curve (5 s, 0 dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cmc_curve.png", dpi=200)

    print("\nplots saved")


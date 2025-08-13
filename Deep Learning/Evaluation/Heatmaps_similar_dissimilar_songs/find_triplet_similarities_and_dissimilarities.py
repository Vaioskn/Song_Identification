import os
import csv
import random
import warnings
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from transformers import AutoModel, AutoFeatureExtractor
import itertools

warnings.filterwarnings("ignore")

TEST_DATA_DIR = 'TEST_DATASET'

TEST_METADATA_CSV = 'TEST_DATASET/metadata.csv'

OUTPUT_DIR = 'TRIPLET_ANALYSIS_RESULTS'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_HF_PATH = "m-a-p/MERT-v1-95M"
SR = 24000

CLIP_DURATION = 5.0 
CLIPS_PER_SONG_FOR_SIGNATURE = 10
NUM_RESULTS_TO_SHOW = 20

class EmbeddingModel(nn.Module):
    def __init__(self, model_hf_path):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_hf_path, trust_remote_code=True)
        
    def forward(self, x):
        return torch.mean(self.base_model(x).last_hidden_state, dim=1)

def _load_random_clip(path, duration, sr):
    target_len = int(duration * sr)
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
    except Exception as e:
        print(f"Warning: Could not load {path}, skipping. Error: {e}")
        return None
        
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), 'constant')
    
    if len(y) > target_len:
        start = random.randint(0, len(y) - target_len)
        y = y[start:start + target_len]
        
    return y

@torch.no_grad()
def get_song_signature(track_id, model, feature_extractor, device):
    path = os.path.join(TEST_DATA_DIR, f"{str(track_id)}.mp3")
    clip_embeddings = []
    
    for _ in range(CLIPS_PER_SONG_FOR_SIGNATURE):
        audio_clip = _load_random_clip(path, CLIP_DURATION, SR)
        if audio_clip is None: continue
            
        inputs = feature_extractor(audio_clip, sampling_rate=SR, return_tensors="pt", padding=True).to(device)
        embedding = model(inputs['input_values'])
        clip_embeddings.append(embedding)
    
    if not clip_embeddings: return None
        
    return torch.mean(torch.cat(clip_embeddings, dim=0), dim=0)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        print(f"Loading feature extractor and model: {MODEL_HF_PATH}...")
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_HF_PATH, trust_remote_code=True)
        model = EmbeddingModel(model_hf_path=MODEL_HF_PATH).to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model")
        print(f"Details: {e}")
        return

    try:
        test_metadata = pd.read_csv(TEST_METADATA_CSV)
        test_track_ids = sorted(test_metadata['track_id'].astype(str).unique().tolist())
        print(f"Found {len(test_track_ids)} unique songs to analyze.")
    except Exception as e:
        print(f"ERROR: Could not read test metadata at {TEST_METADATA_CSV}. {e}")
        return
        
    print("\n" + "="*80)
    print("Step 1: unique 'sound signature for every song...")
    print("="*80)
    
    song_signatures = {}
    for tid in tqdm(test_track_ids, desc="Generating Signatures"):
        signature = get_song_signature(tid, model, feature_extractor, device)
        if signature is not None:
            song_signatures[tid] = signature.cpu()

    if len(song_signatures) < 3:
        print("Not enough song signatures were created to perform triplet comparison. Exiting.")
        return

    print("\n" + "="*80)
    print("Step 2: Calculating similarity between all song triplets...")
    print("="*80)

    track_ids = list(song_signatures.keys())
    all_triplets = []
    
    for tid1, tid2, tid3 in tqdm(itertools.combinations(track_ids, 3), desc="Comparing Triplets"):
        z1 = song_signatures[tid1].unsqueeze(0)
        z2 = song_signatures[tid2].unsqueeze(0)
        z3 = song_signatures[tid3].unsqueeze(0)
        
        sim12 = F.cosine_similarity(z1, z2).item()
        sim13 = F.cosine_similarity(z1, z3).item()
        sim23 = F.cosine_similarity(z2, z3).item()
        
        avg_similarity = (sim12 + sim13 + sim23) / 3.0
        
        all_triplets.append((tid1, tid2, tid3, avg_similarity))

    print("\n" + "="*80)
    print("Step 3: Ranking triplets and presenting results...")
    print("="*80)

    if not all_triplets:
        print("No triplets could be formed. Exiting.")
        return

    all_triplets_df = pd.DataFrame(all_triplets, columns=['Song A ID', 'Song B ID', 'Song C ID', 'Average Similarity'])
    
    most_similar_df = all_triplets_df.sort_values(by='Average Similarity', ascending=False).head(NUM_RESULTS_TO_SHOW)
    most_dissimilar_df = all_triplets_df.sort_values(by='Average Similarity', ascending=True).head(NUM_RESULTS_TO_SHOW)

    print("\nTop 20 Most Similar Sounding Song Triplets (Sonic Clusters)")
    print(most_similar_df.to_string(index=False))
    most_similar_df.to_csv(os.path.join(OUTPUT_DIR, 'most_similar_triplets.csv'), index=False)
    print(f"\nSaved to {os.path.join(OUTPUT_DIR, 'most_similar_triplets.csv')}")

    print("\nTop 20 Most Dissimilar Sounding Song Triplets (Diverse Sets)")
    print(most_dissimilar_df.to_string(index=False))
    most_dissimilar_df.to_csv(os.path.join(OUTPUT_DIR, 'most_dissimilar_triplets.csv'), index=False)
    print(f"\nSaved to {os.path.join(OUTPUT_DIR, 'most_dissimilar_triplets.csv')}")
    
    print("\n\nfinished")


if __name__ == "__main__":
    main()

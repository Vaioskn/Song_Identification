import os
import csv
import pickle
import random
import time
import itertools
import warnings
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoFeatureExtractor
from audiomentations import Compose, AddGaussianNoise
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

MODEL_TYPE = "MERT"

MODEL_PATH = r'AST/finetuned_mert_model.pt'

USE_ATTENTION_POOLING = False

LABELMAP_PATH = r'AST/finetuned_mert_label_map.pkl'

TEST_DATA_DIR = r'TEST_DATASET'

TEST_METADATA_CSV = r'TEST_DATASET/metadata.csv'

OUTPUT_DIR = f'FINAL_ANALYSIS_RESULTS_{MODEL_TYPE}'
if USE_ATTENTION_POOLING:
    OUTPUT_DIR += "_WithAttention"
else:
    OUTPUT_DIR += "_NoAttention"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SNR_LEVELS = [None, 20, 10, 0]
DURATION_LEVELS = [2, 5, 10]
DB_CLIP_DURATION = 5.0
T_SNE_SONGS_TO_PLOT = 30
T_SNE_CLIPS_PER_SONG = 5
HEATMAP_SONGS_TO_PLOT = 4
HEATMAP_CLIPS_PER_SONG = 3
NUM_HARD_CASES_TO_SHOW = 10

class AttentionPooling(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.GELU(),
            nn.Linear(in_features, 1)
        )
    def forward(self, x):
        attention_scores = self.attention(x)
        attention_weights = torch.softmax(attention_scores, dim=1)
        return torch.sum(x * attention_weights, dim=1)

class MERTForSongID(nn.Module):
    def __init__(self, model_name="m-a-p/MERT-v1-95M", emb_dim=256, use_attention_pooling=True, **kwargs):
        super().__init__()
        self.use_attention_pooling = use_attention_pooling
        self.base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        mert_embedding_size = self.base_model.config.hidden_size

        if self.use_attention_pooling:
            self.pooling = AttentionPooling(in_features=mert_embedding_size)

        self.proj_head = nn.Sequential(
            nn.Linear(mert_embedding_size, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, emb_dim)
        )
    def forward(self, x):
        hidden_states = self.base_model(x).last_hidden_state
        if self.use_attention_pooling:
            mert_embedding = self.pooling(hidden_states)
        else:
            mert_embedding = torch.mean(hidden_states, dim=1)

        z_unnorm = self.proj_head(mert_embedding)
        return F.normalize(z_unnorm, p=2, dim=1)

class ASTForSongID(nn.Module):
    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593", emb_dim=256, use_attention_pooling=True, **kwargs):
        super().__init__()
        self.use_attention_pooling = use_attention_pooling
        self.base_model = AutoModel.from_pretrained(model_name)
        ast_embedding_size = self.base_model.config.hidden_size

        if self.use_attention_pooling:
            self.pooling = AttentionPooling(in_features=ast_embedding_size)

        self.proj_head = nn.Sequential(
            nn.Linear(ast_embedding_size, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, emb_dim)
        )
    def forward(self, x):
        hidden_states = self.base_model(x).last_hidden_state
        if self.use_attention_pooling:
            ast_embedding = self.pooling(hidden_states)
        else:
            ast_embedding = torch.mean(hidden_states, dim=1)

        z_unnorm = self.proj_head(ast_embedding)
        return F.normalize(z_unnorm, p=2, dim=1)


def _load_random_clip(path, duration, sr):
    target_len = int(duration * sr)
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return np.zeros(target_len, dtype=np.float32)
    if len(y) > target_len:
        start = random.randint(0, len(y) - target_len)
        y = y[start:start + target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)), 'constant')
    return y

@torch.no_grad()
def get_embedding(audio_clip, model, feature_extractor, sr, device):
    inputs = feature_extractor(audio_clip, sampling_rate=sr, return_tensors="pt", padding=True).to(device)
    embedding = model(inputs['input_values'])
    return embedding.cpu()

def run_robustness_analysis(model, feature_extractor, test_track_ids, sr):
    print("\n" + "="*80)
    print("ANALYSIS 1: ROBUSTNESS TO DURATION AND NOISE")
    print("="*80)

    device = next(model.parameters()).device
    model.eval()

    results = pd.DataFrame(index=[f"{snr if snr is not None else 'Clean'} dB" for snr in SNR_LEVELS], columns=[f"{dur}s" for dur in DURATION_LEVELS])

    print("Building clean reference database...")
    embeddings_db = {}
    for tid in tqdm(test_track_ids, desc="Building DB"):
        path = os.path.join(TEST_DATA_DIR, f"{tid}.mp3")
        clip_embs = [get_embedding(_load_random_clip(path, DB_CLIP_DURATION, sr), model, feature_extractor, sr, device) for _ in range(5)]
        embeddings_db[tid] = torch.mean(torch.cat(clip_embs, dim=0), dim=0)

    snr_to_amplitude_map = {
        20: 0.005,
        10: 0.015,
        0:  0.05,
    }

    for snr in SNR_LEVELS:
        for duration in DURATION_LEVELS:
            print(f"\n-- Evaluating: Duration={duration}s, SNR={snr if snr is not None else 'Clean'} dB --")
            correct_predictions = 0
            for query_tid in tqdm(test_track_ids, desc=f"Querying ({duration}s, {snr}dB)"):
                path = os.path.join(TEST_DATA_DIR, f"{query_tid}.mp3")
                y_query_clean = _load_random_clip(path, duration, sr)

                if snr is not None:
                    amplitude = snr_to_amplitude_map.get(snr, 0)
                    noise_adder = Compose([AddGaussianNoise(min_amplitude=amplitude, max_amplitude=amplitude, p=1.0)])
                    y_query_noisy = noise_adder(samples=y_query_clean, sample_rate=sr)
                else:
                    y_query_noisy = y_query_clean

                zq = get_embedding(y_query_noisy, model, feature_extractor, sr, device)
                sims = {tid: F.cosine_similarity(zq, emb.unsqueeze(0)).item() for tid, emb in embeddings_db.items()}
                predicted_tid = max(sims, key=sims.get)

                if predicted_tid == query_tid:
                    correct_predictions += 1

            accuracy = (correct_predictions / len(test_track_ids)) * 100 if test_track_ids else 0.0
            results.loc[f"{snr if snr is not None else 'Clean'} dB", f"{duration}s"] = f"{accuracy:.2f}%"

    print("\n--- Robustness Analysis Results ---")
    print(results)
    results.to_csv(os.path.join(OUTPUT_DIR, 'robustness_analysis.csv'))
    print(f"Results table saved to {os.path.join(OUTPUT_DIR, 'robustness_analysis.csv')}")

def run_tsne_analysis(model, feature_extractor, test_metadata, sr):
    print("\n" + "="*80)
    print("ANALYSIS 2: t-SNE VISUALIZATION OF EMBEDDING SPACE")
    print("="*80)
    if 'genre' not in test_metadata.columns:
        print("WARNING: 'genre' column not found in metadata. Skipping t-SNE plot.")
        return

    device = next(model.parameters()).device
    model.eval()

    print(f"Selecting {T_SNE_SONGS_TO_PLOT} random songs for visualization...")
    subset_df = test_metadata.sample(n=min(T_SNE_SONGS_TO_PLOT, len(test_metadata)), random_state=42)

    embeddings = []
    labels = []
    for _, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc="Generating t-SNE embeddings"):
        tid = row['track_id']
        genre = row['genre']
        path = os.path.join(TEST_DATA_DIR, f"{tid}.mp3")
        for _ in range(T_SNE_CLIPS_PER_SONG):
            y_clip = _load_random_clip(path, DB_CLIP_DURATION, sr)
            embeddings.append(get_embedding(y_clip, model, feature_extractor, sr, device))
            labels.append(genre)

    embeddings_cat = torch.cat(embeddings, dim=0).numpy()

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, verbose=1, perplexity=min(30, len(embeddings_cat)-1), n_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(embeddings_cat)

    plot_df = pd.DataFrame({'x': tsne_results[:,0], 'y': tsne_results[:,1], 'genre': labels})

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="x", y="y",
        hue="genre",
        palette=sns.color_palette("hsv", len(plot_df['genre'].unique())),
        data=plot_df,
        legend="full",
        alpha=0.8
    )
    plt.title('t-SNE Projection of Song Embeddings, Colored by Genre')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'tsne_genre_visualization.png')
    plt.savefig(save_path)
    print(f"t-SNE plot saved to {save_path}")
    plt.close()

def run_distance_distribution_analysis(model, feature_extractor, test_track_ids, sr):
    print("\n" + "="*80)
    print("ANALYSIS 3: INTRA-CLASS VS. INTER-CLASS SIMILARITY")
    print("="*80)

    device = next(model.parameters()).device
    model.eval()

    intra_class_sims = []
    inter_class_sims = []

    print("Generating embeddings for distance analysis...")
    song_embeddings = {}
    for tid in tqdm(test_track_ids, desc="Generating embeddings"):
        path = os.path.join(TEST_DATA_DIR, f"{tid}.mp3")
        clip1 = get_embedding(_load_random_clip(path, DB_CLIP_DURATION, sr), model, feature_extractor, sr, device)
        clip2 = get_embedding(_load_random_clip(path, DB_CLIP_DURATION, sr), model, feature_extractor, sr, device)
        song_embeddings[tid] = (clip1, clip2)

    print("Calculating similarities...")
    track_list = list(song_embeddings.keys())
    for i in range(len(track_list)):
        tid1 = track_list[i]
        z1_1, z1_2 = song_embeddings[tid1]

        intra_class_sims.append(F.cosine_similarity(z1_1, z1_2).item())

        j = i
        while j == i:
            j = random.randint(0, len(track_list) - 1)
        tid2 = track_list[j]
        z2_1, _ = song_embeddings[tid2]
        inter_class_sims.append(F.cosine_similarity(z1_1, z2_1).item())

    plt.figure(figsize=(10, 6))
    sns.histplot(intra_class_sims, color="blue", label='Intra-Class (Same Song)', kde=True, stat="density", element="step")
    sns.histplot(inter_class_sims, color="red", label='Inter-Class (Different Songs)', kde=True, stat="density", element="step")
    plt.title('Distribution of Embedding Similarities')
    plt.xlabel('Cosine Similarity')
    plt.legend()

    save_path = os.path.join(OUTPUT_DIR, 'similarity_distributions.png')
    plt.savefig(save_path)
    print(f"Distance distribution plot saved to {save_path}")
    plt.close()

def run_similarity_matrix_analysis(model, feature_extractor, test_track_ids, sr):
    print("\n" + "="*80)
    print("ANALYSIS 4: SIMILARITY MATRIX HEATMAP")
    print("="*80)

    device = next(model.parameters()).device
    model.eval()

    if len(test_track_ids) < HEATMAP_SONGS_TO_PLOT:
        print("Not enough unique songs in test set to generate heatmap. Skipping.")
        return

    selected_tids = random.sample(test_track_ids, HEATMAP_SONGS_TO_PLOT)

    embeddings = []
    labels = []
    for tid in selected_tids:
        path = os.path.join(TEST_DATA_DIR, f"{tid}.mp3")
        for i in range(HEATMAP_CLIPS_PER_SONG):
            y_clip = _load_random_clip(path, DB_CLIP_DURATION, sr)
            embeddings.append(get_embedding(y_clip, model, feature_extractor, sr, device))
            labels.append(f"{str(tid)[:5]}-{i+1}")

    embeddings_cat = torch.cat(embeddings, dim=0)
    sim_matrix = F.cosine_similarity(embeddings_cat.unsqueeze(1), embeddings_cat.unsqueeze(0), dim=-1)

    plt.figure(figsize=(12, 10))
    sns.heatmap(sim_matrix.numpy(), xticklabels=labels, yticklabels=labels, cmap='viridis', annot=False)
    plt.title('Cosine Similarity Matrix of Sample Clips')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'similarity_heatmap.png')
    plt.savefig(save_path)
    print(f"Similarity heatmap saved to {save_path}")
    plt.close()

def run_hard_mining_analysis(model, feature_extractor, test_track_ids, sr):
    print("\n" + "="*80)
    print("ANALYSIS 5: HARD-CASE MINING")
    print("="*80)

    device = next(model.parameters()).device
    model.eval()

    print("Building reference database for hard-case mining...")
    embeddings_db = {}
    for tid in tqdm(test_track_ids, desc="Building DB"):
        path = os.path.join(TEST_DATA_DIR, f"{tid}.mp3")
        clip_embs = [get_embedding(_load_random_clip(path, DB_CLIP_DURATION, sr), model, feature_extractor, sr, device) for _ in range(5)]
        embeddings_db[tid] = torch.mean(torch.cat(clip_embs, dim=0), dim=0)

    print("Finding hardest negative examples...")
    hard_negatives = []
    for query_tid in tqdm(test_track_ids, desc="Mining Hard Negatives"):
        y_query = _load_random_clip(os.path.join(TEST_DATA_DIR, f"{query_tid}.mp3"), DB_CLIP_DURATION, sr)
        zq = get_embedding(y_query, model, feature_extractor, sr, device)

        sims = {tid: F.cosine_similarity(zq, emb.unsqueeze(0)).item() for tid, emb in embeddings_db.items()}
        sims.pop(query_tid, None)

        if sims:
            hardest_negative_tid = max(sims, key=sims.get)
            score = sims[hardest_negative_tid]
            hard_negatives.append((query_tid, hardest_negative_tid, score))

    hard_negatives_df = pd.DataFrame(hard_negatives, columns=['Query Song ID', 'Most Confused With (Impostor)', 'Similarity Score'])
    hard_negatives_df = hard_negatives_df.sort_values(by='Similarity Score', ascending=False).head(NUM_HARD_CASES_TO_SHOW)

    print("Finding hardest positive examples...")
    hard_positives = []
    for tid in tqdm(test_track_ids, desc="Mining Hard Positives"):
        path = os.path.join(TEST_DATA_DIR, f"{tid}.mp3")
        clips = [_load_random_clip(path, DB_CLIP_DURATION, sr) for _ in range(10)]
        embeddings = [get_embedding(c, model, feature_extractor, sr, device) for c in clips]
        embeddings_cat = torch.cat(embeddings, dim=0)

        sim_matrix = F.cosine_similarity(embeddings_cat.unsqueeze(1), embeddings_cat.unsqueeze(0), dim=-1)
        sim_matrix.fill_diagonal_(1.0)

        min_sim_val, _ = torch.min(sim_matrix.view(-1), 0)
        hard_positives.append((tid, min_sim_val.item()))

    hard_positives_df = pd.DataFrame(hard_positives, columns=['Song ID', 'Lowest Intra-Song Similarity'])
    hard_positives_df = hard_positives_df.sort_values(by='Lowest Intra-Song Similarity', ascending=True).head(NUM_HARD_CASES_TO_SHOW)

    print("\nHard Negative Analysis Results (Top Confusions)")
    print(hard_negatives_df)
    hard_negatives_df.to_csv(os.path.join(OUTPUT_DIR, 'hard_negatives_analysis.csv'), index=False)

    print("\nHard Positive Analysis Results (Most Dissimilar Clips from Same Song)")
    print(hard_positives_df)
    hard_positives_df.to_csv(os.path.join(OUTPUT_DIR, 'hard_positives_analysis.csv'), index=False)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        with open(LABELMAP_PATH, 'rb') as f:
            id2idx = pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: Label map file not found at {LABELMAP_PATH}")
        return

    print(f"Loading configuration for model type: {MODEL_TYPE}")
    if MODEL_TYPE == "MERT":
        model_class = MERTForSongID
        sr = 24000
        model_hf_path = "m-a-p/MERT-v1-95M"
        trust_remote = True
    elif MODEL_TYPE == "AST":
        model_class = ASTForSongID
        sr = 16000
        model_hf_path = "MIT/ast-finetuned-audioset-10-10-0.4593"
        trust_remote = False
    else:
        raise ValueError(f"Invalid MODEL_TYPE in config: {MODEL_TYPE}. must be 'MERT' or 'AST'.")

    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_hf_path, trust_remote_code=trust_remote)

        model = model_class(
            model_name=model_hf_path,
            emb_dim=256,
            use_attention_pooling=USE_ATTENTION_POOLING
        )

        full_state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(full_state_dict, strict=False)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"ERROR: Failed to load model from {MODEL_PATH}. {e}")
        return

    try:
        test_metadata = pd.read_csv(TEST_METADATA_CSV)
        test_track_ids = sorted(test_metadata['track_id'].astype(str).unique().tolist())
    except FileNotFoundError:
        print(f"ERROR: Test metadata file not found at {TEST_METADATA_CSV}")
        return

    run_robustness_analysis(model, feature_extractor, test_track_ids, sr)
    run_tsne_analysis(model, feature_extractor, test_metadata, sr)
    run_distance_distribution_analysis(model, feature_extractor, test_track_ids, sr)
    run_similarity_matrix_analysis(model, feature_extractor, test_track_ids, sr)
    run_hard_mining_analysis(model, feature_extractor, test_track_ids, sr)

    print("\n\nComprehensive analysis finished!")

if __name__ == "__main__":
    main()
import os
import csv
import pickle
import random
import time
import warnings
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoFeatureExtractor
from audiomentations import Compose, AddGaussianNoise

warnings.filterwarnings("ignore")

MASTER_OUTPUT_DIR = 'MASTER_ANALYSIS_RESULTS'

TEST_DATA_DIR = 'TEST_DATASET'

TEST_METADATA_CSV = 'TEST_DATASET/metadata.csv'

MODELS_TO_EVALUATE = [
    {
        "display_name": "Simple_CNN_3Layer",
        "model_type": "CNN",
        "model_path": "basic_cnn_model.pt",
        "cnn_arch": "Encoder3Layer",
        "sr": 22050,
        "emb_dim": 128
    },
    {
        "display_name": "Enhanced_CNN_4Layer",
        "model_type": "CNN",
        "model_path": "enhanced_cnn_model.pt",
        "cnn_arch": "Encoder4Layer",
        "sr": 22050,
        "emb_dim": 256
    },
    {
        "display_name": "MERT_Simple_Pooling",
        "model_type": "MERT",
        "model_path": "MERT_model.pt",
        "use_attention_pooling": False,
        "sr": 24000,
        "model_hf_path": "m-a-p/MERT-v1-95M"
    },
    {
        "display_name": "MERT_with_Attention",
        "model_type": "MERT",
        "model_path": "MERT_better_model.pt",
        "use_attention_pooling": True,
        "sr": 24000,
        "model_hf_path": "m-a-p/MERT-v1-95M"
    },
    {
        "display_name": "AST_Simple_Pooling",
        "model_type": "AST",
        "model_path": "AST_basic_model.pt",
        "use_attention_pooling": False,
        "sr": 16000,
        "model_hf_path": "MIT/ast-finetuned-audioset-10-10-0.4593"
    },
    {
        "display_name": "AST_with_Attention",
        "model_type": "AST",
        "model_path": "AST_better_model.pt",
        "use_attention_pooling": True,
        "sr": 16000,
        "model_hf_path": "MIT/ast-finetuned-audioset-10-10-0.4593"
    },
]

SONGS_FOR_VISUALIZATION = []


SNR_LEVELS = [None, 20, 10, 0]
DURATION_LEVELS = [2, 5, 10]
DB_CLIP_DURATION = 5.0
T_SNE_SONGS_TO_PLOT = 30
T_SNE_CLIPS_PER_SONG = 5
HEATMAP_SONGS_TO_PLOT = 4
HEATMAP_CLIPS_PER_SONG = 3
NUM_HARD_CASES_TO_SHOW = 10
N_MELS = 64

class AttentionPooling(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features), nn.LayerNorm(in_features),
            nn.GELU(), nn.Linear(in_features, 1)
        )
    def forward(self, x):
        attn_scores = self.attention(x)
        attn_weights = torch.softmax(attn_scores, dim=1)
        return torch.sum(x * attn_weights, dim=1)

class MERTForAnalysis(nn.Module):
    def __init__(self, model_hf_path, use_attention_pooling=True, emb_dim=256, **kwargs):
        super().__init__()
        self.use_attention_pooling = use_attention_pooling
        self.base_model = AutoModel.from_pretrained(model_hf_path, trust_remote_code=True)
        embedding_size = self.base_model.config.hidden_size
        if self.use_attention_pooling: self.pooling = AttentionPooling(in_features=embedding_size)
        self.proj_head = nn.Sequential(nn.Linear(embedding_size, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, emb_dim))
    def forward(self, x):
        hidden_states = self.base_model(x).last_hidden_state
        embedding = self.pooling(hidden_states) if self.use_attention_pooling else torch.mean(hidden_states, dim=1)
        return F.normalize(self.proj_head(embedding), p=2, dim=1)

class ASTForAnalysis(nn.Module):
    def __init__(self, model_hf_path, use_attention_pooling=True, emb_dim=256, **kwargs):
        super().__init__()
        self.use_attention_pooling = use_attention_pooling
        self.base_model = AutoModel.from_pretrained(model_hf_path)
        embedding_size = self.base_model.config.hidden_size
        if self.use_attention_pooling: self.pooling = AttentionPooling(in_features=embedding_size)
        self.proj_head = nn.Sequential(nn.Linear(embedding_size, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, emb_dim))
    def forward(self, x):
        hidden_states = self.base_model(x).last_hidden_state
        embedding = self.pooling(hidden_states) if self.use_attention_pooling else torch.mean(hidden_states, dim=1)
        return F.normalize(self.proj_head(embedding), p=2, dim=1)

class Encoder3Layer(nn.Module):
    def __init__(self, emb_dim=128, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.proj = nn.Sequential(nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, emb_dim))
    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return F.normalize(self.proj(h), dim=1)

class Encoder4Layer(nn.Module):
    def __init__(self, emb_dim=256, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.proj = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, emb_dim))
    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return F.normalize(self.proj(h), dim=1)

def _load_random_clip(path, duration, sr):
    target_len = int(duration * sr)
    try: y, _ = librosa.load(path, sr=sr, mono=True)
    except Exception: return np.zeros(target_len, dtype=np.float32)
    if len(y) > target_len:
        start = random.randint(0, len(y) - target_len)
        y = y[start:start + target_len]
    else: y = np.pad(y, (0, target_len - len(y)), 'constant')
    return y

def compute_log_mel(y, sr, n_mels):
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=n_mels)
    logm = librosa.power_to_db(melspec)
    logm = (logm - logm.mean()) / (logm.std() + 1e-6)
    return torch.from_numpy(logm).unsqueeze(0)

@torch.no_grad()
def get_embedding(audio_clip, model, feature_extractor, sr, n_mels, device):
    if feature_extractor:
        inputs = feature_extractor(audio_clip, sampling_rate=sr, return_tensors="pt", padding=True).to(device)
        embedding = model(inputs['input_values'])
    else:
        mel_tensor = compute_log_mel(audio_clip, sr=sr, n_mels=n_mels).unsqueeze(0).to(device)
        embedding = model(mel_tensor)
    return embedding.cpu()

def run_robustness_analysis(model, feature_extractor, test_track_ids, sr, n_mels, output_dir):
    print("\n" + "="*80); print("ANALYSIS 1: ROBUSTNESS (Top-5 Accuracy)"); print("="*80)
    device = next(model.parameters()).device; model.eval()
    results = pd.DataFrame(index=[f"{snr if snr is not None else 'Clean'} dB" for snr in SNR_LEVELS], columns=[f"{dur}s" for dur in DURATION_LEVELS])

    print("Building clean reference database...")
    embeddings_db = {}
    for tid in tqdm(test_track_ids, desc="Building DB"):
        path = os.path.join(TEST_DATA_DIR, f"{tid}.mp3")
        clip_embs = [get_embedding(_load_random_clip(path, DB_CLIP_DURATION, sr), model, feature_extractor, sr, n_mels, device) for _ in range(5)]
        embeddings_db[tid] = torch.mean(torch.cat(clip_embs, dim=0), dim=0)

    snr_to_amplitude_map = {20: 0.005, 10: 0.015, 0: 0.05}
    for snr in SNR_LEVELS:
        for duration in DURATION_LEVELS:
            correct_top5 = 0
            for query_tid in tqdm(test_track_ids, desc=f"Querying ({duration}s, {snr}dB)"):
                y_query_clean = _load_random_clip(os.path.join(TEST_DATA_DIR, f"{query_tid}.mp3"), duration, sr)
                y_query_noisy = y_query_clean
                if snr is not None:
                    noise_adder = Compose([AddGaussianNoise(min_amplitude=snr_to_amplitude_map.get(snr, 0), max_amplitude=snr_to_amplitude_map.get(snr, 0), p=1.0)])
                    y_query_noisy = noise_adder(samples=y_query_clean, sample_rate=sr)

                zq = get_embedding(y_query_noisy, model, feature_extractor, sr, n_mels, device)
                sims = {tid: F.cosine_similarity(zq, emb.unsqueeze(0)).item() for tid, emb in embeddings_db.items()}
                top_5_preds = sorted(sims, key=sims.get, reverse=True)[:5]
                if query_tid in top_5_preds: correct_top5 += 1
            accuracy = (correct_top5 / len(test_track_ids)) * 100 if test_track_ids else 0.0
            results.loc[f"{snr if snr is not None else 'Clean'} dB", f"{duration}s"] = f"{accuracy:.2f}%"

    print("\nRobustness Analysis Results (Top-5 Accuracy)"); print(results)
    results.to_csv(os.path.join(output_dir, 'robustness_analysis_top5.csv'))
    return results.loc['Clean dB', '5s']

def run_tsne_analysis(model, feature_extractor, test_metadata, sr, n_mels, output_dir, model_config, songs_to_plot):
    print("\n" + "="*80); print("ANALYSIS 2: t-SNE VISUALIZATION"); print("="*80)
    if 'genre' not in test_metadata.columns: print("WARNING: 'genre' column not found in metadata. Skipping t-SNE plot."); return
    device = next(model.parameters()).device; model.eval()

    subset_df = test_metadata[test_metadata['track_id'].astype(str).isin(songs_to_plot)]

    embeddings, labels = [], []
    for _, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc="Generating t-SNE embeddings"):
        tid, genre = row['track_id'], row['genre']
        path = os.path.join(TEST_DATA_DIR, f"{str(tid)}.mp3")
        for _ in range(T_SNE_CLIPS_PER_SONG):
            y_clip = _load_random_clip(path, DB_CLIP_DURATION, sr)
            embeddings.append(get_embedding(y_clip, model, feature_extractor, sr, n_mels, device))
            labels.append(genre)

    if not embeddings: print("Could not generate any embeddings for t-SNE plot."); return

    embeddings_cat = torch.cat(embeddings, dim=0).numpy()
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, verbose=1, perplexity=min(30, len(embeddings_cat)-1), n_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(embeddings_cat)
    plot_df = pd.DataFrame({'x': tsne_results[:,0], 'y': tsne_results[:,1], 'genre': labels})

    plt.figure(figsize=(16, 10))
    sns.scatterplot(x="x", y="y", hue="genre", palette=sns.color_palette("hsv", len(plot_df['genre'].unique())), data=plot_df, legend="full", alpha=0.8)
    plt.title(f't-SNE Projection of Song Embeddings\n({model_config["display_name"]})')
    plt.xlabel('t-SNE Dimension 1'); plt.ylabel('t-SNE Dimension 2'); plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout(); save_path = os.path.join(output_dir, 'tsne_genre_visualization.png')
    plt.savefig(save_path); print(f"t-SNE plot saved to {save_path}"); plt.close()

def run_distance_distribution_analysis(model, feature_extractor, test_track_ids, sr, n_mels, output_dir, model_config):
    print("\n" + "="*80); print("ANALYSIS 3: SIMILARITY DISTRIBUTION"); print("="*80)
    device = next(model.parameters()).device; model.eval()
    intra_class_sims, inter_class_sims = [], []

    print("Generating embeddings for distance analysis...")
    song_embeddings = {}
    for tid in tqdm(test_track_ids, desc="Generating embeddings"):
        path = os.path.join(TEST_DATA_DIR, f"{tid}.mp3")
        clip1 = get_embedding(_load_random_clip(path, DB_CLIP_DURATION, sr), model, feature_extractor, sr, n_mels, device)
        clip2 = get_embedding(_load_random_clip(path, DB_CLIP_DURATION, sr), model, feature_extractor, sr, n_mels, device)
        song_embeddings[tid] = (clip1, clip2)

    print("Calculating similarities...")
    track_list = list(song_embeddings.keys())
    for i in range(len(track_list)):
        tid1 = track_list[i]; z1_1, z1_2 = song_embeddings[tid1]
        intra_class_sims.append(F.cosine_similarity(z1_1, z1_2).item())
        j = i
        while j == i: j = random.randint(0, len(track_list) - 1)
        tid2 = track_list[j]; z2_1 = song_embeddings[tid2][0]
        inter_class_sims.append(F.cosine_similarity(z1_1, z2_1).item())

    plt.figure(figsize=(10, 6)); sns.histplot(intra_class_sims, color="blue", label='Intra-Class (Same Song)', kde=True, stat="density", element="step")
    sns.histplot(inter_class_sims, color="red", label='Inter-Class (Different Songs)', kde=True, stat="density", element="step")
    plt.title(f'Distribution of Embedding Similarities\n({model_config["display_name"]})'); plt.xlabel('Cosine Similarity'); plt.legend()
    save_path = os.path.join(output_dir, 'similarity_distributions.png')
    plt.savefig(save_path); print(f"Distance distribution plot saved to {save_path}"); plt.close()

def run_similarity_matrix_analysis(model, feature_extractor, test_track_ids, sr, n_mels, output_dir, model_config, songs_to_plot):
    print("\n" + "="*80); print("ANALYSIS 4: SIMILARITY MATRIX HEATMAP"); print("="*80)
    device = next(model.parameters()).device; model.eval()

    embeddings, labels = [], []
    for tid in songs_to_plot:
        path = os.path.join(TEST_DATA_DIR, f"{str(tid)}.mp3")
        for i in range(HEATMAP_CLIPS_PER_SONG):
            y_clip = _load_random_clip(path, DB_CLIP_DURATION, sr)
            embeddings.append(get_embedding(y_clip, model, feature_extractor, sr, n_mels, device))
            labels.append(f"{str(tid)[:5]}-{i+1}")

    embeddings_cat = torch.cat(embeddings, dim=0); sim_matrix = F.cosine_similarity(embeddings_cat.unsqueeze(1), embeddings_cat.unsqueeze(0), dim=-1)
    plt.figure(figsize=(12, 10)); sns.heatmap(sim_matrix.numpy(), xticklabels=labels, yticklabels=labels, cmap='viridis', annot=False)
    plt.title(f'Cosine Similarity Matrix of Sample Clips\n({model_config["display_name"]})'); plt.xticks(rotation=90); plt.yticks(rotation=0)
    plt.tight_layout(); save_path = os.path.join(output_dir, 'similarity_heatmap.png')
    plt.savefig(save_path); print(f"Similarity heatmap saved to {save_path}"); plt.close()

def run_hard_mining_analysis(model, feature_extractor, test_track_ids, sr, n_mels, output_dir):
    print("\n" + "="*80); print("ANALYSIS 5: HARD-CASE MINING"); print("="*80)
    device = next(model.parameters()).device; model.eval()

    print("Building reference database for hard-case mining...")
    embeddings_db = {}
    for tid in tqdm(test_track_ids, desc="Building DB"):
        path = os.path.join(TEST_DATA_DIR, f"{tid}.mp3")
        clip_embs = [get_embedding(_load_random_clip(path, DB_CLIP_DURATION, sr), model, feature_extractor, sr, n_mels, device) for _ in range(5)]
        embeddings_db[tid] = torch.mean(torch.cat(clip_embs, dim=0), dim=0)

    print("Finding hardest negative examples..."); hard_negatives = []
    for query_tid in tqdm(test_track_ids, desc="Mining Hard Negatives"):
        y_query = _load_random_clip(os.path.join(TEST_DATA_DIR, f"{query_tid}.mp3"), DB_CLIP_DURATION, sr)
        zq = get_embedding(y_query, model, feature_extractor, sr, n_mels, device)
        sims = {tid: F.cosine_similarity(zq, emb.unsqueeze(0)).item() for tid, emb in embeddings_db.items()}
        sims.pop(query_tid, None)
        if sims:
            hardest_negative_tid = max(sims, key=sims.get)
            hard_negatives.append((query_tid, hardest_negative_tid, sims[hardest_negative_tid]))
    hard_negatives_df = pd.DataFrame(hard_negatives, columns=['Query_Song_ID', 'Most_Confused_With', 'Similarity_Score']).sort_values(by='Similarity_Score', ascending=False).head(NUM_HARD_CASES_TO_SHOW)

    print("Finding hardest positive examples..."); hard_positives = []
    for tid in tqdm(test_track_ids, desc="Mining Hard Positives"):
        path = os.path.join(TEST_DATA_DIR, f"{tid}.mp3")
        embeddings = [get_embedding(_load_random_clip(path, DB_CLIP_DURATION, sr), model, feature_extractor, sr, n_mels, device) for _ in range(10)]
        embeddings_cat = torch.cat(embeddings, dim=0)
        sim_matrix = F.cosine_similarity(embeddings_cat.unsqueeze(1), embeddings_cat.unsqueeze(0), dim=-1)
        sim_matrix.fill_diagonal_(1.0)
        min_sim_val, _ = torch.min(sim_matrix.view(-1), 0)
        hard_positives.append((tid, min_sim_val.item()))
    hard_positives_df = pd.DataFrame(hard_positives, columns=['Song_ID', 'Lowest_Intra-Song_Similarity']).sort_values(by='Lowest_Intra-Song_Similarity', ascending=True).head(NUM_HARD_CASES_TO_SHOW)

    print("\nHard Negative Analysis Results (Top Confusions)"); print(hard_negatives_df)
    hard_negatives_df.to_csv(os.path.join(output_dir, 'hard_negatives_analysis.csv'), index=False)
    print("\nHard Positive Analysis Results (Most Dissimilar Clips)"); print(hard_positives_df)
    hard_positives_df.to_csv(os.path.join(output_dir, 'hard_positives_analysis.csv'), index=False)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    os.makedirs(MASTER_OUTPUT_DIR, exist_ok=True)

    try:
        test_metadata = pd.read_csv(TEST_METADATA_CSV)
        test_track_ids = sorted(test_metadata['track_id'].astype(str).unique().tolist())
    except Exception as e:
        print(f"ERROR: Could not read test metadata at {TEST_METADATA_CSV}: {e}"); return

    songs_for_viz = SONGS_FOR_VISUALIZATION
    if not songs_for_viz:
        num_to_sample = min(T_SNE_SONGS_TO_PLOT, len(test_track_ids))
        if num_to_sample > 0:
            songs_for_viz = random.sample(test_track_ids, num_to_sample)
            print(f"No specific songs provided. Randomly selected {len(songs_for_viz)} songs for visualization.")
        else:
            print("Not enough songs in test set for visualization.")

    summary_results = {}
    for model_config in MODELS_TO_EVALUATE:
        display_name = model_config["display_name"]
        model_type = model_config["model_type"]
        model_path = model_config["model_path"]
        sr = model_config["sr"]

        model_output_dir = os.path.join(MASTER_OUTPUT_DIR, display_name)
        os.makedirs(model_output_dir, exist_ok=True)

        print("\n\n" + "#"*80); print(f"# EVALUATING MODEL: {display_name}"); print("#"*80)

        model, feature_extractor = None, None
        try:
            print(f"Loading architecture for {display_name}...")
            if model_type == "MERT":
                model = MERTForAnalysis(**model_config)
                feature_extractor = AutoFeatureExtractor.from_pretrained(model_config["model_hf_path"], trust_remote_code=True)
            elif model_type == "AST":
                model = ASTForAnalysis(**model_config)
                feature_extractor = AutoFeatureExtractor.from_pretrained(model_config["model_hf_path"])
            elif model_type == "CNN":
                cnn_arch = model_config["cnn_arch"]
                if cnn_arch in globals():
                    model = globals()[cnn_arch](**model_config)
                else: raise NameError(f"CNN architecture '{cnn_arch}' not found.")

            full_state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(full_state_dict, strict=False)
            model.to(device)
            model.eval()
            print(f"Successfully loaded model weights from {model_path}")

        except Exception as e:
            print(f"\nERROR: Could not load model '{display_name}'. {e}"); continue

        key_metric = run_robustness_analysis(model, feature_extractor, test_track_ids, sr, model_config.get("n_mels", 128), model_output_dir)

        songs_for_heatmap = songs_for_viz[:min(len(songs_for_viz), HEATMAP_SONGS_TO_PLOT)]

        run_tsne_analysis(model, feature_extractor, test_metadata, sr, model_config.get("n_mels", 128), model_output_dir, model_config, songs_for_viz)
        run_distance_distribution_analysis(model, feature_extractor, test_track_ids, sr, model_config.get("n_mels", 128), model_output_dir, model_config)
        run_similarity_matrix_analysis(model, feature_extractor, test_track_ids, sr, model_config.get("n_mels", 128), model_output_dir, model_config, songs_for_heatmap)
        run_hard_mining_analysis(model, feature_extractor, test_track_ids, sr, model_config.get("n_mels", 128), model_output_dir)

        summary_results[display_name] = float(key_metric.replace('%', ''))

    if summary_results:
        print("\n\n" + "="*80); print("ANALYSIS 6: OVERALL MODEL PERFORMANCE COMPARISON"); print("="*80)
        summary_df = pd.DataFrame(list(summary_results.items()), columns=['Model', 'Top-5 Accuracy (%)'])
        summary_df = summary_df.sort_values(by='Top-5 Accuracy (%)', ascending=False)
        plt.figure(figsize=(12, max(6, len(summary_df) * 0.8)))
        ax = sns.barplot(x='Top-5 Accuracy (%)', y='Model', data=summary_df, palette='viridis', orient='h')
        ax.set_xlabel('Top-5 Accuracy (%) on Clean, 5s Clips', fontsize=12)
        ax.set_ylabel('Model', fontsize=12); ax.set_title('Overall Model Performance Comparison', fontsize=16)
        plt.xlim(0, 100)
        for container in ax.containers: ax.bar_label(container, fmt='%.2f%%', padding=3)
        plt.tight_layout()
        save_path = os.path.join(MASTER_OUTPUT_DIR, 'overall_performance_comparison.png')
        plt.savefig(save_path); print(f"Final summary plot saved to {save_path}"); plt.close()

    print("\n\nfinished")


if __name__ == "__main__":
    main()

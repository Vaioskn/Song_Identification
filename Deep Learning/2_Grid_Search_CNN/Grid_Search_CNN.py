import os
import csv
import pickle
import random
import itertools
import time
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = 'audio_1000'
METADATA_CSV = os.path.join(DATA_DIR, 'metadata.csv')
LABELMAP_PATH = 'label_map_gpu_grid_search.pkl'
TEMP_MODEL_PATH = 'temp_best_model_for_trial.pt' 

SR = 22050
CLIP_DURATION = 5.0
VALIDATION_SPLIT = 0.15
N_REFERENCE_CLIPS = 5
PATIENCE_LR = 5
PATIENCE_EARLY_STOP = 10

def compute_log_mel(y, sr=SR):
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=64)
    logm = librosa.power_to_db(melspec)
    logm = (logm - logm.mean()) / (logm.std() + 1e-6)
    return torch.from_numpy(logm).unsqueeze(0)

def info_nce_loss(z1, z2, temp):
    B = z1.size(0)
    if B == 0:
        return torch.tensor(0.0, device=z1.device, requires_grad=True)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temp
    mask = torch.eye(2 * B, device=sim.device).bool()
    sim = sim.masked_fill(mask, -torch.finfo(sim.dtype).max)
    labels_np = np.concatenate([np.arange(B, 2 * B), np.arange(0, B)])
    labels = torch.from_numpy(labels_np).long().to(sim.device)
    return F.cross_entropy(sim, labels)

class ContrastiveAudioDataset(Dataset):
    def __init__(self, csv_file, data_dir, clip_duration, sr, track_ids_for_this_set, id2idx, mode='train'):
        self.data_dir = data_dir
        self.clip_dur = clip_duration
        self.sr = sr
        self.id2idx = id2idx
        self.mode = mode
        all_samples_temp = []
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_samples_temp.append(row['track_id'])
        self.samples = [tid for tid in all_unique_track_ids_global if tid in track_ids_for_this_set]
        if not self.samples:
            print(f"Warning: No samples found for mode {self.mode}...")

    def __len__(self):
        return len(self.samples)

    def _load_random_clip(self, track_id):
        path = os.path.join(self.data_dir, f"{track_id}.mp3")
        try:
            y, _ = librosa.load(path, sr=self.sr, mono=True)
        except Exception as e:
            print(f"Error loading audio file {path}: {e}. Returning silent clip.")
            return np.zeros(int(self.clip_dur * self.sr), dtype=np.float32)

        total_duration_sec = len(y) / self.sr
        target_len_samples = int(self.clip_dur * self.sr)
        if total_duration_sec > self.clip_dur:
            start_time_sec = random.uniform(0, total_duration_sec - self.clip_dur)
            i0 = int(start_time_sec * self.sr)
            y_clip = y[i0:i0 + target_len_samples]
        else:
            y_clip = np.pad(y, (0, max(0, target_len_samples - len(y))), mode='constant')
        
        if len(y_clip) != target_len_samples:
             y_clip = np.pad(y_clip, (0, target_len_samples - len(y_clip)), mode='constant')[:target_len_samples]
        return y_clip

    def __getitem__(self, idx):
        tid = self.samples[idx]
        label = self.id2idx[tid]
        y1 = self._load_random_clip(tid)
        y2 = self._load_random_clip(tid)
        x1 = compute_log_mel(y1, self.sr)
        x2 = compute_log_mel(y2, self.sr)
        return x1, x2, label

class Encoder(nn.Module):
    def __init__(self, emb_dim, n_classes=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, emb_dim)
        )
        self.classifier = nn.Linear(emb_dim, n_classes) if n_classes > 0 else None

    def forward(self, x):
        h = self.conv(x)
        h = h.view(x.size(0), -1)
        z = self.proj(h)
        z = F.normalize(z, dim=1)
        logits = self.classifier(z) if self.classifier else None
        return z, logits

@torch.no_grad()
def evaluate_accuracy(model, val_ds, device):
    if not val_ds or len(val_ds.samples) == 0:
        return 0.0

    model.eval()

    print("[Eval] Building reference database...")
    embeddings_db = {}
    for track_id in tqdm(val_ds.samples, desc="[Eval] Building DB"):
        clip_embs = []
        for _ in range(N_REFERENCE_CLIPS):
            y_clip = val_ds._load_random_clip(track_id)
            x_clip = compute_log_mel(y_clip, sr=val_ds.sr)
            
            x_clip = x_clip.unsqueeze(0).to(device)
            
            z, _ = model(x_clip)
            clip_embs.append(z)
        
        if clip_embs:
            embeddings_db[track_id] = torch.mean(torch.cat(clip_embs, dim=0), dim=0).cpu()

    if not embeddings_db:
        return 0.0

    print("[Eval] Querying database...")
    correct_predictions = 0
    for query_tid in tqdm(val_ds.samples, desc="[Eval] Querying"):
        
        y_query = val_ds._load_random_clip(query_tid)
        x_query = compute_log_mel(y_query, sr=val_ds.sr)
        
        x_query = x_query.unsqueeze(0).to(device)
        
        zq, _ = model(x_query)
        zq_cpu = zq.cpu()

        sims = {tid: F.cosine_similarity(zq_cpu, emb.unsqueeze(0)).item() for tid, emb in embeddings_db.items()}
        predicted_tid = max(sims, key=sims.get)
        
        if predicted_tid == query_tid:
            correct_predictions += 1
    
    accuracy = (correct_predictions / len(val_ds.samples)) * 100 if len(val_ds.samples) > 0 else 0.0
    return accuracy

def run_training_trial(config, trial_id, all_unique_track_ids, id2idx_map, num_workers_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n--- Starting Trial {trial_id + 1}: {config} ---")
    print(f"Trial {trial_id + 1} using device: {device}")

    train_ids, val_ids = train_test_split(all_unique_track_ids, test_size=VALIDATION_SPLIT, random_state=42)
    
    train_ds = ContrastiveAudioDataset(METADATA_CSV, DATA_DIR, CLIP_DURATION, SR, train_ids, id2idx_map, mode='train')
    val_ds = ContrastiveAudioDataset(METADATA_CSV, DATA_DIR, CLIP_DURATION, SR, val_ids, id2idx_map, mode='val')

    if len(train_ds) == 0:
        return float('inf'), 0.0

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=num_workers_loader, pin_memory=True if device=='cuda' else False)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, drop_last=False, num_workers=num_workers_loader, pin_memory=True if device=='cuda' else False) if len(val_ds) > 0 else None

    model = Encoder(emb_dim=config['emb_dim'], n_classes=len(id2idx_map)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=PATIENCE_LR)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    model_saved_for_trial = False

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Trial {trial_id+1} Ep {epoch} [Train]", leave=False, ncols=100)
        for x1, x2, labels in train_pbar:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            opt.zero_grad()
            z1, log1 = model(x1)
            z2, log2 = model(x2)
            loss_c = info_nce_loss(z1, z2, temp=config['temp'])
            loss_ce = F.cross_entropy(log1, labels) + F.cross_entropy(log2, labels)
            loss = loss_c + config['alpha'] * loss_ce
            loss.backward()
            opt.step()
            total_train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else float('inf')

        current_val_loss = float('inf')
        if val_loader and len(val_loader) > 0:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for x1_val, x2_val, labels_val in val_loader:
                    x1_val, x2_val, labels_val = x1_val.to(device), x2_val.to(device), labels_val.to(device)
                    z1_val, log1_val = model(x1_val)
                    z2_val, log2_val = model(x2_val)
                    loss_c_val = info_nce_loss(z1_val, z2_val, temp=config['temp'])
                    loss_ce_val = F.cross_entropy(log1_val, labels_val) + F.cross_entropy(log2_val, labels_val)
                    loss_val = loss_c_val + config['alpha'] * loss_ce_val
                    total_val_loss += loss_val.item()
            current_val_loss = total_val_loss / len(val_loader)
            scheduler.step(current_val_loss)

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), TEMP_MODEL_PATH)
                model_saved_for_trial = True
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE_EARLY_STOP:
                    print(f"Trial {trial_id+1}: Early stopping at epoch {epoch}.")
                    break
        else:
            if not val_loader: best_val_loss = avg_train_loss

        val_loss_display = f"{current_val_loss:.4f}" if not np.isinf(current_val_loss) else "N/A"
        print(f"T{trial_id+1} E{epoch}: TrainLoss={avg_train_loss:.4f}, ValLoss={val_loss_display}")

    best_val_accuracy = 0.0
    if model_saved_for_trial:
        print(f"--- Evaluating accuracy for Trial {trial_id+1}'s best model ---")
        best_model_for_trial = Encoder(emb_dim=config['emb_dim'], n_classes=len(id2idx_map)).to(device)
        best_model_for_trial.load_state_dict(torch.load(TEMP_MODEL_PATH))
        best_val_accuracy = evaluate_accuracy(best_model_for_trial, val_ds, device)
        print(f"--- Accuracy for Trial {trial_id+1}: {best_val_accuracy:.2f}% ---")
    else:
        print("No best model saved for this trial, skipping accuracy evaluation.")

    print(f"--- Finished Trial {trial_id+1}: Best Val Loss = {best_val_loss:.4f}, Best Val Acc = {best_val_accuracy:.2f}% ---")
    return best_val_loss, best_val_accuracy

if __name__ == '__main__':
    main_start_time = time.time()
    if not os.path.isdir(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        exit()
    if not os.path.isfile(METADATA_CSV):
        print(f"Error: Metadata CSV '{METADATA_CSV}' not found.")
        exit()

    with open(METADATA_CSV, newline='') as f:
        reader = csv.DictReader(f)
        all_unique_track_ids_global = sorted(list(set(row['track_id'] for row in reader)))
    id2idx_global = {tid: i for i, tid in enumerate(all_unique_track_ids_global)}
    if not id2idx_global:
        raise ValueError("No track IDs found in metadata.csv.")
    with open(LABELMAP_PATH, 'wb') as f:
        pickle.dump(id2idx_global, f)
    print(f"Global Label map saved. Total unique tracks: {len(all_unique_track_ids_global)}")

    num_workers_loader = 0
    if torch.cuda.is_available():
        print(f"CUDA is available! PyTorch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
        num_workers_loader = min(os.cpu_count() if os.cpu_count() is not None else 1, 4)
    else:
        print("CUDA not available. Running on CPU.")

    param_grid = {
        'lr': [1e-4, 5e-4],
        'temp': [0.07, 0.1],
        'alpha': [0.25, 0.5, 0.75],
        'emb_dim': [128, 256],
        'epochs': [15],
        'batch_size': [32, 64]
    }
    keys, values = zip(*param_grid.items())
    hyperparameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"\nStarting grid search with {len(hyperparameter_combinations)} combinations...")

    results_list = []
    for i, current_config in enumerate(hyperparameter_combinations):
        trial_val_loss, trial_accuracy = run_training_trial(current_config, i, all_unique_track_ids_global, id2idx_global, num_workers_loader)
        
        result_entry = current_config.copy()
        result_entry['best_val_loss'] = trial_val_loss
        result_entry['best_val_accuracy'] = trial_accuracy
        results_list.append(result_entry)
        
        df_current_results = pd.DataFrame(results_list)
        df_current_results.to_csv('gpu_grid_search_results.csv', index=False)
        
        min_val_loss_display = "N/A"
        if 'best_val_loss' in df_current_results.columns and not df_current_results['best_val_loss'].isnull().all():
            valid_losses = df_current_results['best_val_loss'].replace([np.inf, -np.inf], np.nan).dropna()
            if not valid_losses.empty:
                min_val_loss_display = f"{valid_losses.min():.4f}"
        print(f"Results for trial {i+1}/{len(hyperparameter_combinations)} saved. Best val_loss so far: {min_val_loss_display}")

    print("\n--- Grid Search Finished ---")
    final_df_results = pd.DataFrame(results_list)
    final_df_results['best_val_loss'] = final_df_results['best_val_loss'].replace([np.inf, -np.inf], np.nan)
    
    print("\nTop 5 Hyperparameter Combinations (Sorted by Validation Loss):")
    print(final_df_results.sort_values(by='best_val_loss', ascending=True).head())

    print("\nTop 5 Hyperparameter Combinations (Sorted by Validation Accuracy):")
    print(final_df_results.sort_values(by='best_val_accuracy', ascending=False).head())
    
    final_df_results.to_csv('gpu_grid_search_results_final.csv', index=False)
    print("\nFull grid search results saved to 'gpu_grid_search_results_final.csv'")

    if not final_df_results.empty and not final_df_results['best_val_loss'].isnull().all():
        best_overall_config = final_df_results.sort_values(by='best_val_loss').iloc[0].to_dict()
        print(f"\nBest Hyperparameter Configuration Found (by loss):")
        for key, value in best_overall_config.items():
            print(f"  {key}: {value}")
    else:
        print("No valid results from grid search.")
        
    total_duration_minutes = (time.time() - main_start_time) / 60
    print(f"\nTotal Grid Search Duration: {total_duration_minutes:.2f} minutes")

    if os.path.exists(TEMP_MODEL_PATH):
        os.remove(TEMP_MODEL_PATH)

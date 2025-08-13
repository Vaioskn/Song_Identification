import os
import csv
import pickle
import random
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
import torchaudio
import torchaudio.transforms as T

DATA_DIR = '..//audio_1000'
METADATA_CSV = os.path.join(DATA_DIR, 'metadata.csv')
MODEL_PATH = 'contrastive_model_v3_best.pt'
LABELMAP_PATH = 'label_map_v3.pkl'

SR = 22050
CLIP_DURATION = 5.0
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
TEMP = 0.07
ALPHA = 0.25
EMB_DIM = 256

VALIDATION_SPLIT = 0.2
N_REFERENCE_CLIPS= 5
PATIENCE_LR = 5
PATIENCE_EARLY_STOP = 10
NUM_WORKERS = 0

FREQ_MASK_PARAM = 20
TIME_MASK_PARAM = 40

def compute_log_mel(y, sr=SR):
    melspec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=64
    )
    logm = librosa.power_to_db(melspec)
    logm = (logm - logm.mean()) / (logm.std() + 1e-6)
    return torch.from_numpy(logm).unsqueeze(0)

spec_augment_transforms = nn.Sequential(
    T.FrequencyMasking(freq_mask_param=FREQ_MASK_PARAM),
    T.TimeMasking(time_mask_param=TIME_MASK_PARAM)
)

class ContrastiveAudioDataset(Dataset):
    def __init__(self, csv_file, data_dir, clip_duration, sr, track_ids_for_this_set, id2idx, augment=False):
        self.data_dir = data_dir
        self.clip_dur = clip_duration
        self.sr = sr
        self.id2idx = id2idx
        self.augment = augment

        all_samples_temp = []
        try:
            with open(csv_file, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_samples_temp.append(row['track_id'])
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata CSV not found at {csv_file}")
        
        self.samples = [tid for tid in all_samples_temp if tid in track_ids_for_this_set]
        
        if not self.samples:
            print(f"Warning: No samples found for the provided track IDs.")

    def __len__(self):
        return len(self.samples)

    def _load_random_clip(self, track_id):
        path = os.path.join(self.data_dir, f"{track_id}.mp3")
        target_len_samples = int(self.clip_dur * self.sr)
        try:
            y, loaded_sr = librosa.load(path, sr=self.sr, mono=True)
            if loaded_sr != self.sr:
                 y = librosa.resample(y, orig_sr=loaded_sr, target_sr=self.sr)
        except Exception as e:
            print(f"Error loading audio file {path}: {e}. Returning silent clip.")
            return np.zeros(target_len_samples, dtype=np.float32)

        total_duration_sec = len(y) / self.sr

        if total_duration_sec > self.clip_dur:
            start_time_sec = random.uniform(0, total_duration_sec - self.clip_dur)
            i0 = int(start_time_sec * self.sr)
            i1 = i0 + target_len_samples
            y_clip = y[i0:i1]
        else:
            y_clip = np.pad(y, (0, max(0, target_len_samples - len(y))), mode='constant')
        
        if len(y_clip) < target_len_samples:
            y_clip = np.pad(y_clip, (0, target_len_samples - len(y_clip)), mode='constant')
        elif len(y_clip) > target_len_samples:
            y_clip = y_clip[:target_len_samples]
        return y_clip

    def __getitem__(self, idx):
        tid = self.samples[idx]
        label = self.id2idx[tid]
        
        y1 = self._load_random_clip(tid)
        y2 = self._load_random_clip(tid)
        
        x1 = compute_log_mel(y1, self.sr)
        x2 = compute_log_mel(y2, self.sr)
        
        if self.augment:
            x1 = spec_augment_transforms(x1)
            x2 = spec_augment_transforms(x2)
            
        return x1, x2, label

class Encoder(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, n_classes=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.proj = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, emb_dim)
        )
        self.classifier = nn.Linear(emb_dim, n_classes) if n_classes > 0 else None

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        z = F.normalize(self.proj(h), dim=1)
        logits = self.classifier(z) if self.classifier else None
        return z, logits

def info_nce_loss(z1, z2, temp=TEMP):
    B = z1.size(0)
    if B == 0: return torch.tensor(0.0, device=z1.device, requires_grad=True)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temp
    mask = torch.eye(2*B, device=sim.device).bool()
    sim = sim.masked_fill(mask, -torch.finfo(sim.dtype).max)
    
    labels_np = np.concatenate([np.arange(B, 2*B), np.arange(0, B)])
    labels = torch.from_numpy(labels_np).long().to(sim.device)
    
    return F.cross_entropy(sim, labels)

def train_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"PyTorch CUDA version: {torch.version.cuda}")

    with open(METADATA_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_unique_track_ids = sorted(list(set(row['track_id'] for row in reader)))
    
    if not all_unique_track_ids:
        raise ValueError("No track IDs found in metadata.csv. Cannot train.")
    
    id2idx = {tid: i for i, tid in enumerate(all_unique_track_ids)}
    with open(LABELMAP_PATH, 'wb') as f:
        pickle.dump(id2idx, f)
    print(f"Label map saved to {LABELMAP_PATH} with {len(id2idx)} unique tracks.")

    train_ids, val_ids = train_test_split(all_unique_track_ids, test_size=VALIDATION_SPLIT, random_state=42, stratify=None)
    print(f"Total unique tracks: {len(all_unique_track_ids)}")
    print(f"Training tracks (IDs): {len(train_ids)}, Validation tracks (IDs): {len(val_ids)}")

    train_ds = ContrastiveAudioDataset(METADATA_CSV, DATA_DIR, CLIP_DURATION, SR, train_ids, id2idx, augment=True)
    val_ds = ContrastiveAudioDataset(METADATA_CSV, DATA_DIR, CLIP_DURATION, SR, val_ids, id2idx, augment=False)

    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty. Check metadata, data directory, and splits.")
    
    pin_memory_flag = True if device == 'cuda' else False
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS, pin_memory=pin_memory_flag)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS, pin_memory=pin_memory_flag) if len(val_ds) > 0 else None

    model = Encoder(emb_dim=EMB_DIM, n_classes=len(id2idx)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=PATIENCE_LR)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", ncols=100)
        for x1, x2, labels in train_pbar:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            
            opt.zero_grad()
            z1, log1 = model(x1)
            z2, log2 = model(x2)
            
            loss_c = info_nce_loss(z1, z2)
            loss_ce = F.cross_entropy(log1, labels) + F.cross_entropy(log2, labels)
            loss = loss_c + ALPHA * loss_ce

            loss.backward()
            opt.step()
            
            total_train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else float('inf')

        current_val_loss = float('inf')
        if val_loader and len(val_loader) > 0:
            model.eval()
            total_val_loss = 0.0
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]", ncols=100)
            with torch.no_grad():
                for x1_v, x2_v, labels_v in val_pbar:
                    x1_v, x2_v, labels_v = x1_v.to(device), x2_v.to(device), labels_v.to(device)
                    z1_v, log1_v = model(x1_v)
                    z2_v, log2_v = model(x2_v)
                    
                    loss_c_val = info_nce_loss(z1_v, z2_v)
                    loss_ce_val = F.cross_entropy(log1_v, labels_v) + F.cross_entropy(log2_v, labels_v)
                    loss_val = loss_c_val + ALPHA * loss_ce_val
                    total_val_loss += loss_val.item()
                    val_pbar.set_postfix(loss=f"{loss_val.item():.4f}")
            current_val_loss = total_val_loss / len(val_loader)
            scheduler.step(current_val_loss)

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"Epoch {epoch}: New best model saved to {MODEL_PATH} (Val Loss: {best_val_loss:.4f})")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
        
        epoch_duration = time.time() - epoch_start_time
        train_loss_disp = f"{avg_train_loss:.4f}" if not np.isinf(avg_train_loss) else "N/A"
        val_loss_disp = f"{current_val_loss:.4f}" if not np.isinf(current_val_loss) else "N/A"
        print(f"Epoch {epoch}/{EPOCHS} Summary: Train Loss: {train_loss_disp}, Val Loss: {val_loss_disp}, Duration: {epoch_duration:.2f}s")

        if epochs_no_improve >= PATIENCE_EARLY_STOP:
            print(f"Early stopping triggered after {PATIENCE_EARLY_STOP} epochs with no improvement on validation loss.")
            break
    
    print("Training finished.")
    
    if os.path.exists(MODEL_PATH):
        print(f"Loading best model from {MODEL_PATH} for evaluation.")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model path not found. Evaluating with the last model state.")
        
    return model, id2idx, val_ids

def build_track_embeddings(model, track_ids_to_embed, id2idx_map, data_dir=DATA_DIR, sr=SR, clip_duration=CLIP_DURATION, n_clips=N_REFERENCE_CLIPS):
    device = next(model.parameters()).device
    embeddings_db = {}
    
    model.eval()
    with torch.no_grad():
        for tid in tqdm(track_ids_to_embed, desc="Building Reference Embeddings"):
            path = os.path.join(data_dir, f"{tid}.mp3")
            target_len_samples = int(clip_duration * sr)
            try:
                y_full, loaded_sr = librosa.load(path, sr=sr, mono=True)
                if loaded_sr != sr: y_full = librosa.resample(y_full, orig_sr=loaded_sr, target_sr=sr)
            except Exception as e:
                print(f"Error loading {path} for embedding: {e}. Skipping.")
                continue

            if len(y_full) == 0:
                print(f"Warning: {path} is empty. Skipping.")
                continue

            clip_embeddings_list = []
            for _ in range(n_clips):
                total_duration_sec = len(y_full) / sr
                if total_duration_sec > clip_duration:
                    start_time_sec = random.uniform(0, total_duration_sec - clip_duration)
                    i0 = int(start_time_sec * sr)
                    i1 = i0 + target_len_samples
                    y_clip = y_full[i0:i1]
                else:
                    y_clip = np.pad(y_full, (0, max(0, target_len_samples - len(y_full))), mode='constant')
                
                if len(y_clip) < target_len_samples:
                     y_clip = np.pad(y_clip, (0, target_len_samples - len(y_clip)), mode='constant')
                elif len(y_clip) > target_len_samples:
                     y_clip = y_clip[:target_len_samples]

                x = compute_log_mel(y_clip, sr).unsqueeze(0).to(device)
                z, _ = model(x)
                clip_embeddings_list.append(z)
            
            if clip_embeddings_list:
                avg_z = torch.stack(clip_embeddings_list).mean(dim=0)
                embeddings_db[tid] = avg_z.cpu()
            else:
                print(f"Could not generate clips for {tid}. Skipping.")
    return embeddings_db

def evaluate_model(model, embeddings_to_search, query_track_ids, id2idx_map, data_dir=DATA_DIR, sr=SR, clip_duration=CLIP_DURATION):
    if not embeddings_to_search:
        print("Embeddings database is empty. Cannot evaluate.")
        return 0.0

    device = next(model.parameters()).device
    correct = 0
    total_evaluated = 0
    
    model.eval()
    with torch.no_grad():
        for query_tid in tqdm(query_track_ids, desc="Evaluating Model"):
            path = os.path.join(data_dir, f"{query_tid}.mp3")
            target_len_samples = int(clip_duration * sr)
            try:
                y_query_full, loaded_sr = librosa.load(path, sr=sr, mono=True)
                if loaded_sr != sr: y_query_full = librosa.resample(y_query_full, orig_sr=loaded_sr, target_sr=sr)

            except Exception as e:
                print(f"Error loading query track {path}: {e}. Skipping.")
                continue
            
            if len(y_query_full) == 0:
                print(f"Warning: Query track {path} is empty. Skipping.")
                continue


            total_duration_sec = len(y_query_full) / sr
            if total_duration_sec > clip_duration:
                start_time_sec = random.uniform(0, total_duration_sec - clip_duration)
                i0 = int(start_time_sec * sr)
                i1 = i0 + target_len_samples
                y_query_clip = y_query_full[i0:i1]
            else:
                y_query_clip = np.pad(y_query_full, (0, max(0, target_len_samples - len(y_query_full))), mode='constant')

            if len(y_query_clip) < target_len_samples:
                 y_query_clip = np.pad(y_query_clip, (0, target_len_samples - len(y_query_clip)), mode='constant')
            elif len(y_query_clip) > target_len_samples:
                 y_query_clip = y_query_clip[:target_len_samples]

            x_query = compute_log_mel(y_query_clip, sr).unsqueeze(0).to(device)
            zq_query, _ = model(x_query)
            zq_query_cpu = zq_query.cpu()

            sims = {}
            for db_tid, z_ref_db_cpu in embeddings_to_search.items():
                sims[db_tid] = torch.cosine_similarity(zq_query_cpu, z_ref_db_cpu).item()

            if not sims: continue

            pred_tid = max(sims, key=sims.get)
            
            if pred_tid == query_tid:
                correct += 1
            total_evaluated += 1
            
    accuracy = 0.0
    if total_evaluated > 0:
        accuracy = correct / total_evaluated
        print(f"Top-1 Accuracy on {len(query_track_ids)} query tracks (vs. {len(embeddings_to_search)} in DB): {accuracy:.2%}")
    else:
        print("No tracks were evaluated.")
    return accuracy

if __name__=='__main__':
    overall_start_time = time.time()

    if not os.path.isdir(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found. Please check the path.")
        exit()
    if not os.path.isfile(METADATA_CSV):
        print(f"Error: Metadata CSV '{METADATA_CSV}' not found in '{DATA_DIR}'.")
        exit()
    
    if os.name != 'posix':
        NUM_WORKERS = 0
        print(f"Running on non-POSIX OS, setting NUM_WORKERS to 0 for DataLoader.")
    elif torch.cuda.is_available():
         NUM_WORKERS = min(os.cpu_count() if os.cpu_count() is not None else 1, 4)
         print(f"CUDA available, setting NUM_WORKERS to {NUM_WORKERS}.")
    else:
         NUM_WORKERS = min(os.cpu_count() if os.cpu_count() is not None else 1, 4)
         print(f"CUDA not available, setting NUM_WORKERS to {NUM_WORKERS}.")


    trained_model, global_id2idx, validation_ids = train_model()
    
    print("\nBuilding track embeddings for the validation set...")
    validation_embeddings = build_track_embeddings(trained_model, validation_ids, global_id2idx) 
    
    if validation_embeddings:
        print(f"\nEvaluating model on the {len(validation_ids)} validation tracks...")
        evaluate_model(trained_model, validation_embeddings, validation_ids, global_id2idx)
    else:
        print("Could not build reference embeddings for the validation set. Skipping evaluation.")
    
    overall_duration = time.time() - overall_start_time
    print(f"\nScript finished in {overall_duration/60:.2f} minutes.")
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
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoFeatureExtractor
from audiomentations import Compose, TimeStretch, PitchShift, AddGaussianNoise, Gain
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = '..//audio_1000'
METADATA_CSV = os.path.join(DATA_DIR, 'metadata.csv')
DRIVE_OUTPUT_DIR = 'Audio_Project_AST_SOTA_Results'
os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(DRIVE_OUTPUT_DIR, 'finetuned_ast_sota_model.pt')
LABELMAP_PATH = os.path.join(DRIVE_OUTPUT_DIR, 'finetuned_ast_sota_label_map.pkl')

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR = 16000
CLIP_DURATION = 10.0
BATCH_SIZE = 12
EPOCHS = 15
LR_HEAD = 1e-4
LR_BACKBONE = 5e-5
ALPHA = 0.5 
LABEL_SMOOTHING = 0.1
EMB_DIM = 256
VALIDATION_SPLIT = 0.2
GRADIENT_CLIP_VAL = 1.0
WARMUP_RATIO = 0.1

ACCUMULATION_STEPS = 4 
NUM_DROPOUT_SAMPLES = 4

class WaveformDataset(Dataset):
    def __init__(self, csv_file, data_dir, track_ids, id2idx, feature_extractor, augment=False):
        self.data_dir = data_dir
        self.id2idx = id2idx
        self.feature_extractor = feature_extractor
        self.augment = augment
        with open(csv_file, newline='', encoding='utf-8') as f:
            all_tracks = [row['track_id'] for row in csv.DictReader(f)]
        self.samples = [tid for tid in all_tracks if tid in track_ids]
        if self.augment:
            self.augmentation_pipeline = Compose([
                TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3),
                PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
                Gain(min_gain_db=-6, max_gain_db=6, p=0.3)
            ])

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_random_clip(path, duration=CLIP_DURATION, sr=SR):
        target_len = int(duration * sr)
        try:
            y, _ = librosa.load(path, sr=sr, mono=True)
        except Exception as e:
            return np.zeros(target_len, dtype=np.float32)
        if len(y) > target_len:
            start = random.randint(0, len(y) - target_len)
            y = y[start:start + target_len]
        else:
            y = np.pad(y, (0, target_len - len(y)), 'constant')
        return y

    def __getitem__(self, idx):
        tid = self.samples[idx]
        label = self.id2idx[tid]
        path = os.path.join(self.data_dir, f"{tid}.mp3")
        y1 = self._load_random_clip(path)
        y2 = self._load_random_clip(path)
        if self.augment:
            y1 = self.augmentation_pipeline(samples=y1, sample_rate=SR)
            y2 = self.augmentation_pipeline(samples=y2, sample_rate=SR)
        inputs1 = self.feature_extractor(y1, sampling_rate=SR, return_tensors="pt")
        inputs2 = self.feature_extractor(y2, sampling_rate=SR, return_tensors="pt")
        return inputs1['input_values'].squeeze(0), inputs2['input_values'].squeeze(0), label

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
        weighted_average = torch.sum(x * attention_weights, dim=1)
        return weighted_average

class ASTForSongID(nn.Module):
    def __init__(self, model_name, emb_dim, n_classes, n_dropout_samples):
        super().__init__()
        print(f"Loading pretrained model: {model_name}")
        self.base_model = AutoModel.from_pretrained(model_name)
        ast_embedding_size = self.base_model.config.hidden_size
        self.n_dropout_samples = n_dropout_samples
        
        self.temperature = nn.Parameter(torch.tensor(0.07))

        self.pooling = AttentionPooling(in_features=ast_embedding_size)
        
        self.proj_head = nn.Sequential(
            nn.Linear(ast_embedding_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, emb_dim)
        )
        self.classifier = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        ast_embedding = self.pooling(self.base_model(x).last_hidden_state)
        
        if self.training and self.n_dropout_samples > 1:
            z_unnorm_samples = torch.stack(
                [self.proj_head(ast_embedding) for _ in range(self.n_dropout_samples)],
                dim=0
            )
            z_unnorm = torch.mean(z_unnorm_samples, dim=0)
        else:
            z_unnorm = self.proj_head(ast_embedding)
            
        z_norm = F.normalize(z_unnorm, p=2, dim=1)
        logits = self.classifier(z_unnorm)
        return z_norm, logits

def info_nce_loss(z1, z2, temp):
    temp = torch.clamp(temp, min=0.01)
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temp
    mask = torch.eye(2 * B, device=sim.device).bool()
    sim = sim.masked_fill(mask, -torch.finfo(sim.dtype).max)
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)]).long().to(sim.device)
    return F.cross_entropy(sim, labels)

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * (current_step - warmup_steps) / (total_steps - warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)

def train_and_evaluate():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    with open(METADATA_CSV, newline='', encoding='utf-8') as f:
        all_unique_track_ids = sorted(list(set(row['track_id'] for row in csv.DictReader(f))))
    id2idx = {tid: i for i, tid in enumerate(all_unique_track_ids)}
    with open(LABELMAP_PATH, 'wb') as f: pickle.dump(id2idx, f)
    print(f"Saved label map with {len(id2idx)} tracks to {LABELMAP_PATH}")

    train_ids, val_ids = train_test_split(all_unique_track_ids, test_size=VALIDATION_SPLIT, random_state=42)
    train_ds = WaveformDataset(METADATA_CSV, DATA_DIR, train_ids, id2idx, feature_extractor, augment=True)
    val_ds = WaveformDataset(METADATA_CSV, DATA_DIR, val_ids, id2idx, feature_extractor, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = ASTForSongID(
        model_name=MODEL_NAME, emb_dim=EMB_DIM, n_classes=len(id2idx), n_dropout_samples=NUM_DROPOUT_SAMPLES
    ).to(device)

    param_groups = [
        {'params': model.base_model.parameters(), 'lr': LR_BACKBONE},
        {'params': model.pooling.parameters(), 'lr': LR_HEAD},
        {'params': model.proj_head.parameters(), 'lr': LR_HEAD},
        {'params': model.classifier.parameters(), 'lr': LR_HEAD},
        {'params': [model.temperature], 'lr': 1e-3} 
    ]
    optimizer = torch.optim.AdamW(param_groups)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))
    best_val_loss = float('inf')

    print(f"Starting Final SOTA AST fine-tuning for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Finetune]", ncols=100)
        optimizer.zero_grad() 

        for i, (y1, y2, labels) in enumerate(train_pbar):
            y1, y2, labels = y1.to(device), y2.to(device), labels.to(device)
            
            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                z1, log1 = model(y1)
                z2, log2 = model(y2)
                loss_c = info_nce_loss(z1, z2, model.temperature)
 
                loss_ce = F.cross_entropy(log1, labels, label_smoothing=LABEL_SMOOTHING) + \
                          F.cross_entropy(log2, labels, label_smoothing=LABEL_SMOOTHING)
                loss = loss_c + ALPHA * loss_ce
                
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_train_loss += loss.item() * ACCUMULATION_STEPS 
            train_pbar.set_postfix(loss=f"{loss.item() * ACCUMULATION_STEPS:.4f}", temp=f"{model.temperature.item():.3f}")
        
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for y1_v, y2_v, labels_v in val_loader:
                y1_v, y2_v, labels_v = y1_v.to(device), y2_v.to(device), labels_v.to(device)
                with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                    z1_v, log1_v = model(y1_v)
                    z2_v, log2_v = model(y2_v)
                    loss_c_val = info_nce_loss(z1_v, z2_v, model.temperature)
                    loss_ce_val = F.cross_entropy(log1_v, labels_v) + F.cross_entropy(log2_v, labels_v)
                    val_loss = loss_c_val + ALPHA * loss_ce_val
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch} Summary: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Temp: {model.temperature.item():.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"New best model saved to {MODEL_PATH}")

    print("\nFine-tuning Finished. Starting Final Evaluation.")
    if os.path.exists(MODEL_PATH):
        print(f"Loading best model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("Warning: No saved model found. Evaluating with last epoch model.")

    evaluate_final_model(model, feature_extractor, val_ids, id2idx)

def evaluate_final_model(model, feature_extractor, query_track_ids, id2idx, n_clips_ref=5):
    device = next(model.parameters()).device
    model.eval()
    embeddings_db = {}
    print("\nBuilding reference embeddings for the validation set.")
    with torch.no_grad():
        for tid in tqdm(query_track_ids, desc="Building DB"):
            path = os.path.join(DATA_DIR, f"{tid}.mp3")
            clip_embs = []
            for _ in range(n_clips_ref):
                y_clip = WaveformDataset._load_random_clip(path)
                inputs = feature_extractor(y_clip, sampling_rate=SR, return_tensors="pt").to(device)
                z, _ = model(inputs['input_values'])
                clip_embs.append(z)
            if clip_embs:
                embeddings_db[tid] = torch.stack(clip_embs).mean(dim=0).cpu()

    correct_predictions = 0
    print("\nEvaluating Top-1 retrieval accuracy.")
    with torch.no_grad():
        for query_tid in tqdm(query_track_ids, desc="Querying"):
            y_query = WaveformDataset._load_random_clip(os.path.join(DATA_DIR, f"{query_tid}.mp3"))
            inputs_query = feature_extractor(y_query, sampling_rate=SR, return_tensors="pt").to(device)
            zq, _ = model(inputs_query['input_values'])
            zq_cpu = zq.cpu()
            sims = {tid: F.cosine_similarity(zq_cpu, emb).item() for tid, emb in embeddings_db.items()}
            predicted_tid = max(sims, key=sims.get)
            if predicted_tid == query_tid:
                correct_predictions += 1

    accuracy = (correct_predictions / len(query_track_ids)) if query_track_ids else 0.0
    print("\n" + "="*50)
    print(f"Final Top-1 Accuracy (AST SOTA Fine-tuned): {accuracy:.2%}")
    print("="*50)

if __name__ == '__main__':
    train_and_evaluate()
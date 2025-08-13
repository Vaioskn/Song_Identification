import os
import csv
import pickle
import random
import time
import itertools
import traceback

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torchaudio.transforms as T
from audiomentations import Compose, TimeStretch, PitchShift, AddGaussianNoise, Gain, Mp3Compression

DATA_DIR = 'audio_1000'
METADATA_CSV = os.path.join(DATA_DIR, 'metadata.csv')
RESULTS_CSV_PATH = 'hyperparameter_search_results.csv'

SR = 22050
CLIP_DURATION = 5.0
EPOCHS = 15
VALIDATION_SPLIT = 0.2
N_REFERENCE_CLIPS= 5
PATIENCE_LR = 3
PATIENCE_EARLY_STOP = 5

SEARCH_SPACE = {
    'batch_size': [32],
    'learning_rate': [1e-4],
    'emb_dim': [256],
    'temp': [0.07],
    'alpha': [0.25],
    'optimizer': ['Adam', 'AdamW'],
    'scheduler': ['ReduceLROnPlateau', 'CosineAnnealingLR'],
    'model_arch': ['Encoder3Layer', 'Encoder4Layer'],
    'augmentation': [
        'none',
        'specaugment',
        'audioment_light',
        'audioment_heavy',
    ]
}

spec_augment_transform = nn.Sequential(
    T.FrequencyMasking(freq_mask_param=20),
    T.TimeMasking(time_mask_param=40)
)
audioment_pipelines = {
    'audioment_light': Compose([
        TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
    ]),
    'audioment_heavy': Compose([
        TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
        Gain(min_gain_db=-6, max_gain_db=6, p=0.3),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.2),
        Mp3Compression(min_bitrate=32, max_bitrate=128, p=0.2, backend='pydub'),
    ])
}

def compute_log_mel(y, sr=SR):
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    logm = librosa.power_to_db(melspec)
    logm = (logm - logm.mean()) / (logm.std() + 1e-6)
    return torch.from_numpy(logm).unsqueeze(0)

class ContrastiveAudioDataset(Dataset):
    def __init__(self, csv_file, data_dir, clip_duration, sr, track_ids, id2idx, augment_config='none'):
        self.data_dir = data_dir
        self.clip_dur = clip_duration
        self.sr = sr
        self.id2idx = id2idx
        self.augment_config = augment_config
        with open(csv_file, newline='', encoding='utf-8') as f:
            all_tracks = [row['track_id'] for row in csv.DictReader(f)]
        self.samples = [tid for tid in all_tracks if tid in track_ids]

    def __len__(self): return len(self.samples)

    @staticmethod
    def _load_random_clip(path, duration, sr):
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
        y1 = self._load_random_clip(path, self.clip_dur, self.sr)
        y2 = self._load_random_clip(path, self.clip_dur, self.sr)

        if self.augment_config in audioment_pipelines:
            pipeline = audioment_pipelines[self.augment_config]
            y1 = pipeline(samples=y1, sample_rate=self.sr)
            y2 = pipeline(samples=y2, sample_rate=self.sr)

        x1, x2 = compute_log_mel(y1, sr=self.sr), compute_log_mel(y2, sr=self.sr)

        if self.augment_config == 'specaugment':
            x1, x2 = spec_augment_transform(x1), spec_augment_transform(x2)

        return x1, x2, label

class Encoder3Layer(nn.Module):
    def __init__(self, emb_dim=256, n_classes=1000):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.proj = nn.Sequential(nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, emb_dim))
        self.classifier = nn.Linear(emb_dim, n_classes)
    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        z_unnorm = self.proj(h)
        z = F.normalize(z_unnorm, dim=1)
        logits = self.classifier(z)
        return z, logits

class Encoder4Layer(nn.Module):
    def __init__(self, emb_dim=256, n_classes=1000):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.proj = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, emb_dim))
        self.classifier = nn.Linear(emb_dim, n_classes)
    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        z_unnorm = self.proj(h)
        z = F.normalize(z_unnorm, dim=1)
        logits = self.classifier(z)
        return z, logits

class EncoderDeeper(nn.Module): # did not use this one :)
    def __init__(self, emb_dim=256, n_classes=1000):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2,4)),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.proj = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, emb_dim))
        self.classifier = nn.Linear(emb_dim, n_classes)
    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        z_unnorm = self.proj(h)
        z = F.normalize(z_unnorm, dim=1)
        logits = self.classifier(z)
        return z, logits

def info_nce_loss(z1, z2, temp):
    B = z1.size(0)
    if B == 0: return torch.tensor(0.0, device=z1.device, requires_grad=True)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temp
    mask = torch.eye(2*B, device=sim.device).bool()
    sim = sim.masked_fill(mask, -1e9)
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).long().to(sim.device)
    return F.cross_entropy(sim, labels)

@torch.no_grad()
def evaluate(model, val_ds, config):
    device = next(model.parameters()).device
    model.eval()

    embeddings_db = {}
    for tid in tqdm(val_ds.samples, desc="[Eval] Building DB", leave=False):
        path = os.path.join(val_ds.data_dir, f"{tid}.mp3")
        clip_embs = []
        for _ in range(N_REFERENCE_CLIPS):
            y_clip = ContrastiveAudioDataset._load_random_clip(path, val_ds.clip_dur, val_ds.sr)
            x = compute_log_mel(y_clip, sr=val_ds.sr).unsqueeze(0).to(device)
            z, _ = model(x)
            clip_embs.append(z.squeeze(0))
        if clip_embs:
            embeddings_db[tid] = torch.stack(clip_embs).mean(dim=0).cpu()

    correct, total = 0, 0
    if not embeddings_db: return 0.0

    for query_tid in tqdm(val_ds.samples, desc="[Eval] Querying", leave=False):
        path = os.path.join(val_ds.data_dir, f"{query_tid}.mp3")
        y_query = ContrastiveAudioDataset._load_random_clip(path, val_ds.clip_dur, val_ds.sr)
        x_query = compute_log_mel(y_query, sr=val_ds.sr).unsqueeze(0).to(device)
        zq, _ = model(x_query)
        zq_cpu = zq.squeeze(0).cpu()

        sims = {tid: F.cosine_similarity(zq_cpu.unsqueeze(0), emb.unsqueeze(0)).item() for tid, emb in embeddings_db.items()}
        if not sims: continue
        pred_tid = max(sims, key=sims.get)
        if pred_tid == query_tid:
            correct += 1
        total += 1
    return (correct / total * 100) if total > 0 else 0.0

def run_experiment(exp_config, all_track_ids, id2idx):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_ids, val_ids = train_test_split(all_track_ids, test_size=VALIDATION_SPLIT, random_state=42)
    
    train_ds = ContrastiveAudioDataset(METADATA_CSV, DATA_DIR, CLIP_DURATION, SR, train_ids, id2idx, augment_config=exp_config['augmentation'])
    val_ds = ContrastiveAudioDataset(METADATA_CSV, DATA_DIR, CLIP_DURATION, SR, val_ids, id2idx, augment_config='none')
    
    train_loader = DataLoader(train_ds, batch_size=exp_config['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=exp_config['batch_size'], shuffle=False)

    model_class = globals()[exp_config['model_arch']]
    model = model_class(emb_dim=exp_config['emb_dim'], n_classes=len(id2idx)).to(device)
    optimizer = getattr(torch.optim, exp_config['optimizer'])(model.parameters(), lr=exp_config['learning_rate'])
    
    if exp_config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=PATIENCE_LR)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * EPOCHS)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x1, x2, labels in train_loader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            optimizer.zero_grad()
            z1, log1 = model(x1)
            z2, log2 = model(x2)
            loss_c = info_nce_loss(z1, z2, temp=exp_config['temp'])
            loss_ce = F.cross_entropy(log1, labels) + F.cross_entropy(log2, labels)
            loss = loss_c + exp_config['alpha'] * loss_ce
            loss.backward()
            optimizer.step()
            if isinstance(scheduler, CosineAnnealingLR): scheduler.step()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x1_v, x2_v, labels_v in val_loader:
                x1_v, x2_v, labels_v = x1_v.to(device), x2_v.to(device), labels_v.to(device)
                z1_v, log1_v = model(x1_v); z2_v, log2_v = model(x2_v)
                loss_c_v = info_nce_loss(z1_v, z2_v, temp=exp_config['temp'])
                loss_ce_v = F.cross_entropy(log1_v, labels_v) + F.cross_entropy(log2_v, labels_v)
                total_val_loss += (loss_c_v + exp_config['alpha'] * loss_ce_v).item()
        
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        if isinstance(scheduler, ReduceLROnPlateau): scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE_EARLY_STOP: break
    
    final_accuracy = evaluate(model, val_ds, exp_config)
    return final_accuracy

if __name__ == '__main__':
    if not os.path.isfile(METADATA_CSV):
        raise FileNotFoundError(f"Metadata CSV not found at {METADATA_CSV}")

    with open(METADATA_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_unique_track_ids = sorted(list(set(row['track_id'] for row in reader)))
    id2idx = {tid: i for i, tid in enumerate(all_unique_track_ids)}

    keys, values = zip(*SEARCH_SPACE.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    total_experiments = len(experiments)
    print(f"Generated a total of {total_experiments} experiments to run.")

    completed_configs = set()
    header = list(SEARCH_SPACE.keys()) + ['accuracy', 'duration_minutes', 'timestamp']
    if os.path.exists(RESULTS_CSV_PATH):
        print(f"Found existing results file at '{RESULTS_CSV_PATH}'. Reading experiments...")
        try:
            with open(RESULTS_CSV_PATH, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        config_from_row = {}
                        for key in SEARCH_SPACE.keys():
                            original_value_type = type(SEARCH_SPACE[key][0])
                            config_from_row[key] = original_value_type(row[key])
                        completed_configs.add(tuple(sorted(config_from_row.items())))
                    except (KeyError, ValueError) as e:
                        print(f"Warning: Could not parse a row in the CSV for resume logic: {row}. {e}")
            print(f"Found {len(completed_configs)} completed experiments. skipping.")
        except Exception as e:
            print(f"Error reading or parsing CSV file: {e}")
    else:
        with open(RESULTS_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        print(f"Created new results file at '{RESULTS_CSV_PATH}'.")

    experiments_to_run_count = total_experiments - len(completed_configs)
    print(f"Will run {experiments_to_run_count} new experiments.")

    current_experiment_num = 0
    for config in experiments:
        config_tuple = tuple(sorted(config.items()))
        if config_tuple in completed_configs:
            continue

        current_experiment_num += 1
        start_time = time.time()

        print("\n" + "="*80)
        print(f"Running Experiment {current_experiment_num}/{experiments_to_run_count} (Overall #{experiments.index(config)+1}/{total_experiments})")
        print(f"Config: {config}")
        print("="*80)

        try:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)

            accuracy = run_experiment(config, all_unique_track_ids, id2idx)

        except Exception as e:
            print(f"EXPERIMENT FAILED: {e}")
            traceback.print_exc()
            accuracy = 'FAIL'

        duration_minutes = (time.time() - start_time) / 60
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        log_row = list(config.values()) + [f"{accuracy:.4f}" if isinstance(accuracy, float) else accuracy, f"{duration_minutes:.2f}", timestamp]

        with open(RESULTS_CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(log_row)

        completed_configs.add(config_tuple)
        print(f"Experiment finished. Accuracy: {accuracy}.")

    print("\n\ncomplete")
    print(f"Results saved: '{RESULTS_CSV_PATH}'")


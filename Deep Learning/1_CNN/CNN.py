import os
import csv
import pickle
import random

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DATA_DIR = 'audio_1000'
METADATA_CSV = os.path.join(DATA_DIR, 'metadata.csv')
MODEL_PATH = 'contrastive_model_improved.pt'
LABELMAP_PATH = 'label_map.pkl'

SR = 22050
CLIP_DURATION = 5.0
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
TEMP = 0.1
ALPHA = 1.0
EMB_DIM = 128

def compute_log_mel(y, sr=SR):
    melspec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=64
    )
    logm = librosa.power_to_db(melspec)
    logm = (logm - logm.mean()) / (logm.std() + 1e-6)
    return torch.from_numpy(logm).unsqueeze(0)  # 1×64×T

class ContrastiveAudioDataset(Dataset):
    def __init__(self, csv_file, data_dir, clip_duration, sr):
        self.data_dir = data_dir
        self.clip_dur = clip_duration
        self.sr = sr
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            track_ids = [row['track_id'] for row in reader]
        self.ids = sorted(set(track_ids))
        self.id2idx = {tid: i for i, tid in enumerate(self.ids)}
        self.samples = track_ids

    def __len__(self):
        return len(self.samples)

    def _load_random_clip(self, track_id):
        path = os.path.join(self.data_dir, f"{track_id}.mp3")
        y, _ = librosa.load(path, sr=self.sr, mono=True)
        total = len(y) / self.sr
        if total > self.clip_dur:
            start = random.uniform(0, total - self.clip_dur)
            i0 = int(start * self.sr)
            i1 = int((start + self.clip_dur) * self.sr)
            y = y[i0:i1]
        else:
            target = int(self.clip_dur * self.sr)
            y = np.pad(y, (0, max(0, target - len(y))), mode='wrap')
        return y

    def __getitem__(self, idx):
        tid = self.samples[idx]
        label = self.id2idx[tid]
        y1 = self._load_random_clip(tid)
        y2 = self._load_random_clip(tid)
        x1 = compute_log_mel(y1, self.sr)
        x2 = compute_log_mel(y2, self.sr)
        return x1, x2, label

class Encoder(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, n_classes=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, emb_dim)
        )
        self.classifier = nn.Linear(emb_dim, n_classes) if n_classes>0 else None

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        z = F.normalize(self.proj(h), dim=1)
        logits = self.classifier(z) if self.classifier else None
        return z, logits

def info_nce_loss(z1, z2, temp=TEMP):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temp
    mask = torch.eye(2*B, device=sim.device).bool()
    sim = sim.masked_fill(mask, -9e15)
    labels = torch.cat([torch.arange(B,2*B), torch.arange(0,B)]).to(sim.device)
    return F.cross_entropy(sim, labels)

def train_contrastive():
    ds = ContrastiveAudioDataset(METADATA_CSV, DATA_DIR, CLIP_DURATION, SR)
    torch.save(ds.id2idx, LABELMAP_PATH)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Encoder(emb_dim=EMB_DIM, n_classes=len(ds.ids)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        for x1,x2,labels in tqdm(loader, desc=f"Epoch {epoch}"):
            x1,x2,labels = x1.to(device), x2.to(device), labels.to(device)
            z1,log1 = model(x1)
            z2,log2 = model(x2)
            loss_c = info_nce_loss(z1, z2)
            loss_ce = F.cross_entropy(log1, labels) + F.cross_entropy(log2, labels)
            loss = loss_c + ALPHA * loss_ce
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} avg loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    return model, ds.id2idx

def build_track_embeddings(model, id2idx):
    device = next(model.parameters()).device
    inv_map = {v:k for k,v in id2idx.items()}
    embeddings = {}
    model.eval()
    with torch.no_grad():
        for idx in range(len(inv_map)):
            tid = inv_map[idx]
            y,_ = librosa.load(os.path.join(DATA_DIR,f"{tid}.mp3"), sr=SR)
            clip = y[:int(CLIP_DURATION*SR)]
            x = compute_log_mel(clip).unsqueeze(0).to(device)
            z,_ = model(x)
            embeddings[tid] = z.cpu()
    return embeddings

def evaluate(model, embeddings):
    device = next(model.parameters()).device
    correct = 0
    N = len(embeddings)
    model.eval()
    with torch.no_grad():
        for tid,z_ref in embeddings.items():
            y,_ = librosa.load(os.path.join(DATA_DIR,f"{tid}.mp3"), sr=SR)
            start = random.uniform(0, max(0,len(y)/SR-CLIP_DURATION))
            i0,i1 = int(start*SR), int((start+CLIP_DURATION)*SR)
            clip = y[i0:i1]
            x = compute_log_mel(clip).unsqueeze(0).to(device)
            zq,_ = model(x)
            sims = {k: torch.cosine_similarity(zq,v).item() for k,v in embeddings.items()}
            pred = max(sims, key=sims.get)
            if pred==tid: correct+=1
    print(f"Top-1 accuracy (clean, 5s): {correct/N:.2%}")

if __name__=='__main__':
    model, id2idx = train_contrastive()
    embeddings = build_track_embeddings(model, id2idx)
    evaluate(model, embeddings)

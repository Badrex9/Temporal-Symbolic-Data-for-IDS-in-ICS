"""
train_cicids.py
===============
Entraînement CICIDS2017 — toutes les configs d_hist × pooling.

  d_hist  : 1, 5, 10, 20, 40
  pooling : avgpool, weightedpool, maskguided

Split stratifié 70/30 (random_state=843, fixe).
Validation : 10% du train (stratifié, random_state=843).
Early stopping sur F1 weighted val (patience=15 epochs, check tous les 3 epochs).
Sauvegarde : ./cicids_pooling/model_cicids_{pooling}_d{d_hist}.pth

Usage :
  python train_cicids.py
  python train_cicids.py --d_hist 10 --pooling maskguided   # config unique
"""

import os, csv, time, argparse
import numpy as np
from collections import defaultdict, deque
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ================================================================
# CONFIG
# ================================================================
PREPROCESSED_DIR = "./preprocessed_cicids/"
MODEL_DIR        = "./cicids_pooling/"
CSV_OUT          = "./summary_cicids_pooling.csv"

D_HIST_LIST  = [1, 5, 10, 20, 40]
POOLING_LIST = ["avgpool", "weightedpool", "maskguided"]

BATCH_SIZE   = 256
LR           = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
EPOCHS       = 100
ES_PATIENCE  = 15   # en epochs (check tous les 3 epochs → patience réelle = 45 epochs)
TAU          = 1.0
SPLIT_SEED   = 843

N_FEATURES   = 82
N_CLASSES    = 15

CLASS_NAMES = [
    "BENIGN", "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk",
    "DoS Slowhttptest", "DoS Slowloris", "FTP-Patator", "Heartbleed",
    "Infiltration", "PortScan", "SSH-Patator", "Web XSS",
    "Web BruteForce", "Web SQLi"
]

ARCH_CFG = {
    1:  dict(conv1_k=1, pool1_K=1,  conv2_k=1, pool2_K=1,  conv3_k=1),
    5:  dict(conv1_k=3, pool1_K=2,  conv2_k=2, pool2_K=1,  conv3_k=1),
    10: dict(conv1_k=3, pool1_K=4,  conv2_k=2, pool2_K=2,  conv3_k=2),
    20: dict(conv1_k=3, pool1_K=6,  conv2_k=2, pool2_K=2,  conv3_k=2),
    40: dict(conv1_k=3, pool1_K=10, conv2_k=2, pool2_K=4,  conv3_k=2),
}
N_FILTERS = 16
FC_HIDDEN  = [32, 128, 128, 64]


# ================================================================
# Pooling
# ================================================================
class TemporalAvgPool(nn.Module):
    def __init__(self, K):
        super().__init__(); self.pool = nn.AdaptiveAvgPool2d((None, K))
    def forward(self, x): return self.pool(x)

class TemporalWeightedPool(nn.Module):
    def __init__(self, K):
        super().__init__(); self.K = K
    def forward(self, x):
        B, C, H, W = x.shape
        if W == 1: return x
        raw = torch.tensor([1./(i+1) for i in range(W)], device=x.device, dtype=x.dtype)
        w = (raw / raw.sum()).view(1, 1, 1, W)
        return F.adaptive_avg_pool2d(x * w, (H, self.K))

class MaskGuidedVariablePooling(nn.Module):
    def __init__(self, K, tau=1.0):
        super().__init__(); self.K, self.tau = K, tau
        self.h_mlp = nn.Sequential(
            nn.LazyLinear(32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, x):
        B, C, H, W = x.shape; eps = 1e-8; dev = x.device
        h_k = self.h_mlp(x.mean(dim=[2, 3])).squeeze(1)
        t   = torch.linspace(0., 1., W, device=dev).unsqueeze(0)
        m   = torch.sigmoid(self.tau * (h_k.unsqueeze(1) - t))
        if self.K == 1:
            wt = m / (m.sum(1, keepdim=True) + eps)
            return (x * wt.unsqueeze(1).unsqueeze(2)).sum(3, keepdim=True)
        s   = torch.relu(m[:, :-1] - m[:, 1:])
        pi  = (s + eps) / (s.sum(1, keepdim=True) + eps)
        cdf = torch.cumsum(pi, 1).clamp(max=1.)
        q   = torch.linspace(0, 1, self.K + 1, device=dev)[1:-1]
        cuts = torch.stack([((cdf >= qi).int().argmax(1)) for qi in q], 1) \
               if self.K > 1 else torch.empty(B, 0, dtype=torch.long, device=dev)
        idx  = torch.arange(W, device=dev).unsqueeze(0)
        prev = torch.zeros(B, dtype=torch.long, device=dev)
        outs = []
        for r in range(self.K):
            curr = cuts[:, r] if r < self.K - 1 else \
                   torch.full((B,), W - 1, dtype=torch.long, device=dev)
            seg = ((idx >= prev.unsqueeze(1)) & (idx <= curr.unsqueeze(1))).float()
            wt  = seg * m; wt = wt / (wt.sum(1, keepdim=True) + eps)
            outs.append((x * wt.unsqueeze(1).unsqueeze(2)).sum(3))
            prev = curr + 1
        return torch.stack(outs, 3)

def make_pool(method, K, tau=TAU):
    if method == "avgpool":      return TemporalAvgPool(K)
    if method == "weightedpool": return TemporalWeightedPool(K)
    if method == "maskguided":   return MaskGuidedVariablePooling(K, tau)


# ================================================================
# Modèle
# ================================================================
class CICIDSModel(nn.Module):
    def __init__(self, d_hist, pooling):
        super().__init__()
        cfg = ARCH_CFG[d_hist]; nf = N_FILTERS
        self.conv1 = nn.Conv2d(1,  nf, (1, cfg["conv1_k"]))
        self.pool1 = make_pool(pooling, cfg["pool1_K"])
        self.conv2 = nn.Conv2d(nf, nf, (1, cfg["conv2_k"]))
        self.pool2 = make_pool(pooling, cfg["pool2_K"])
        self.conv3 = nn.Conv2d(nf, nf, (1, cfg["conv3_k"]))
        self.flat  = nn.Flatten()
        with torch.no_grad():
            d = torch.zeros(1, 1, N_FEATURES, d_hist)
            d = torch.relu(self.conv1(d)); d = self.pool1(d)
            d = torch.relu(self.conv2(d)); d = self.pool2(d)
            d = torch.relu(self.conv3(d))
            fsz = self.flat(d).shape[1]
        layers, in_d = [], fsz
        for out_d in FC_HIDDEN:
            layers += [nn.Linear(in_d, out_d), nn.ReLU()]; in_d = out_d
        layers.append(nn.Linear(in_d, N_CLASSES))
        self.fc   = nn.Sequential(*layers)
        self._npar = sum(p.numel() for p in self.parameters() if p.requires_grad)
    def forward(self, x):
        x = torch.relu(self.conv1(x)); x = self.pool1(x)
        x = torch.relu(self.conv2(x)); x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        return self.fc(self.flat(x))


# ================================================================
# Contexte flux
# ================================================================
def build_flow_context(sip, dip, dport, d_hist_max):
    N = len(sip)
    context = np.zeros((N, d_hist_max), dtype=np.int32)
    flow_buffers = defaultdict(lambda: deque(maxlen=d_hist_max))
    for i in tqdm(range(N), desc="  Contexte flux", leave=False):
        mn  = sip[i] if sip[i] < dip[i] else dip[i]
        mx  = dip[i] if sip[i] < dip[i] else sip[i]
        key = (mn, mx, int(dport[i]))
        buf = flow_buffers[key]; buf.append(i)
        indices = list(buf)[::-1]
        while len(indices) < d_hist_max: indices.append(indices[-1])
        context[i] = indices[:d_hist_max]
    return context


# ================================================================
# Dataset
# ================================================================
class CICIDSFlowDataset(Dataset):
    def __init__(self, X, Y, context, d_hist):
        self.X = X; self.Y = Y; self.context = context; self.d_hist = d_hist
    def __len__(self): return len(self.Y)
    def __getitem__(self, idx):
        mat = self.X[self.context[idx, :self.d_hist]].T
        return (torch.from_numpy(mat.copy()),
                torch.tensor(int(self.Y[idx]), dtype=torch.long))


# ================================================================
# Évaluation
# ================================================================
@torch.no_grad()
def evaluate_val(model, loader, device):
    model.eval()
    preds, labels = [], []
    for xb, yb in loader:
        xb = xb.unsqueeze(1).to(device, non_blocking=True)
        preds.append(model(xb).argmax(1).cpu().numpy())
        labels.append(yb.numpy())
    yt = np.concatenate(labels); yp = np.concatenate(preds)
    return f1_score(yt, yp, average="weighted", zero_division=0) * 100

@torch.no_grad()
def evaluate_full(model, loader, device):
    model.eval()
    preds, labels = [], []
    for xb, yb in loader:
        xb = xb.unsqueeze(1).to(device, non_blocking=True)
        preds.append(model(xb).argmax(1).cpu().numpy())
        labels.append(yb.numpy())
    yt = np.concatenate(labels); yp = np.concatenate(preds)
    f1_w   = f1_score(yt, yp, average="weighted", zero_division=0) * 100
    f1_mac = f1_score(yt, yp, average="macro",    zero_division=0) * 100
    acc    = accuracy_score(yt, yp) * 100
    per_class = {}
    for c in range(N_CLASSES):
        mask = yt == c; n = mask.sum()
        if n == 0:
            per_class[c] = dict(n=0, P=0., R=0., F1=0.); continue
        per_class[c] = dict(
            n   = int(n),
            P   = round(precision_score(yt==c, yp==c, zero_division=0)*100, 4),
            R   = round(recall_score   (yt==c, yp==c, zero_division=0)*100, 4),
            F1  = round(f1_score       (yt==c, yp==c, zero_division=0)*100, 4),
        )
    return dict(f1_weighted=round(f1_w,4), f1_macro=round(f1_mac,4),
                accuracy=round(acc,4), per_class=per_class), yt, yp


# ================================================================
# Entraînement d'une config
# ================================================================
def train_config(d_hist, pooling, X_tr, Y_tr, X_val, Y_val, X_te, Y_te,
                 sip_tr, dip_tr, dp_tr, sip_val, dip_val, dp_val,
                 sip_te, dip_te, dp_te, device, class_weights):

    d_hist_max = max(D_HIST_LIST)  # 40 — contexte precomputable une seule fois
    ctx_tr  = build_flow_context(sip_tr,  dip_tr,  dp_tr,  d_hist_max)
    ctx_val = build_flow_context(sip_val, dip_val, dp_val, d_hist_max)
    ctx_te  = build_flow_context(sip_te,  dip_te,  dp_te,  d_hist_max)

    tr_ds  = CICIDSFlowDataset(X_tr,  Y_tr,  ctx_tr,  d_hist)
    val_ds = CICIDSFlowDataset(X_val, Y_val, ctx_val, d_hist)
    te_ds  = CICIDSFlowDataset(X_te,  Y_te,  ctx_te,  d_hist)

    train_ld = DataLoader(tr_ds,  BATCH_SIZE, shuffle=True,  pin_memory=True, num_workers=2)
    val_ld   = DataLoader(val_ds, BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)
    te_ld    = DataLoader(te_ds,  BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)

    model = CICIDSModel(d_hist, pooling).to(device)
    cw    = torch.tensor(class_weights, dtype=torch.float32).to(device)
    crit  = nn.CrossEntropyLoss(weight=cw)
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    mpath = os.path.join(MODEL_DIR, f"model_cicids_{pooling}_d{d_hist}.pth")
    best_f1, best_ep, no_improve = -1., -1, 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time(); ls = 0.; tot = 0
        for xb, yb in train_ld:
            xb = xb.unsqueeze(1).to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            ls += loss.item() * yb.size(0); tot += yb.size(0)

        if ep % 3 == 0 or ep == EPOCHS:
            vf1 = evaluate_val(model, val_ld, device)
            marker = ""
            if vf1 > best_f1 + 0.05:
                best_f1, best_ep = vf1, ep
                torch.save(model.state_dict(), mpath)
                no_improve = 0; marker = " ◄ BEST"
            else:
                no_improve += 3
            print(f"  [ep {ep:3d}]  loss={ls/max(1,tot):.4f}  "
                  f"{time.time()-t0:.1f}s  valF1w={vf1:.2f}%{marker}")
            if no_improve >= ES_PATIENCE * 3:
                print(f"  ▶ Early stop ep={ep}"); break

    # Évaluation du meilleur modèle
    model.load_state_dict(torch.load(mpath, map_location="cpu"))
    model.to(device)
    m, yt, yp = evaluate_full(model, te_ld, device)

    print(f"\n  F1w={m['f1_weighted']:.4f}%  F1mac={m['f1_macro']:.4f}%  "
          f"Acc={m['accuracy']:.4f}%  best_ep={best_ep}")
    for c in range(N_CLASSES):
        pc   = m['per_class'][c]
        flag = " ◄ exclu" if c in [8, 9] else ""
        print(f"  {CLASS_NAMES[c]:25}  n={pc['n']:>6}  "
              f"P={pc['P']:>6.2f}%  R={pc['R']:>6.2f}%  F1={pc['F1']:>6.2f}%{flag}")

    return m, best_ep, model._npar


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_hist",  type=int, default=None,
                        help="Config unique (défaut : toutes)")
    parser.add_argument("--pooling", default=None,
                        choices=["avgpool", "weightedpool", "maskguided"],
                        help="Config unique (défaut : toutes)")
    parser.add_argument("--data_dir", default=PREPROCESSED_DIR)
    args = parser.parse_args()

    d_list = [args.d_hist]  if args.d_hist  else D_HIST_LIST
    p_list = [args.pooling] if args.pooling else POOLING_LIST

    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    print(f"  Configs : d={d_list}  pooling={p_list}\n")

    # Chargement
    X_all   = np.load(f"{args.data_dir}/X_all.npy")
    Y_all   = np.load(f"{args.data_dir}/Y_all.npy")
    sip_all = np.load(f"{args.data_dir}/sip_all.npy",   allow_pickle=True)
    dip_all = np.load(f"{args.data_dir}/dip_all.npy",   allow_pickle=True)
    dp_all  = np.load(f"{args.data_dir}/dport_all.npy")

    # Split stratifié fixe
    idx_all = np.arange(len(Y_all))
    idx_tr_full, idx_te = train_test_split(
        idx_all, test_size=0.3, stratify=Y_all, random_state=SPLIT_SEED)
    idx_tr, idx_val = train_test_split(
        idx_tr_full, test_size=0.1, stratify=Y_all[idx_tr_full],
        random_state=SPLIT_SEED)

    X_tr   = X_all[idx_tr];   Y_tr   = Y_all[idx_tr]
    X_val  = X_all[idx_val];  Y_val  = Y_all[idx_val]
    X_te   = X_all[idx_te];   Y_te   = Y_all[idx_te]
    sip_tr = sip_all[idx_tr]; dip_tr = dip_all[idx_tr]; dp_tr = dp_all[idx_tr]
    sip_val= sip_all[idx_val];dip_val= dip_all[idx_val];dp_val= dp_all[idx_val]
    sip_te = sip_all[idx_te]; dip_te = dip_all[idx_te]; dp_te = dp_all[idx_te]

    print(f"  Train : {len(Y_tr):,}  Val : {len(Y_val):,}  Test : {len(Y_te):,}\n")

    # Class weights sur le train
    class_counts  = np.bincount(Y_tr.astype(int), minlength=N_CLASSES)
    class_weights = 1.0 / np.maximum(np.sqrt(class_counts), 1)
    class_weights = class_weights / class_weights.sum() * N_CLASSES

    # CSV
    per_class_cols = []
    for c in range(N_CLASSES):
        name = CLASS_NAMES[c].replace(" ", "_")
        per_class_cols += [f"{name}_n", f"{name}_P", f"{name}_R", f"{name}_F1"]
    fieldnames = (["d_hist", "pooling", "best_epoch", "params",
                   "f1_weighted", "f1_macro", "accuracy"] + per_class_cols)
    write_hdr = not os.path.exists(CSV_OUT)
    fcsv = open(CSV_OUT, "a", newline="")
    w = csv.DictWriter(fcsv, fieldnames=fieldnames)
    if write_hdr: w.writeheader()

    # Boucle configs
    for d_hist in d_list:
        for pooling in p_list:
            print(f"\n{'═'*65}")
            print(f"  d_hist={d_hist}  pooling={pooling}")
            print(f"{'═'*65}")

            m, best_ep, npar = train_config(
                d_hist, pooling,
                X_tr, Y_tr, X_val, Y_val, X_te, Y_te,
                sip_tr, dip_tr, dp_tr,
                sip_val, dip_val, dp_val,
                sip_te, dip_te, dp_te,
                device, class_weights)

            row = dict(d_hist=d_hist, pooling=pooling, best_epoch=best_ep,
                       params=npar, f1_weighted=m['f1_weighted'],
                       f1_macro=m['f1_macro'], accuracy=m['accuracy'])
            for c in range(N_CLASSES):
                pc   = m['per_class'][c]
                name = CLASS_NAMES[c].replace(" ", "_")
                row[f"{name}_n"]  = pc['n']
                row[f"{name}_P"]  = pc['P']
                row[f"{name}_R"]  = pc['R']
                row[f"{name}_F1"] = pc['F1']
            w.writerow(row); fcsv.flush()

    fcsv.close()
    print(f"\n  ✔ Terminé → {CSV_OUT}")
    print(f"  ✔ Modèles → {MODEL_DIR}\n")


if __name__ == "__main__":
    main()

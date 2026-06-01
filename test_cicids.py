"""
test_cicids.py
==============
Évalue les modèles CICIDS2017 entraînés par train_cicids.py.
Affiche les métriques au format papier :
  - Regroupements DoS, Patator, WebAttack
  - Heartbleed et Infiltration affichés mais exclus du F1 global
  - F1 weighted global sur les 7 classes retenues

Usage :
  # Évalue tous les modèles du dossier cicids_pooling/
  python test_cicids.py

  # Config unique
  python test_cicids.py --d_hist 10 --pooling maskguided

  # Modèle spécifique
  python test_cicids.py --model ./cicids_pooling/model_cicids_maskguided_d10.pth
"""

import os, csv, argparse
import numpy as np
from collections import defaultdict, deque
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ================================================================
# CONFIG
# ================================================================
PREPROCESSED_DIR = "./preprocessed_cicids/"
MODEL_DIR        = "./cicids_pooling/"
CSV_OUT          = "./summary_cicids_test.csv"

D_HIST_LIST  = [1, 5, 10, 20, 40]
POOLING_LIST = ["avgpool", "weightedpool", "maskguided"]

BATCH_SIZE = 256
SPLIT_SEED = 843
TAU        = 1.0

N_FEATURES = 82
N_CLASSES  = 15

CLASS_NAMES = [
    "BENIGN", "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk",
    "DoS Slowhttptest", "DoS Slowloris", "FTP-Patator", "Heartbleed",
    "Infiltration", "PortScan", "SSH-Patator", "Web XSS",
    "Web BruteForce", "Web SQLi"
]

# Regroupements papier
GROUP_MAP = {
    0:  "BENIGN",     1:  "Bot",        2:  "DDoS",
    3:  "DoS",        4:  "DoS",        5:  "DoS",        6:  "DoS",
    7:  "Patator",    8:  "Heartbleed", 9:  "Infiltration",
    10: "PortScan",   11: "Patator",
    12: "WebAttack",  13: "WebAttack",  14: "WebAttack",
}
GROUP_ORDER = ["BENIGN", "Bot", "DDoS", "DoS", "Patator",
               "Heartbleed", "Infiltration", "PortScan", "WebAttack"]
EXCLUDED    = {"Heartbleed", "Infiltration"}

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
# Pooling + Modèle
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
        self.fc    = nn.Sequential(*layers)
        self._npar = sum(p.numel() for p in self.parameters() if p.requires_grad)
    def forward(self, x):
        x = torch.relu(self.conv1(x)); x = self.pool1(x)
        x = torch.relu(self.conv2(x)); x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        return self.fc(self.flat(x))


# ================================================================
# Contexte flux + Dataset
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

class CICIDSFlowDataset(Dataset):
    def __init__(self, X, Y, context, d_hist):
        self.X = X; self.Y = Y; self.context = context; self.d_hist = d_hist
    def __len__(self): return len(self.Y)
    def __getitem__(self, idx):
        mat = self.X[self.context[idx, :self.d_hist]].T
        return (torch.from_numpy(mat.copy()),
                torch.tensor(int(self.Y[idx]), dtype=torch.long))


# ================================================================
# Prédiction + métriques papier
# ================================================================
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    for xb, yb in loader:
        xb = xb.unsqueeze(1).to(device, non_blocking=True)
        preds.append(model(xb).argmax(1).cpu().numpy())
        labels.append(yb.numpy())
    return np.concatenate(labels), np.concatenate(preds)

def compute_paper_metrics(y_true, y_pred):
    y_true_g = np.array([GROUP_MAP[c] for c in y_true])
    y_pred_g = np.array([GROUP_MAP[c] for c in y_pred])
    results  = {}
    for grp in GROUP_ORDER:
        mask_t = y_true_g == grp; n = mask_t.sum()
        if n == 0:
            results[grp] = dict(n=0, P=0., R=0., F1=0.); continue
        mask_p = y_pred_g == grp
        results[grp] = dict(
            n  = int(n),
            P  = round(precision_score(mask_t, mask_p, zero_division=0)*100, 4),
            R  = round(recall_score   (mask_t, mask_p, zero_division=0)*100, 4),
            F1 = round(f1_score       (mask_t, mask_p, zero_division=0)*100, 4),
        )
    ret_mask = np.array([g not in EXCLUDED for g in y_true_g])
    f1_global = f1_score(y_true_g[ret_mask], y_pred_g[ret_mask],
                         average="weighted", zero_division=0) * 100
    acc = accuracy_score(y_true, y_pred) * 100
    return results, round(f1_global, 4), round(acc, 4)


# ================================================================
# Évaluation d'une config
# ================================================================
def eval_config(d_hist, pooling, mpath, X_te, Y_te,
                sip_te, dip_te, dp_te, device):
    d_hist_max = max(D_HIST_LIST)
    ctx_te = build_flow_context(sip_te, dip_te, dp_te, d_hist_max)
    te_ds  = CICIDSFlowDataset(X_te, Y_te, ctx_te, d_hist)
    te_ld  = DataLoader(te_ds, BATCH_SIZE, shuffle=False,
                        pin_memory=True, num_workers=2)

    model = CICIDSModel(d_hist, pooling).to(device)
    model.load_state_dict(torch.load(mpath, map_location="cpu"))
    y_true, y_pred = predict(model, te_ld, device)
    results, f1_global, acc = compute_paper_metrics(y_true, y_pred)

    print(f"\n  {'═'*62}")
    print(f"  d={d_hist}  pooling={pooling}  params={model._npar:,}")
    print(f"  {'─'*62}")
    print(f"  {'Groupe':15}  {'N':>8}  {'P%':>7}  {'R%':>7}  {'F1%':>7}  exclu")
    print(f"  {'─'*62}")
    for grp in GROUP_ORDER:
        r   = results[grp]
        exc = "✗" if grp in EXCLUDED else ""
        print(f"  {grp:15}  {r['n']:>8}  {r['P']:>6.2f}%  "
              f"{r['R']:>6.2f}%  {r['F1']:>6.2f}%  {exc}")
    print(f"  {'─'*62}")
    print(f"  F1 global (7 classes) : {f1_global:.4f}%")
    print(f"  Accuracy (toutes)     : {acc:.4f}%")
    print(f"  {'═'*62}")

    return results, f1_global, acc, model._npar


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default=None, help="Chemin .pth spécifique")
    parser.add_argument("--d_hist",   type=int, default=None)
    parser.add_argument("--pooling",  default=None,
                        choices=["avgpool", "weightedpool", "maskguided"])
    parser.add_argument("--data_dir", default=PREPROCESSED_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")

    # Données
    X_all   = np.load(f"{args.data_dir}/X_all.npy")
    Y_all   = np.load(f"{args.data_dir}/Y_all.npy")
    sip_all = np.load(f"{args.data_dir}/sip_all.npy",   allow_pickle=True)
    dip_all = np.load(f"{args.data_dir}/dip_all.npy",   allow_pickle=True)
    dp_all  = np.load(f"{args.data_dir}/dport_all.npy")

    idx_all = np.arange(len(Y_all))
    _, idx_te = train_test_split(
        idx_all, test_size=0.3, stratify=Y_all, random_state=SPLIT_SEED)

    X_te   = X_all[idx_te];   Y_te   = Y_all[idx_te]
    sip_te = sip_all[idx_te]; dip_te = dip_all[idx_te]; dp_te = dp_all[idx_te]
    print(f"  Test : {len(Y_te):,} samples\n")

    # Configs à évaluer
    if args.model:
        fname   = os.path.basename(args.model)
        d_hist  = args.d_hist  or 10
        pooling = args.pooling or "maskguided"
        configs = [(d_hist, pooling, args.model)]
    else:
        d_list  = [args.d_hist]  if args.d_hist  else D_HIST_LIST
        p_list  = [args.pooling] if args.pooling else POOLING_LIST
        configs = []
        for d in d_list:
            for p in p_list:
                mpath = os.path.join(MODEL_DIR, f"model_cicids_{p}_d{d}.pth")
                if os.path.exists(mpath):
                    configs.append((d, p, mpath))
                else:
                    print(f"  ⚠ Modèle manquant : {mpath}")

    # CSV
    group_cols = []
    for grp in GROUP_ORDER:
        group_cols += [f"{grp}_n", f"{grp}_P", f"{grp}_R", f"{grp}_F1"]
    fieldnames = ["d_hist", "pooling", "params", "f1_global", "accuracy"] + group_cols
    write_hdr  = not os.path.exists(CSV_OUT)
    fcsv = open(CSV_OUT, "a", newline="")
    w    = csv.DictWriter(fcsv, fieldnames=fieldnames)
    if write_hdr: w.writeheader()

    for d_hist, pooling, mpath in configs:
        results, f1_global, acc, npar = eval_config(
            d_hist, pooling, mpath,
            X_te, Y_te, sip_te, dip_te, dp_te, device)

        row = dict(d_hist=d_hist, pooling=pooling, params=npar,
                   f1_global=f1_global, accuracy=acc)
        for grp in GROUP_ORDER:
            r = results.get(grp, dict(n=0, P=0., R=0., F1=0.))
            row[f"{grp}_n"]  = r['n']
            row[f"{grp}_P"]  = r['P']
            row[f"{grp}_R"]  = r['R']
            row[f"{grp}_F1"] = r['F1']
        w.writerow(row); fcsv.flush()

    fcsv.close()
    print(f"\n  ✔ Résultats → {CSV_OUT}\n")


if __name__ == "__main__":
    main()

"""
test_ot_unsup.py
================
Évalue les modèles OT non supervisés entraînés par train_ot_unsup.py.
Calcule : P, R, F1 (point-wise) + F1_PA (point-adjust) + AUC-ROC

Usage :
  # Évalue toutes les configs du dossier
  python test_ot_unsup.py --dataset swat

  # Config unique
  python test_ot_unsup.py --dataset swat --d_hist 10 --pooling maskguided

  # Modèle spécifique
  python test_ot_unsup.py --dataset swat \
      --model ./swat_unsup_models/unsup_swat_maskguided_d10.pth
"""

import os, csv, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# ================================================================
# CONFIG
# ================================================================
DATASET = "swat"   # défaut si pas d'argument

DATA_DIRS  = {"swat": "./preprocessed_swat_unsup/",
              "wadi": "./preprocessed_wadi_unsup/"}
MODEL_DIRS = {"swat": "./swat_unsup_models/",
              "wadi": "./wadi_unsup_models/"}
CSV_OUT    = {"swat": "./summary_swat_test.csv",
              "wadi": "./summary_wadi_test.csv"}

D_HIST_LIST  = [1, 5, 10, 20, 40]
POOLING_LIST = ["avgpool", "weightedpool", "maskguided"]

BATCH_SIZE    = 512
THRESHOLD_PCT = 99
TAU           = 1.0

ARCH_CFG = {
    1:  dict(conv1_k=1, pool1_K=1, conv2_k=1, pool2_K=1, conv3_k=1),
    5:  dict(conv1_k=2, pool1_K=4, conv2_k=2, pool2_K=3, conv3_k=3),
    10: dict(conv1_k=3, pool1_K=6, conv2_k=2, pool2_K=3, conv3_k=3),
    20: dict(conv1_k=3, pool1_K=6, conv2_k=2, pool2_K=3, conv3_k=3),
    40: dict(conv1_k=5, pool1_K=6, conv2_k=2, pool2_K=3, conv3_k=3),
}
N_FILTERS = {1: 16, 5: 16, 10: 16, 20: 16, 40: 16}
FC_HIDDEN  = {"swat": [64, 128, 128, 64], "wadi": [64, 128, 128, 64]}


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
        w   = (raw / raw.sum()).view(1, 1, 1, W)
        return F.adaptive_avg_pool2d(x * w, (H, self.K))

class MaskGuidedVariablePooling(nn.Module):
    def __init__(self, K, tau=1.0, seg_tau=10.0):
        super().__init__()
        self.K, self.tau, self.seg_tau = K, tau, seg_tau
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
        cdf = torch.cat([torch.zeros(B, 1, device=dev), cdf], dim=1)
        bounds = torch.linspace(0., 1., self.K + 1, device=dev)
        outs = []
        for r in range(self.K):
            lo, hi = bounds[r], bounds[r + 1]
            sw = (torch.sigmoid(self.seg_tau * (cdf - lo))
                  * torch.sigmoid(self.seg_tau * (hi - cdf)))
            wt = sw * m; wt = wt / (wt.sum(1, keepdim=True) + eps)
            outs.append((x * wt.unsqueeze(1).unsqueeze(2)).sum(3))
        return torch.stack(outs, 3)

def make_pool(method, K, tau=TAU):
    if method == "avgpool":      return TemporalAvgPool(K)
    if method == "weightedpool": return TemporalWeightedPool(K)
    if method == "maskguided":   return MaskGuidedVariablePooling(K, tau)

class OTUnsupModel(nn.Module):
    def __init__(self, n_features, d_hist, dataset, pooling):
        super().__init__()
        cfg = ARCH_CFG[d_hist]; nf = N_FILTERS[d_hist]; fc = FC_HIDDEN[dataset]
        self.conv1 = nn.Conv2d(1,  nf, (1, cfg["conv1_k"]))
        self.pool1 = make_pool(pooling, cfg["pool1_K"])
        self.conv2 = nn.Conv2d(nf, nf, (1, cfg["conv2_k"]))
        self.pool2 = make_pool(pooling, cfg["pool2_K"])
        self.conv3 = nn.Conv2d(nf, nf, (1, cfg["conv3_k"]))
        self.flat  = nn.Flatten()
        with torch.no_grad():
            d = torch.zeros(1, 1, n_features, d_hist)
            d = torch.relu(self.conv1(d)); d = self.pool1(d)
            d = torch.relu(self.conv2(d)); d = self.pool2(d)
            d = torch.relu(self.conv3(d))
            fsz = self.flat(d).shape[1]
        layers, in_d = [], fsz
        for out_d in fc:
            layers += [nn.Linear(in_d, out_d), nn.ReLU()]; in_d = out_d
        layers.append(nn.Linear(in_d, n_features))
        self.fc    = nn.Sequential(*layers)
        self._npar = sum(p.numel() for p in self.parameters() if p.requires_grad)
    def forward(self, x):
        x = torch.relu(self.conv1(x)); x = self.pool1(x)
        x = torch.relu(self.conv2(x)); x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        return self.fc(self.flat(x))


# ================================================================
# Dataset + scores
# ================================================================
class OTPredDataset(Dataset):
    def __init__(self, X, d_hist):
        self.X = X.astype(np.float32, copy=False)
        self.W = int(d_hist)
        self.indices = np.arange(1, len(X))
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        i   = self.indices[idx]
        seq = [self.X[max(0, i-j)] for j in range(1, self.W+1)]
        return torch.from_numpy(np.stack(seq, axis=1)), torch.from_numpy(self.X[i])

def point_adjust(y_true, y_pred):
    y_adj = y_pred.copy()
    in_seg, start = False, 0
    for i, v in enumerate(y_true):
        if v == 1 and not in_seg:
            in_seg, start = True, i
        elif v == 0 and in_seg:
            if y_pred[start:i].any(): y_adj[start:i] = 1
            in_seg = False
    if in_seg and y_pred[start:].any(): y_adj[start:] = 1
    return y_adj

@torch.no_grad()
def compute_scores(model, X, d_hist, device):
    model.eval()
    ds = OTPredDataset(X, d_hist)
    ld = DataLoader(ds, BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)
    scores = []
    for xb, tgt in ld:
        xb  = xb.unsqueeze(1).to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        err = (model(xb) - tgt) ** 2
        k   = max(1, err.shape[1] // 10)
        scores.append(err.topk(k, dim=1).values.mean(dim=1).cpu().numpy())
    return np.concatenate(scores)

def compute_metrics(scores_te, y_te, thr):
    yt    = y_te[1:]
    yp    = (scores_te >= thr).astype(int)
    yp_pa = point_adjust(yt, yp)
    auc   = roc_auc_score(yt, scores_te) * 100 if len(np.unique(yt)) > 1 else 0.
    return dict(
        precision = round(float(precision_score(yt, yp,    zero_division=0)) * 100, 4),
        recall    = round(float(recall_score   (yt, yp,    zero_division=0)) * 100, 4),
        f1        = round(float(f1_score       (yt, yp,    zero_division=0)) * 100, 4),
        f1_pa     = round(float(f1_score       (yt, yp_pa, zero_division=0)) * 100, 4),
        roc_auc   = round(auc, 4),
    )


# ================================================================
# Évaluation d'une config
# ================================================================
def eval_config(mpath, d_hist, pooling, dataset, X_tr, X_te, y_te, device):
    model = OTUnsupModel(X_tr.shape[1], d_hist, dataset, pooling).to(device)
    model.load_state_dict(torch.load(mpath, map_location="cpu"))

    scores_tr = compute_scores(model, X_tr, d_hist, device)
    scores_te = compute_scores(model, X_te, d_hist, device)
    thr = np.percentile(scores_tr, THRESHOLD_PCT)
    m   = compute_metrics(scores_te, y_te, thr)

    print(f"\n  {'═'*55}")
    print(f"  d={d_hist}  pooling={pooling}  params={model._npar:,}")
    print(f"  {'─'*55}")
    print(f"  {'Métrique':20}  {'Point-wise':>10}  {'Point-Adjust':>12}")
    print(f"  {'─'*55}")
    for k, k_pa in [("precision","—"), ("recall","—"), ("f1","f1_pa")]:
        pw  = f"{m[k]:.2f}%"
        pa  = f"{m[k_pa]:.2f}%" if k_pa in m else "—"
        print(f"  {k:20}  {pw:>10}  {pa:>12}")
    print(f"  {'roc_auc':20}  {m['roc_auc']:>9.2f}%  {'—':>12}")
    print(f"  {'═'*55}")

    return m, model._npar


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default=DATASET, choices=["swat", "wadi"])
    parser.add_argument("--model",    default=None)
    parser.add_argument("--d_hist",   type=int, default=None)
    parser.add_argument("--pooling",  default=None,
                        choices=["avgpool", "weightedpool", "maskguided"])
    parser.add_argument("--data_dir", default=None)
    args = parser.parse_args()

    dataset  = args.dataset
    data_dir = args.data_dir or DATA_DIRS[dataset]
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}  Dataset : {dataset}")

    X_tr = np.load(os.path.join(data_dir, "X_train.npy"))
    X_te = np.load(os.path.join(data_dir, "X_test.npy"))
    y_te = np.load(os.path.join(data_dir, "y_test.npy"))
    print(f"  Train : {X_tr.shape}  Test : {X_te.shape}  "
          f"atk={y_te.sum()} ({y_te.mean()*100:.1f}%)\n")

    # Configs à évaluer
    if args.model:
        d_hist  = args.d_hist  or 10
        pooling = args.pooling or "maskguided"
        configs = [(d_hist, pooling, args.model)]
    else:
        d_list  = [args.d_hist]  if args.d_hist  else D_HIST_LIST
        p_list  = [args.pooling] if args.pooling else POOLING_LIST
        configs = []
        for d in d_list:
            p_iter = ["avgpool"] if d == 1 else p_list
            for p in p_iter:
                mpath = os.path.join(MODEL_DIRS[dataset],
                                     f"unsup_{dataset}_{p}_d{d}.pth")
                if os.path.exists(mpath):
                    configs.append((d, p, mpath))
                else:
                    print(f"  ⚠ Manquant : {mpath}")

    # CSV
    fieldnames = ["dataset", "pooling", "d_hist", "params",
                  "precision", "recall", "f1", "f1_pa", "roc_auc"]
    csv_path  = CSV_OUT[dataset]
    write_hdr = not os.path.exists(csv_path)
    fcsv = open(csv_path, "a", newline="")
    w    = csv.DictWriter(fcsv, fieldnames=fieldnames)
    if write_hdr: w.writeheader()

    for d_hist, pooling, mpath in configs:
        m, npar = eval_config(mpath, d_hist, pooling, dataset,
                              X_tr, X_te, y_te, device)
        w.writerow(dict(dataset=dataset, pooling=pooling, d_hist=d_hist,
                        params=npar, **m))
        fcsv.flush()

    fcsv.close()
    print(f"\n  ✔ Résultats → {csv_path}\n")


if __name__ == "__main__":
    main()

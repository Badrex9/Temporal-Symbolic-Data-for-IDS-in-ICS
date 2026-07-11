"""
reeval_duogat_pct.py
=====================
Recharge un modele DuoGAT DEJA ENTRAINE (checkpoint model.pt) et refait
uniquement l'evaluation avec seuil = 99e percentile du train (StandardScaler
fit une seule fois sur le train, applique au train ET au test -- corrige le
bug de rescaling independant de get_score()).

Pas de reentrainement -- juste un forward pass + calcul de metriques.

Usage :
  python reeval_duogat_pct.py --dataset WADI --model_path output/WADI/<run_id> \
      --lookback 50 --our_data_dir ./preprocessed_wadi_unsup/
  python reeval_duogat_pct.py --dataset SWAT --model_path output/SWAT/<run_id> \
      --lookback 5 --our_data_dir ./preprocessed_swat_unsup/
"""

import os, json, argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from utils import SlidingWindowDataset, get_target_dims, load
from duogat import DuoGAT
from prediction import Predictor


def load_our_preprocessed(dataset, dif_n, our_data_dir):
    x_train = np.load(os.path.join(our_data_dir, "X_train.npy")).astype(np.float32)
    x_test  = np.load(os.path.join(our_data_dir, "X_test.npy")).astype(np.float32)
    y_test  = np.load(os.path.join(our_data_dir, "y_test.npy")).astype(np.uint8)

    def make_diff(X, dif_n):
        X_diff = np.zeros_like(X, dtype=np.float32)
        if dif_n > 0:
            X_diff[dif_n:] = X[dif_n:] - X[:-dif_n]
        return X_diff

    return x_train, x_test, y_test, make_diff(x_train, dif_n), make_diff(x_test, dif_n)


def point_adjust(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.uint8)
    y_pred = np.asarray(y_pred).astype(np.uint8).copy()
    n = len(y_true)
    i = 0
    while i < n:
        if y_true[i] == 1:
            start = i
            while i < n and y_true[i] == 1:
                i += 1
            end = i
            if y_pred[start:end].sum() > 0:
                y_pred[start:end] = 1
        else:
            i += 1
    return y_pred


def raw_scores(model, window_size, target_dims, batch_size, values, dif_values, device):
    """MSE moyenne sur toutes les features -- meme formule que notre
    propre modele et GTA, sans StandardScaler."""
    data = SlidingWindowDataset(values, window_size, target_dims)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    dif_data = SlidingWindowDataset(dif_values, window_size, target_dims)
    dif_loader = torch.utils.data.DataLoader(dif_data, batch_size=batch_size, shuffle=False)

    model.eval()
    preds = []
    with torch.no_grad():
        for (x, y), (dif_x, dif_y) in zip(loader, dif_loader):
            x = x.to(device)
            dif_x = dif_x.to(device)
            y_hat = model(x, dif_x)
            preds.append(y_hat.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    actual = values.detach().cpu().numpy()[window_size:]
    if target_dims is not None:
        actual = actual[:, target_dims]
    errors = (preds - actual) ** 2
    return errors.mean(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["SWAT", "WADI"])
    parser.add_argument("--model_path", required=True,
                        help="Dossier contenant model.pt (ex: output/WADI/07072026_183000)")
    parser.add_argument("--lookback", type=int, required=True)
    parser.add_argument("--our_data_dir", required=True)
    parser.add_argument("--dif_n", type=int, default=1)
    parser.add_argument("--gru_n_layers", type=int, default=1)
    parser.add_argument("--gru_hid_dim", type=int, default=150)
    parser.add_argument("--fc_n_layers", type=int, default=1)
    parser.add_argument("--fc_hid_dim", type=int, default=150)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--bs", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"  Re-evaluation DuoGAT (checkpoint existant, pas de reentrainement)")
    print(f"  dataset={args.dataset}  model_path={args.model_path}  device={device}")
    print(f"{'='*70}\n")

    x_train, x_test, y_test, dif_x_train, dif_x_test = load_our_preprocessed(
        args.dataset, args.dif_n, args.our_data_dir)
    print(f"  x_train : {x_train.shape}   x_test : {x_test.shape}")

    x_train = torch.from_numpy(x_train).float()
    x_test  = torch.from_numpy(x_test).float()
    dif_x_train = torch.from_numpy(dif_x_train).float()
    dif_x_test  = torch.from_numpy(dif_x_test).float()

    n_features = x_train.shape[1]
    target_dims = get_target_dims(args.dataset)
    out_dim = n_features if target_dims is None else (
        1 if isinstance(target_dims, int) else len(target_dims))

    model = DuoGAT(
        n_features, args.lookback, out_dim, batch_size=args.bs,
        gru_n_layers=args.gru_n_layers, gru_hid_dim=args.gru_hid_dim,
        forecast_n_layers=args.fc_n_layers, forecast_hid_dim=args.fc_hid_dim,
        dropout=args.dropout, alpha=args.alpha)

    ckpt_path = os.path.join(args.model_path, "model.pt")
    load(model, ckpt_path, device=device)
    model.to(device)
    print(f"  Checkpoint charge : {ckpt_path}")

    print("\n-- Calcul des scores sur le TRAIN --")
    train_scores = raw_scores(model, args.lookback, target_dims, args.bs,
                              x_train, dif_x_train, device)
    print("-- Calcul des scores sur le TEST --")
    test_scores = raw_scores(model, args.lookback, target_dims, args.bs,
                             x_test, dif_x_test, device)

    threshold = float(np.percentile(train_scores, 99))
    print(f"\n  Seuil (pct=99, train, score harmonise MSE-moyenne) : {threshold:.6f}")

    label = y_test[args.lookback:]
    if len(label) != len(test_scores):
        raise ValueError(f"Longueurs incoherentes : label={len(label)}, scores={len(test_scores)}")

    auc = roc_auc_score(label, test_scores) * 100
    y_pred = (test_scores > threshold).astype(np.uint8)
    y_pa   = point_adjust(label, y_pred)

    p_pw  = precision_score(label, y_pred, zero_division=0) * 100
    r_pw  = recall_score(label, y_pred, zero_division=0) * 100
    f1_pw = f1_score(label, y_pred, zero_division=0) * 100
    p_pa  = precision_score(label, y_pa, zero_division=0) * 100
    r_pa  = recall_score(label, y_pa, zero_division=0) * 100
    f1_pa = f1_score(label, y_pa, zero_division=0) * 100

    print(f"\n{'='*70}")
    print(f"  RESULTATS -- {args.dataset}  (seuil = 99e percentile train, score MSE-moyenne harmonise)")
    print(f"{'='*70}")
    print(f"  AUC-ROC                  : {auc:.4f}%")
    print(f"  Precision (point-wise)   : {p_pw:.2f}%")
    print(f"  Recall    (point-wise)   : {r_pw:.2f}%")
    print(f"  F1        (point-wise)   : {f1_pw:.2f}%")
    print(f"  Precision (point-adjust) : {p_pa:.2f}%")
    print(f"  Recall    (point-adjust) : {r_pa:.2f}%")
    print(f"  F1        (point-adjust) : {f1_pa:.2f}%")
    print(f"{'='*70}")
    print("  TSV :")
    print("dataset\tAUC\tF1_pw\tP_pw\tR_pw\tF1_PA\tP_PA\tR_PA")
    print(f"{args.dataset}\t{auc:.4f}\t{f1_pw:.4f}\t{p_pw:.4f}\t{r_pw:.4f}\t"
          f"{f1_pa:.4f}\t{p_pa:.4f}\t{r_pa:.4f}")
    print(f"{'='*70}\n")

    metrics = dict(auc=auc, threshold=threshold,
                   point_wise=dict(precision=p_pw, recall=r_pw, f1=f1_pw),
                   point_adjust=dict(precision=p_pa, recall=r_pa, f1=f1_pa))
    with open(os.path.join(args.model_path, "metrics_pct99_v2.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()

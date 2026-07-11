"""
compute_auroc_gta_v2.py
========================
Version corrigee de compute_auroc_gta.py : le seuil d'anomalie est
desormais calcule sur le 99e percentile des scores du TRAIN (donnees
normales uniquement), et non sur le test lui-meme -- coherent avec
la methodologie utilisee pour notre propre modele et pour DuoGAT
(seuil = 99e percentile des scores sur les donnees d'entrainement
normales, cf. train_ot_unsup.py / test_ot_unsup.py).

Recharge le checkpoint entraine par run_gta_train_eval.py, refait un
forward pass sur le train pour obtenir les scores normaux, calcule le
seuil, puis evalue sur les pred/true/label sauvegardes du test.

Usage :
  python compute_auroc_gta_v2.py --dataset swat --data_dir ./gta_data/ \
      --setting gta_SWaT_sl60_ll30_pl24
  python compute_auroc_gta_v2.py --dataset wadi --data_dir ./gta_data/ \
      --setting gta_WADI_sl60_ll30_pl24
"""

import sys, os, argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GTA"))
from exp.exp_gta_dad import Exp_GTA_DAD  # noqa: E402


class Args:
    pass


DATASET_CFG = {
    "swat": dict(data="SWaT", target="FIT_101",
                normal_file="SWaT_normaldata_downsampled.csv"),
    "wadi": dict(data="WADI", target="1_LS_001_AL",
                normal_file="WADI_14days_downsampled.csv"),
}


def detect_num_nodes(data_dir, cfg):
    import pandas as pd
    path = os.path.join(data_dir, cfg["normal_file"])
    df = pd.read_csv(path, nrows=1)
    return len(df.columns) - 1


def build_args(dataset, data_dir, seq_len, label_len, pred_len, batch_size):
    cfg = DATASET_CFG[dataset]
    num_nodes = detect_num_nodes(data_dir, cfg)
    args = Args()
    args.model = "gta"
    args.data = cfg["data"]
    args.root_path = data_dir
    args.data_path = ""
    args.features = "M"
    args.target = cfg["target"]

    args.seq_len = seq_len
    args.label_len = label_len
    args.pred_len = pred_len
    args.num_nodes = num_nodes
    args.num_levels = 3
    args.d_model = 128
    args.n_heads = 8
    args.e_layers = 3
    args.d_layers = 2
    args.d_ff = 128
    args.factor = 5

    args.dropout = 0.05
    args.attn = "prob"
    args.embed = "fixed"
    args.activation = "gelu"
    args.num_workers = 0

    args.batch_size = batch_size
    args.learning_rate = 1e-4
    args.lradj = "type1"

    args.use_gpu = torch.cuda.is_available()
    args.gpu = 0
    return args


def point_adjust(y_true, y_pred):
    y_adj = y_pred.copy()
    in_seg, start = False, 0
    for i, v in enumerate(y_true):
        if v == 1 and not in_seg:
            in_seg, start = True, i
        elif v == 0 and in_seg:
            if y_pred[start:i].any():
                y_adj[start:i] = 1
            in_seg = False
    if in_seg and y_pred[start:].any():
        y_adj[start:] = 1
    return y_adj


def compute_train_scores(exp, args):
    """Forward pass sur le train (normal), retourne un vecteur de scores
    (MSE moyenne par pas de temps sur les features), aplati."""
    train_data, train_loader = exp._get_data(flag="train")
    exp.model.eval()
    scores = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            batch_x = batch_x.double().to(exp.device)
            batch_y = batch_y.double().to(exp.device)
            batch_x_mark = batch_x_mark.double().to(exp.device)
            batch_y_mark = batch_y_mark.double().to(exp.device)

            outputs = exp.model(batch_x, batch_y, batch_x_mark, batch_y_mark)
            true = batch_y[:, -args.pred_len:, :]

            err = ((outputs - true) ** 2).mean(dim=-1)  # (B, pred_len)
            scores.append(err.detach().cpu().numpy().reshape(-1))
    return np.concatenate(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["swat", "wadi"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--setting", required=True,
                        help="Nom du dossier checkpoint/results (voir "
                             "sortie de run_gta_train_eval.py)")
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--label_len", type=int, default=30)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pct", type=int, default=99)
    args_cli = parser.parse_args()

    args = build_args(args_cli.dataset, args_cli.data_dir,
                      args_cli.seq_len, args_cli.label_len,
                      args_cli.pred_len, args_cli.batch_size)
    if args.use_gpu:
        torch.cuda.set_device(args.gpu)

    print(f"\n{'='*60}")
    print(f"  GTA -- AUROC / F1 (seuil calcule sur le TRAIN, coherent")
    print(f"  avec la methodologie de notre modele et DuoGAT)")
    print(f"  setting = {args_cli.setting}")
    print(f"{'='*60}\n")

    # Recharger le modele entraine
    exp = Exp_GTA_DAD(args)
    ckpt_path = f"./checkpoints/{args_cli.setting}/checkpoint.pth"
    exp.model.load_state_dict(torch.load(ckpt_path, map_location=exp.device))
    print(f"  Checkpoint charge : {ckpt_path}")

    # Scores sur le train (normal) -> seuil
    print("  Calcul des scores sur le train (normal)...")
    train_scores = compute_train_scores(exp, args)
    thr = np.percentile(train_scores, args_cli.pct)
    print(f"  Seuil (pct={args_cli.pct}, sur train normal) : {thr:.6f}")

    # Scores sur le test (deja sauvegardes par exp.test())
    folder = f"./results/{args_cli.setting}/"
    preds  = np.load(folder + "pred.npy")
    trues  = np.load(folder + "true.npy")
    labels = np.load(folder + "label.npy")

    err = ((preds - trues) ** 2).mean(axis=-1)
    scores_flat = err.reshape(-1)
    labels_flat = labels.reshape(-1)

    n_anom = labels_flat.sum()
    print(f"\n  Total points test : {len(labels_flat):,}  "
          f"({n_anom:,} anomalies, {100*n_anom/len(labels_flat):.2f}%)")

    if len(np.unique(labels_flat)) < 2:
        print("\n  ATTENTION : un seul type de label present, AUROC non calculable.")
        return

    auc = roc_auc_score(labels_flat, scores_flat) * 100
    print(f"\n  AUC-ROC : {auc:.4f}%  (independant du seuil)")

    y_pred = (scores_flat >= thr).astype(int)
    y_pa   = point_adjust(labels_flat, y_pred)

    p_pw  = precision_score(labels_flat, y_pred, zero_division=0) * 100
    r_pw  = recall_score(labels_flat, y_pred, zero_division=0) * 100
    f1_pw = f1_score(labels_flat, y_pred, zero_division=0) * 100
    f1_pa = f1_score(labels_flat, y_pa, zero_division=0) * 100

    print(f"  Precision (point-wise)   : {p_pw:.2f}%")
    print(f"  Recall    (point-wise)   : {r_pw:.2f}%")
    print(f"  F1        (point-wise)   : {f1_pw:.2f}%")
    print(f"  F1        (point-adjust) : {f1_pa:.2f}%")

    p_pa  = precision_score(labels_flat, y_pa, zero_division=0) * 100
    r_pa  = recall_score(labels_flat, y_pa, zero_division=0) * 100
    print(f"  Precision (point-adjust) : {p_pa:.2f}%")
    print(f"  Recall    (point-adjust) : {r_pa:.2f}%")

    print(f"\n{'='*60}")
    print("  TSV :")
    print("setting\tAUC\tF1_pw\tF1_PA\tP_pw\tR_pw\tP_PA\tR_PA")
    print(f"{args_cli.setting}\t{auc:.4f}\t{f1_pw:.4f}\t{f1_pa:.4f}\t{p_pw:.4f}\t{r_pw:.4f}\t{p_pa:.4f}\t{r_pa:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

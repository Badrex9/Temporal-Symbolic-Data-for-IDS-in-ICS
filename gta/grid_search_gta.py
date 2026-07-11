"""
grid_search_gta.py
===================
Grid search exhaustif sur le seuil d'anomalie, fidele a la Section V.B.4
du papier GTA : "we apply a grid search on all possible anomaly
thresholds to search for the best F1-score (**) and Recall (*)".

Contrairement a compute_auroc_gta_v2.py (seuil fixe = 99e percentile du
train), ce script cherche le seuil qui MAXIMISE le F1 (point-adjust,
protocole officiel du papier -- "we adopt the point-adjust way to
calculate the performance metrics") directement sur les scores du test,
comme fait le papier original.

Optimisation : point_adjust boucle sur les SEGMENTS d'anomalies (quelques
dizaines) et non sur tous les points (des centaines de milliers), ce qui
rend le grid search tractable meme avec des milliers de seuils candidats.

Usage :
  python grid_search_gta.py --setting gta_SWaT_sl60_ll30_pl24
  python grid_search_gta.py --setting gta_WADI_sl60_ll30_pl24 --n_thresholds 5000
"""

import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


def point_adjust_vec(y_true, y_pred):
    """Point-adjust vectorise : boucle sur les segments d'anomalies
    (contigus dans y_true), pas sur tous les points."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    padded = np.concatenate(([0], y_true, [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]  # exclusif

    y_adj = y_pred.copy()
    for s, e in zip(starts, ends):
        if y_pred[s:e].any():
            y_adj[s:e] = 1
    return y_adj


def grid_search(scores, labels, n_thresholds=2000):
    """Cherche le seuil maximisant F1 (point-wise) et F1 (point-adjust)
    parmi n_thresholds candidats repartis sur la plage des scores."""
    lo, hi = np.percentile(scores, 0.0), np.percentile(scores, 100.0)
    candidates = np.linspace(lo, hi, n_thresholds)

    best_pw  = dict(f1=-1, thr=None, p=0, r=0)
    best_pa  = dict(f1=-1, thr=None, p=0, r=0)

    for thr in candidates:
        y_pred = (scores >= thr).astype(int)
        if y_pred.sum() == 0:
            continue

        f1_pw = f1_score(labels, y_pred, zero_division=0)
        if f1_pw > best_pw["f1"]:
            best_pw.update(
                f1=f1_pw, thr=thr,
                p=precision_score(labels, y_pred, zero_division=0),
                r=recall_score(labels, y_pred, zero_division=0),
            )

        y_pa = point_adjust_vec(labels, y_pred)
        f1_pa = f1_score(labels, y_pa, zero_division=0)
        if f1_pa > best_pa["f1"]:
            best_pa.update(
                f1=f1_pa, thr=thr,
                p=precision_score(labels, y_pa, zero_division=0),
                r=recall_score(labels, y_pa, zero_division=0),
            )

    return best_pw, best_pa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True,
                        help="Nom du dossier dans ./results/")
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--n_thresholds", type=int, default=2000,
                        help="Nombre de seuils candidats testes")
    args = parser.parse_args()

    folder = f"{args.results_dir}/{args.setting}/"
    preds  = np.load(folder + "pred.npy")
    trues  = np.load(folder + "true.npy")
    labels = np.load(folder + "label.npy")

    err = ((preds - trues) ** 2).mean(axis=-1)
    scores = err.reshape(-1)
    labels_flat = labels.reshape(-1)

    print(f"\n{'='*60}")
    print(f"  GTA -- Grid search sur le seuil (protocole officiel du papier)")
    print(f"  setting = {args.setting}")
    print(f"{'='*60}\n")
    print(f"  Total points : {len(labels_flat):,}  "
          f"({labels_flat.sum():,} anomalies, "
          f"{100*labels_flat.sum()/len(labels_flat):.2f}%)")
    print(f"  Seuils testes : {args.n_thresholds}")
    print(f"  (estimation ~10-15 min pour ~1M points x 2000 seuils -- "
          f"reduire --n_thresholds si trop lent)")

    if len(np.unique(labels_flat)) < 2:
        print("\n  ATTENTION : un seul type de label present, AUROC non calculable.")
        return

    auc = roc_auc_score(labels_flat, scores) * 100
    print(f"\n  AUC-ROC : {auc:.4f}%  (independant du seuil)")

    best_pw, best_pa = grid_search(scores, labels_flat, args.n_thresholds)

    print(f"\n  -- Meilleur F1 point-wise (grid search) --")
    print(f"  Seuil     : {best_pw['thr']:.6f}")
    print(f"  Precision : {best_pw['p']*100:.2f}%")
    print(f"  Recall    : {best_pw['r']*100:.2f}%")
    print(f"  F1        : {best_pw['f1']*100:.2f}%")

    print(f"\n  -- Meilleur F1 point-adjust (grid search, protocole papier) --")
    print(f"  Seuil     : {best_pa['thr']:.6f}")
    print(f"  Precision : {best_pa['p']*100:.2f}%")
    print(f"  Recall    : {best_pa['r']*100:.2f}%")
    print(f"  F1        : {best_pa['f1']*100:.2f}%")

    print(f"\n{'='*60}")
    print("  TSV :")
    print("setting\tAUC\tF1_pw_best\tP_pw_best\tR_pw_best\tF1_PA_best\tP_PA_best\tR_PA_best")
    print(f"{args.setting}\t{auc:.4f}\t{best_pw['f1']*100:.4f}\t"
          f"{best_pw['p']*100:.4f}\t{best_pw['r']*100:.4f}\t"
          f"{best_pa['f1']*100:.4f}\t{best_pa['p']*100:.4f}\t{best_pa['r']*100:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

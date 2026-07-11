"""
train_eval_duogat_pct.py
=========================
Entraine DuoGAT sur nos donnees preprocessees (identique a
main_duogat_our_preproc_pa_gpu_progress.py), puis evalue avec un seuil
fixe = 99e percentile des scores du TRAIN -- protocole coherent avec
notre propre modele (test_ot_unsup.py) et avec GTA
(compute_auroc_gta_v2.py), plutot que le grid search sur le test
(bf_search_point) utilise par defaut dans prediction.py.

Split train/val : CHRONOLOGIQUE 70/30 (pas de shuffle aleatoire), au lieu
du split aleatoire par defaut du repo officiel (utils.py::
create_data_loaders, val_split=0.1, shuffle=True). Le split aleatoire
officiel melange des fenetres glissantes qui se chevauchent entre train
et validation, ce qui peut faire fuiter de l'information et rendre le
early stopping instable d'un run a l'autre. Ici, les 30% les plus
recents de la periode normale (train) servent de validation, dans
l'ordre chronologique d'origine -- meme logique que le split
normal/attaque deja utilise pour notre propre modele.

Entrainement : sinon strictement identique au script officiel adapte
(load_our_preprocessed, Trainer, DuoGAT, memes hyperparametres).
Seule l'etape d'evaluation change : au lieu d'appeler
predictor.predict_anomalies() (qui fait un grid search sur le test),
on appelle directement predictor.get_score() sur le train ET le test,
on fixe le seuil au 99e percentile des scores du train, puis on calcule
AUC-ROC (independant du seuil) et F1 point-wise / point-adjust sur le
test a ce seuil.

Usage :
  python train_eval_duogat_pct.py --dataset SWAT --epochs 30 --lookback 5
  python train_eval_duogat_pct.py --dataset WADI --epochs 30 --lookback 50
"""

import os
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from args import get_parser
from utils import *
from duogat import DuoGAT
from prediction import Predictor
from training import Trainer
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Subset


def create_data_loaders_chronological(train_dataset, batch_size, val_split=0.3,
                                      test_dataset=None):
    """Split train/val CHRONOLOGIQUE (pas de shuffle des fenetres), pour
    eviter la fuite de donnees du split aleatoire officiel (utils.py::
    create_data_loaders), qui melange des fenetres glissantes qui se
    chevauchent entre train et validation. Les val_split derniers % de
    la periode d'entrainement (normale) servent de validation, dans
    l'ordre chronologique d'origine.
    """
    dataset_size = len(train_dataset)
    split = int(dataset_size * (1 - val_split))

    train_indices = list(range(0, split))
    val_indices   = list(range(split, dataset_size))

    train_loader = DataLoader(Subset(train_dataset, train_indices),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(train_dataset, val_indices),
                            batch_size=batch_size, shuffle=False)

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ================================================================
# Chargement de nos donnees preprocessees (identique au script officiel adapte)
# ================================================================
def load_our_preprocessed(dataset, dif_n=1, our_data_dir=None):
    d = dataset.lower()
    if our_data_dir is None:
        if d in ["swat", "swat_a1_a2"]:
            our_data_dir = "./preprocessed_swat_unsup/"
        elif d == "wadi":
            our_data_dir = "./preprocessed_wadi_unsup/"
        else:
            raise ValueError(f"Dataset inconnu '{dataset}'")

    x_train = np.load(os.path.join(our_data_dir, "X_train.npy")).astype(np.float32)
    x_test  = np.load(os.path.join(our_data_dir, "X_test.npy")).astype(np.float32)
    y_test  = np.load(os.path.join(our_data_dir, "y_test.npy")).astype(np.uint8)

    def make_diff(X, dif_n):
        X_diff = np.zeros_like(X, dtype=np.float32)
        if dif_n > 0:
            X_diff[dif_n:] = X[dif_n:] - X[:-dif_n]
        return X_diff

    dif_x_train = make_diff(x_train, dif_n)
    dif_x_test  = make_diff(x_test, dif_n)

    print(f"  x_train : {x_train.shape}")
    print(f"  x_test  : {x_test.shape}")
    print(f"  y_test  : {y_test.shape}  ({y_test.sum()} anomalies, "
          f"{100*y_test.mean():.2f}%)")

    return x_train, x_test, y_test, dif_x_train, dif_x_test


# ================================================================
# Point-adjust
# ================================================================
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


def main():
    parser = get_parser()
    parser.add_argument("--our_data_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed pour torch/numpy -- permet de tester la "
                             "repetabilite en variant cette valeur d'un run "
                             "a l'autre")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    run_id = datetime.now().strftime("%d%m%Y_%H%M%S")
    dataset = args.dataset
    window_size = args.lookback
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda
    print_every = args.print_every
    log_tensorboard = args.log_tensorboard
    args_summary = str(args.__dict__)
    dif_n = args.dif_n

    print(args_summary)
    print(f"\n{'='*70}")
    print(f"  DuoGAT -- entrainement + evaluation (seuil = 99e percentile train)")
    print(f"  dataset={dataset}  epochs={n_epochs}")
    print(f"{'='*70}\n")

    output_path = f"output/{dataset}"
    x_train, x_test, y_test, dif_x_train, dif_x_test = load_our_preprocessed(
        dataset=dataset, dif_n=dif_n, our_data_dir=args.our_data_dir)

    log_dir = f"{output_path}/logs"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    save_path = f"{output_path}/{run_id}"
    os.makedirs(save_path)
    print("save_path:", save_path)

    x_train = torch.from_numpy(x_train).float()
    x_test  = torch.from_numpy(x_test).float()
    dif_x_train = torch.from_numpy(dif_x_train).float()
    dif_x_test  = torch.from_numpy(dif_x_test).float()

    n_features = x_train.shape[1]
    target_dims = get_target_dims(dataset)
    out_dim = n_features if target_dims is None else (
        1 if isinstance(target_dims, int) else len(target_dims))

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset  = SlidingWindowDataset(x_test, window_size, target_dims)
    train_loader, val_loader, test_loader = create_data_loaders_chronological(
        train_dataset, batch_size, val_split=0.3, test_dataset=test_dataset)

    dif_train_dataset = SlidingWindowDataset(dif_x_train, window_size, target_dims)
    dif_test_dataset  = SlidingWindowDataset(dif_x_test, window_size, target_dims)
    dif_train_loader, dif_val_loader, dif_test_loader = create_data_loaders_chronological(
        dif_train_dataset, batch_size, val_split=0.3, test_dataset=dif_test_dataset)

    model = DuoGAT(
        n_features, window_size, out_dim, batch_size=args.bs,
        gru_n_layers=args.gru_n_layers, gru_hid_dim=args.gru_hid_dim,
        forecast_n_layers=args.fc_n_layers, forecast_hid_dim=args.fc_hid_dim,
        dropout=args.dropout, alpha=args.alpha)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, factor=0.8)
    forecast_criterion = nn.MSELoss()
    early_stopping = 10

    trainer = Trainer(
        model, optimizer, scheduler, early_stopping, window_size, n_features,
        target_dims, n_epochs, batch_size, init_lr, forecast_criterion,
        use_cuda, save_path, log_dir, print_every, log_tensorboard, args_summary)

    train_time = trainer.fit(dif_train_loader, dif_val_loader, train_loader, val_loader)

    test_loss = trainer.evaluate(test_loader, dif_test_loader)
    print(f"Test forecast loss: {test_loss[0]:.5f}")

    trainer.load(f"{save_path}/model.pt")

    prediction_args = dict(dataset=dataset, target_dims=target_dims, save_path=save_path)
    predictor = Predictor(trainer.model, window_size, n_features, prediction_args)

    # ============================================================
    # Score d'anomalie harmonise avec notre propre modele et GTA :
    # MSE moyenne sur TOUTES les features, sans aucune standardisation
    # (StandardScaler retire), pour que les trois methodes (Ours, GTA,
    # DuoGAT) utilisent exactement la meme definition du score et du
    # protocole de seuillage (99e percentile des scores du train,
    # calcules avec la meme formule).
    # ============================================================
    def raw_scores(predictor, values, dif_values):
        """MSE moyenne sur toutes les features, un score scalaire par
        pas de temps -- meme formule que compute_scores() (notre modele)
        et le score GTA (((outputs-true)**2).mean(dim=-1))."""
        data = SlidingWindowDataset(values, predictor.window_size, predictor.target_dims)
        loader = torch.utils.data.DataLoader(data, batch_size=predictor.batch_size, shuffle=False)
        device = "cuda" if predictor.use_cuda and torch.cuda.is_available() else "cpu"

        dif_data = SlidingWindowDataset(dif_values, predictor.window_size, predictor.target_dims)
        dif_loader = torch.utils.data.DataLoader(dif_data, batch_size=predictor.batch_size, shuffle=False)

        predictor.model.eval()
        preds = []
        with torch.no_grad():
            for (x, y), (dif_x, dif_y) in zip(loader, dif_loader):
                x = x.to(device)
                dif_x = dif_x.to(device)
                y_hat = predictor.model(x, dif_x)
                preds.append(y_hat.detach().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        actual = values.detach().cpu().numpy()[predictor.window_size:]
        if predictor.target_dims is not None:
            actual = actual[:, predictor.target_dims]
        errors = (preds - actual) ** 2          # erreur quadratique (pas absolue)
        return errors.mean(axis=1)               # moyenne sur toutes les features

    print("\n-- Calcul des scores sur le TRAIN --")
    train_scores = raw_scores(predictor, x_train, dif_x_train)

    print("-- Calcul des scores sur le TEST --")
    test_scores = raw_scores(predictor, x_test, dif_x_test)

    threshold = float(np.percentile(train_scores, 99))
    print(f"  Seuil (pct=99, train, score harmonise MSE-moyenne) : {threshold:.6f}")

    label = y_test[window_size:]
    if len(label) != len(test_scores):
        raise ValueError(f"Longueurs incoherentes : label={len(label)}, "
                         f"scores={len(test_scores)}")

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
    print(f"  RESULTATS -- {dataset}  (seuil = 99e percentile du train)")
    print(f"{'='*70}")
    print(f"  AUC-ROC              : {auc:.4f}%")
    print(f"  Precision (point-wise)  : {p_pw:.2f}%")
    print(f"  Recall    (point-wise)  : {r_pw:.2f}%")
    print(f"  F1        (point-wise)  : {f1_pw:.2f}%")
    print(f"  Precision (point-adjust): {p_pa:.2f}%")
    print(f"  Recall    (point-adjust): {r_pa:.2f}%")
    print(f"  F1        (point-adjust): {f1_pa:.2f}%")
    print(f"{'='*70}")
    print("  TSV :")
    print("dataset\tAUC\tF1_pw\tP_pw\tR_pw\tF1_PA\tP_PA\tR_PA")
    print(f"{dataset}\t{auc:.4f}\t{f1_pw:.4f}\t{p_pw:.4f}\t{r_pw:.4f}\t"
          f"{f1_pa:.4f}\t{p_pa:.4f}\t{r_pa:.4f}")
    print(f"{'='*70}\n")

    metrics = dict(
        auc=auc, threshold=threshold,
        point_wise=dict(precision=p_pw, recall=r_pw, f1=f1_pw),
        point_adjust=dict(precision=p_pa, recall=r_pa, f1=f1_pa),
    )
    with open(os.path.join(save_path, "metrics_pct99.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    args.__dict__["train_time"] = train_time
    args.__dict__["metrics_pct99"] = metrics
    with open(f"{save_path}/config.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)


if __name__ == "__main__":
    main()

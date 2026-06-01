"""
preprocess_ot_unsup.py
======================
Preprocessing non supervisé SWaT et WADI.
Code repris du notebook preprocessing_unsupervised.ipynb (version validée).

Train → fichier NORMAL entier
Test  → fichier ATTACK entier avec labels binaires
"""

import os
import numpy as np
import pandas as pd

DATASET = "wadi"   # "swat" | "wadi"

# ================================================================
# CONFIG SWaT
# ================================================================
SWAT_NORMAL = "./dataset/SWaT_Dataset_Normal_v1.csv"
SWAT_ATTACK = "./dataset/SWaT_Dataset_Attack_v0.csv"
SWAT_OUT    = "./preprocessed_swat_unsup/"

# ================================================================
# CONFIG WADI
# ================================================================
WADI_NORMAL = "./dataset/WADI_14days.csv"
WADI_ATTACK = "./dataset/WADI_attackdataLABLE.csv"
WADI_OUT    = "./preprocessed_wadi_unsup/"
WADI_LABEL_COL   = "Attack LABLE (1:No Attack, -1:Attack)"
WADI_DROP_COLS   = {"row", "date", "time"}

USE_WINSOR = True
W_Q_LOW    = 0.001
W_Q_HIGH   = 0.999

# ================================================================
# Utils communs
# ================================================================
def fit_minmax(X):
    mn = np.min(X, axis=0).astype(np.float32)
    mx = np.max(X, axis=0).astype(np.float32)
    return mn, mx

def transform_minmax(X, mn, mx, eps=1e-12):
    denom = np.where(mx-mn < eps, 1.0, mx-mn)
    return ((X - mn) / denom).astype(np.float32)

def winsorize(X_raw, q_low, q_high):
    low  = np.quantile(X_raw, q_low,  axis=0)
    high = np.quantile(X_raw, q_high, axis=0)
    low  = np.minimum(low, high).astype(np.float32)
    high = np.maximum(low, high).astype(np.float32)
    return low, high

def to_numeric_and_impute(df):
    for c in df.columns:
        s = df[c].astype(str).str.replace(",", ".", regex=False)
        df[c] = pd.to_numeric(s, errors="coerce")
    df = df.dropna(axis=1, how="all")
    df = df.ffill().bfill().fillna(0.0)
    return df.replace([np.inf, -np.inf], 0.0)

# ================================================================
# SWaT loaders (sep=None auto-detect)
# ================================================================
def load_swat_normal(path):
    df = pd.read_csv(path, sep=None, engine="python")
    y  = None
    df = df.drop(columns=["Timestamp", " Timestamp", "Normal/Attack"],
                 errors="ignore")
    for c in df.columns:
        df[c] = df[c].astype(str).str.replace(",", ".", regex=False)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.ffill().bfill().fillna(0.0)
    df = df.replace([np.inf, -np.inf], 0.0)
    return df.to_numpy(dtype=np.float32), list(df.columns)

def load_swat_attack(path):
    df = pd.read_csv(path, sep=None, engine="python")
    y  = (df["Normal/Attack"].astype(str).str.strip() == "Attack"
          ).astype(np.uint8).values if "Normal/Attack" in df.columns else None
    df = df.drop(columns=["Timestamp", " Timestamp", "Normal/Attack"],
                 errors="ignore")
    for c in df.columns:
        df[c] = df[c].astype(str).str.replace(",", ".", regex=False)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.ffill().bfill().fillna(0.0)
    df = df.replace([np.inf, -np.inf], 0.0)
    return df.to_numpy(dtype=np.float32), y, list(df.columns)

# ================================================================
# WADI loaders
# ================================================================
def find_header_row(path, max_lines=80):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i in range(max_lines):
            line = f.readline()
            if not line: break
            low = line.lower()
            if ("row" in low) and ("date" in low) and ("time" in low):
                return i
    return 0

def read_wadi_csv(path):
    header_row = find_header_row(path)
    df = pd.read_csv(path, header=header_row, sep=",",
                     engine="c", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def load_wadi_normal(path):
    df = read_wadi_csv(path)
    drop = [c for c in df.columns if c.strip().lower() in WADI_DROP_COLS]
    df   = df.drop(columns=drop + [WADI_LABEL_COL], errors="ignore")
    df   = to_numeric_and_impute(df)
    return df.to_numpy(dtype=np.float32), list(df.columns)

def load_wadi_attack(path):
    df = read_wadi_csv(path)
    drop = [c for c in df.columns if c.strip().lower() in WADI_DROP_COLS]
    df   = df.drop(columns=drop, errors="ignore")
    # label
    label_col = None
    for c in df.columns:
        if "attack" in c.lower() and ("lable" in c.lower() or "label" in c.lower()):
            label_col = c; break
    if label_col is None:
        raise ValueError(f"Label column not found. Columns: {list(df.columns)[:20]}")
    y = (pd.to_numeric(df[label_col], errors="coerce").to_numpy() == -1
         ).astype(np.uint8)
    df = df.drop(columns=[label_col], errors="ignore")
    df = to_numeric_and_impute(df)
    return df.to_numpy(dtype=np.float32), y, list(df.columns)

# ================================================================
# Pipeline commun
# ================================================================
def process(X_tr_raw, X_te_raw, y_te, cols_n, cols_a, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Alignement colonnes
    if cols_n != cols_a:
        common = [c for c in cols_n if c in cols_a]
        print(f"  ⚠ Réalignement : {len(common)} colonnes communes")
        idx_n = [cols_n.index(c) for c in common]
        idx_a = [cols_a.index(c) for c in common]
        X_tr_raw = X_tr_raw[:, idx_n]
        X_te_raw = X_te_raw[:, idx_a]
        cols_n   = common

    print(f"  Features : {len(cols_n)}")
    print(f"  Normal : {X_tr_raw.shape}  atk={0}")
    print(f"  Attack : {X_te_raw.shape}  atk={y_te.sum()} ({y_te.mean()*100:.1f}%)")

    if USE_WINSOR:
        w_low, w_high = winsorize(X_tr_raw, W_Q_LOW, W_Q_HIGH)
        X_tr_raw = np.clip(X_tr_raw, w_low, w_high).astype(np.float32)
        X_te_raw = np.clip(X_te_raw, w_low, w_high).astype(np.float32)
        np.save(os.path.join(out_dir, "winsor_low_raw.npy"),  w_low)
        np.save(os.path.join(out_dir, "winsor_high_raw.npy"), w_high)

    mn, mx   = fit_minmax(X_tr_raw)
    X_tr     = transform_minmax(X_tr_raw, mn, mx)
    X_te     = transform_minmax(X_te_raw, mn, mx)

    np.save(os.path.join(out_dir, "X_train.npy"),    X_tr)
    np.save(os.path.join(out_dir, "X_test.npy"),     X_te)
    np.save(os.path.join(out_dir, "y_test.npy"),     y_te.astype(np.uint8))
    np.save(os.path.join(out_dir, "scaler_min.npy"), mn)
    np.save(os.path.join(out_dir, "scaler_max.npy"), mx)
    pd.Series(cols_n).to_csv(
        os.path.join(out_dir, "feature_columns.csv"),
        index=False, header=False)

    print(f"  ✔ Sauvegardé → {out_dir}\n")

# ================================================================
# MAIN
# ================================================================
def main():
    print(f"\n── Preprocessing NON SUPERVISÉ — {DATASET.upper()} ──")

    if DATASET == "swat":
        X_tr_raw, cols_n = load_swat_normal(SWAT_NORMAL)
        X_te_raw, y_te, cols_a = load_swat_attack(SWAT_ATTACK)
        process(X_tr_raw, X_te_raw, y_te, cols_n, cols_a, SWAT_OUT)

    elif DATASET == "wadi":
        X_tr_raw, cols_n = load_wadi_normal(WADI_NORMAL)
        X_te_raw, y_te, cols_a = load_wadi_attack(WADI_ATTACK)
        process(X_tr_raw, X_te_raw, y_te, cols_n, cols_a, WADI_OUT)

if __name__ == "__main__":
    main()

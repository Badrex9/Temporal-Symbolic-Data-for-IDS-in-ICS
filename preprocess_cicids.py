"""
preprocess_cicids.py
====================
Préprocessing CICIDS2017 → sauvegarde numpy.
Pas de SMOTE — le déséquilibre est géré par class-weighted loss à l'entraînement.

Sorties dans ./preprocessed_cicids/ :
  X_all.npy, Y_all.npy, sip_all.npy, dip_all.npy, dport_all.npy

Usage :
  python preprocess_cicids.py [--csv_dir ./TrafficLabelling/] [--out_dir ./preprocessed_cicids/]
"""

import os, glob, argparse
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import LabelEncoder


def load_raw(csv_dir):
    files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {csv_dir}")
    dfs = []
    for f in sorted(files):
        try:
            df = pd.read_csv(f, encoding="cp1252", low_memory=False)
            dfs.append(df)
            print(f"  ✔ {os.path.basename(f)} : {len(df):,} lignes")
        except Exception as e:
            print(f"  ⚠ {os.path.basename(f)} : {e}")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total brut : {len(combined):,} lignes × {len(combined.columns)} colonnes")
    return combined


def build_features(df):
    """
    Structure CICIDS2017 :
      col 0  : Flow ID
      col 1  : Source IP
      col 2  : Source Port
      col 3  : Destination IP
      col 4  : Destination Port
      col 5  : Protocol
      col 6  : Timestamp
      col 7..-2 : features numériques (76 features)
      col -1 : Label
    """
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    label_col    = df.columns[-1]
    feature_cols = df.columns[7:-1]

    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=feature_cols)
    print(f"  Après nettoyage : {len(df):,} lignes")

    sip   = df.iloc[:, 1].astype(str).values
    dip   = df.iloc[:, 3].astype(str).values
    sport = df.iloc[:, 2].astype(float).astype(int)
    dport = df.iloc[:, 4].astype(float).astype(int)

    X_num = df[feature_cols].values.astype(np.float32)

    # Protocol one-hot encoding
    proto = pd.to_numeric(df.iloc[:, 5], errors="coerce").fillna(0).astype(int)
    ohe   = pd.get_dummies(proto, dtype=int).values

    # Ports + features numériques normalisés
    ports = np.stack([sport.values, dport.values], axis=1).astype(np.float32)
    X_raw = np.concatenate([ports, X_num], axis=1)
    X_raw = minmax_scale(X_raw, axis=0).astype(np.float32)

    # OHE protocol préfixé → 82 features au total
    X = np.concatenate([ohe, X_raw], axis=1).astype(np.float32)

    le = LabelEncoder()
    Y  = le.fit_transform(df[label_col].astype(str).values).astype(np.int64)

    print(f"  X shape : {X.shape}   n_classes : {len(le.classes_)}")
    print(f"  Distribution : {dict(Counter(Y))}")
    print(f"  Label mapping : { {i:c for i,c in enumerate(le.classes_)} }")
    return X, Y, sip, dip, sport.values, dport.values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", default="./TrafficLabelling/")
    parser.add_argument("--out_dir", default="./preprocessed_cicids/")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("\n── Chargement CSV ──")
    df = load_raw(args.csv_dir)

    print("\n── Extraction features ──")
    X, Y, sip, dip, sport, dport = build_features(df)

    print(f"\n── Sauvegarde dans {args.out_dir} ──")
    np.save(f"{args.out_dir}/X_all.npy",     X)
    np.save(f"{args.out_dir}/Y_all.npy",     Y)
    np.save(f"{args.out_dir}/sip_all.npy",   sip)
    np.save(f"{args.out_dir}/dip_all.npy",   dip)
    np.save(f"{args.out_dir}/dport_all.npy", dport.astype(np.int64))

    print(f"  X : {X.shape}  Y : {Y.shape}")
    print("✔ Preprocessing terminé\n")


if __name__ == "__main__":
    main()

"""
prepare_gta_data.py
====================
Convertit les CSV bruts SWaT/WADI vers le format exact attendu par le
DataLoader officiel de GTA (zackchen-lb/GTA, data/data_loader_dad.py).

Le repo hardcode :
  - SWaT train : 'SWaT_normaldata_downsampled.csv', colonne timestamp
    nommee ' Timestamp' (avec espace), colonnes capteurs ensuite.
  - SWaT test  : 'SWaT_attackdata_downsampled.csv', memes colonnes +
    'Normal/Attack' en dernier -- DOIT etre numerique (0/1), pas des
    chaines, car le code fait ensuite `.long()` sur le label.
  - WADI train : 'WADI_14days_downsampled.csv', colonne 'date' unique
    (pas Row/Date/Time separes), colonnes capteurs ensuite.
  - WADI test  : 'WADI_attackdata_downsampled.csv', memes colonnes +
    'label' en dernier, numerique (0=normal, 1=attaque).

Downsampling : par blocs de taille --downsample (moyenne des capteurs,
max du label pour ne perdre aucune attaque courte), necessaire car
WADI/SWaT bruts font des centaines de milliers de lignes -- entrainer
un transformer dessus sans reduction serait tres long.

Usage :
  python prepare_gta_data.py \
      --swat_normal SWaT_Dataset_Normal_v1.csv \
      --swat_attack SWaT_Dataset_Attack_v0.csv \
      --wadi_normal WADI_14days.csv \
      --wadi_attack WADI_attackdataLABLE.csv \
      --out_dir ./gta_data/ \
      --downsample 10
"""

import argparse
import os
import numpy as np
import pandas as pd


def downsample_block(df, block, label_col=None):
    """Downsample par blocs de `block` lignes consecutives.
    - colonnes capteurs : MEDIANE du bloc (et non la moyenne -- le papier
      GTA precise explicitement "downsampled to one measurement every 10
      seconds by taking the median values following [GDN]")
    - colonne label (si fournie) : max du bloc (une seule ligne
      d'attaque dans le bloc => bloc marque attaque)
    """
    n = len(df) // block
    df = df.iloc[: n * block]
    sensor_cols = [c for c in df.columns if c != label_col]

    grouped = df[sensor_cols].groupby(np.arange(len(df)) // block).median()
    if label_col is not None:
        grouped[label_col] = df[label_col].groupby(np.arange(len(df)) // block).max().values
    return grouped.reset_index(drop=True)


# ================================================================
# SWaT
# ================================================================
def prepare_swat(normal_path, attack_path, out_dir, downsample):
    print("\n-- SWaT --")

    # ---- Train (normal) ----
    df_n = pd.read_csv(normal_path, sep=None, engine="python")
    df_n.columns = [c.strip() for c in df_n.columns]
    print(f"  [debug] colonnes normal (10 premieres) : {list(df_n.columns[:10])}")

    ts_col_raw = df_n.columns[0]  # premiere colonne = timestamp, quel que soit son nom exact
    if "Normal/Attack" in df_n.columns:
        df_n = df_n.drop(columns=["Normal/Attack"])
    df_n = df_n.rename(columns={ts_col_raw: " Timestamp"})

    ts_col = " Timestamp"
    sensor_cols = [c for c in df_n.columns if c != ts_col]
    for c in sensor_cols:
        df_n[c] = df_n[c].astype(str).str.replace(",", ".", regex=False)
    df_n[sensor_cols] = df_n[sensor_cols].apply(pd.to_numeric, errors="coerce")
    n_before = len(df_n)
    df_n = df_n.dropna(subset=sensor_cols)
    print(f"  [debug] normal : {n_before} lignes avant dropna, {len(df_n)} apres")

    # Normaliser le timestamp en ISO uniforme -- le CSV original SWaT a des
    # formats inconsistants (heures a 1 ou 2 chiffres, espacement variable)
    # qui font planter le parsing strict du repo officiel sinon.
    df_n[ts_col] = pd.to_datetime(df_n[ts_col], dayfirst=True, errors="coerce")
    df_n = df_n.dropna(subset=[ts_col])

    ts = df_n[ts_col].reset_index(drop=True)
    df_n_ds = downsample_block(df_n[sensor_cols], downsample)
    n_blocks = len(df_n_ds)
    df_n_ds.insert(0, ts_col, ts.iloc[:: downsample][: n_blocks].values)

    out_n = os.path.join(out_dir, "SWaT_normaldata_downsampled.csv")
    df_n_ds.to_csv(out_n, index=False)
    print(f"  train : {normal_path} ({len(df_n):,} lignes) "
          f"-> {out_n} ({len(df_n_ds):,} lignes)")

    # ---- Test (attack) ----
    df_a = pd.read_csv(attack_path, sep=None, engine="python")
    df_a.columns = [c.strip() for c in df_a.columns]
    print(f"  [debug] colonnes attack (10 premieres) : {list(df_a.columns[:10])}")
    print(f"  [debug] derniere colonne attack : {df_a.columns[-1]}")

    ts_col_raw_a = df_a.columns[0]
    df_a = df_a.rename(columns={ts_col_raw_a: " Timestamp"})

    label_col = None
    for c in df_a.columns:
        if "normal" in c.strip().lower() and "attack" in c.strip().lower():
            label_col = c
            break
    if label_col is None:
        raise ValueError(f"Colonne label introuvable parmi : {list(df_a.columns)}")
    print(f"  [debug] colonne label detectee : '{label_col}'")

    df_a[label_col] = (df_a[label_col].astype(str).str.strip()
                        .str.lower().map(lambda v: 0 if v == "normal" else 1))

    ts_col = " Timestamp"
    sensor_cols = [c for c in df_a.columns if c not in (ts_col, label_col)]
    for c in sensor_cols:
        df_a[c] = df_a[c].astype(str).str.replace(",", ".", regex=False)
    df_a[sensor_cols] = df_a[sensor_cols].apply(pd.to_numeric, errors="coerce")
    n_before = len(df_a)
    df_a = df_a.dropna(subset=sensor_cols)
    print(f"  [debug] attack : {n_before} lignes avant dropna, {len(df_a)} apres")

    # Normaliser le timestamp en ISO uniforme (meme raison que pour le train)
    df_a[ts_col] = pd.to_datetime(df_a[ts_col], dayfirst=True, errors="coerce")
    df_a = df_a.dropna(subset=[ts_col])

    ts = df_a[ts_col].reset_index(drop=True)
    df_a_ds = downsample_block(df_a[sensor_cols + [label_col]], downsample,
                               label_col=label_col)
    n_blocks = len(df_a_ds)
    df_a_ds.insert(0, ts_col, ts.iloc[:: downsample][: n_blocks].values)
    # remettre le label en derniere colonne (insert(0,...) l'a decale)
    cols = [c for c in df_a_ds.columns if c != label_col] + [label_col]
    df_a_ds = df_a_ds[cols]

    out_a = os.path.join(out_dir, "SWaT_attackdata_downsampled.csv")
    df_a_ds.to_csv(out_a, index=False)
    print(f"  test  : {attack_path} ({len(df_a):,} lignes) "
          f"-> {out_a} ({len(df_a_ds):,} lignes, "
          f"{df_a_ds[label_col].sum()} blocs attaque)")


# ================================================================
# WADI
# ================================================================
def find_label_col(columns):
    for c in columns:
        cl = c.strip().lower()
        if "attack" in cl and ("lable" in cl or "label" in cl):
            return c
    raise ValueError(f"Colonne label introuvable parmi : {list(columns)[:10]}...")


def prepare_wadi(normal_path, attack_path, out_dir, downsample):
    print("\n-- WADI --")

    # ---- Train (normal) ----
    df_n = pd.read_csv(normal_path, low_memory=False)
    df_n.columns = [c.strip() for c in df_n.columns]
    drop_cols = [c for c in df_n.columns if c.strip().lower() == "row"]
    date_col = "date"
    df_n["date"] = pd.to_datetime(
        df_n["Date"].astype(str) + " " + df_n["Time"].astype(str),
        errors="coerce")
    df_n = df_n.drop(columns=drop_cols + ["Date", "Time"], errors="ignore")
    df_n = df_n.dropna(subset=["date"])

    sensor_cols = [c for c in df_n.columns if c != date_col]
    for c in sensor_cols:
        df_n[c] = df_n[c].astype(str).str.replace(",", ".", regex=False)
    df_n[sensor_cols] = df_n[sensor_cols].apply(pd.to_numeric, errors="coerce")
    # colonnes 100% vides OU constantes (std=0) -- le papier GTA/GDN
    # rapporte 112 capteurs pour WADI (127 bruts - colonnes vides/constantes)
    df_n = df_n.dropna(axis=1, how="all")
    sensor_cols = [c for c in df_n.columns if c != date_col]
    const_cols = [c for c in sensor_cols if df_n[c].std(skipna=True) == 0]
    df_n = df_n.drop(columns=const_cols)
    sensor_cols = [c for c in df_n.columns if c != date_col]
    df_n[sensor_cols] = df_n[sensor_cols].ffill().bfill()
    n_before = len(df_n)
    df_n = df_n.dropna(subset=sensor_cols)
    print(f"  [debug] WADI normal : {n_before} lignes avant dropna, {len(df_n)} apres "
          f"({len(sensor_cols)} capteurs conserves, {len(const_cols)} colonnes "
          f"constantes supprimees)")
    if len(sensor_cols) != 112:
        print(f"  [ATTENTION] Le papier GTA/GDN rapporte 112 capteurs pour WADI, "
              f"nous en avons {len(sensor_cols)}. Ecart possible vs les resultats "
              f"publies -- verifier si d'autres colonnes doivent etre exclues.")

    ts = df_n[date_col].reset_index(drop=True)
    df_n_ds = downsample_block(df_n[sensor_cols], downsample)
    n_blocks = len(df_n_ds)
    df_n_ds.insert(0, date_col, ts.iloc[:: downsample][: n_blocks].values)

    out_n = os.path.join(out_dir, "WADI_14days_downsampled.csv")
    df_n_ds.to_csv(out_n, index=False)
    print(f"  train : {normal_path} ({len(df_n):,} lignes) "
          f"-> {out_n} ({len(df_n_ds):,} lignes)")

    # ---- Test (attack) : memes colonnes capteurs que le train (coherence
    #      du schema train/test -- pas de recalcul independant des
    #      colonnes vides/constantes sur le test) ----
    df_a = pd.read_csv(attack_path, low_memory=False)
    df_a.columns = [c.strip() for c in df_a.columns]
    label_col_raw = find_label_col(df_a.columns)
    drop_cols = [c for c in df_a.columns if c.strip().lower() == "row"]

    df_a["date"] = pd.to_datetime(
        df_a["Date"].astype(str) + " " + df_a["Time"].astype(str),
        errors="coerce")
    df_a["label"] = (pd.to_numeric(df_a[label_col_raw], errors="coerce") == -1).astype(int)
    df_a = df_a.drop(columns=drop_cols + ["Date", "Time", label_col_raw], errors="ignore")
    df_a = df_a.dropna(subset=["date"])

    missing = [c for c in sensor_cols if c not in df_a.columns]
    if missing:
        raise ValueError(f"Colonnes du train absentes du fichier attaque : {missing}")

    for c in sensor_cols:
        df_a[c] = df_a[c].astype(str).str.replace(",", ".", regex=False)
    df_a[sensor_cols] = df_a[sensor_cols].apply(pd.to_numeric, errors="coerce")
    df_a = df_a[["date", "label"] + sensor_cols]
    df_a[sensor_cols] = df_a[sensor_cols].ffill().bfill()
    n_before = len(df_a)
    df_a = df_a.dropna(subset=sensor_cols)
    print(f"  [debug] WADI attack : {n_before} lignes avant dropna, {len(df_a)} apres "
          f"({len(sensor_cols)} capteurs, memes que le train)")

    ts = df_a["date"].reset_index(drop=True)
    df_a_ds = downsample_block(df_a[sensor_cols + ["label"]], downsample,
                               label_col="label")
    n_blocks = len(df_a_ds)
    df_a_ds.insert(0, "date", ts.iloc[:: downsample][: n_blocks].values)
    cols = [c for c in df_a_ds.columns if c != "label"] + ["label"]
    df_a_ds = df_a_ds[cols]

    out_a = os.path.join(out_dir, "WADI_attackdata_downsampled.csv")
    df_a_ds.to_csv(out_a, index=False)
    print(f"  test  : {attack_path} ({len(df_a):,} lignes) "
          f"-> {out_a} ({len(df_a_ds):,} lignes, "
          f"{df_a_ds['label'].sum()} blocs attaque)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--swat_normal", required=True)
    parser.add_argument("--swat_attack", required=True)
    parser.add_argument("--wadi_normal", required=True)
    parser.add_argument("--wadi_attack", required=True)
    parser.add_argument("--out_dir", default="./gta_data/")
    parser.add_argument("--downsample", type=int, default=10,
                        help="Taille de bloc pour le downsampling (mediane "
                             "capteurs, max label). Le papier GTA precise "
                             "explicitement un downsampling a 1 mesure / "
                             "10 secondes par mediane, suivant GDN.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    prepare_swat(args.swat_normal, args.swat_attack, args.out_dir, args.downsample)
    prepare_wadi(args.wadi_normal, args.wadi_attack, args.out_dir, args.downsample)

    print(f"\nFichiers prets dans {args.out_dir} :")
    print("  SWaT_normaldata_downsampled.csv")
    print("  SWaT_attackdata_downsampled.csv")
    print("  WADI_14days_downsampled.csv")
    print("  WADI_attackdata_downsampled.csv")


if __name__ == "__main__":
    main()

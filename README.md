# Temporal-Symbolic Data Representation for Intrusion Detection in Industrial Control Systems

> **Enzo Zamaï, David Espes, Audrey C. Therrien, Catherine Dezan**
> Université de Bretagne Occidentale, Brest, France — Université de Sherbrooke, Canada
> *Computers & Security* (under review)

---

## Overview

This repository contains the official implementation of the temporal-symbolic IDS framework presented in the paper. The approach transforms raw system observations (network packets for IT environments, sensor snapshots for OT environments) into structured temporal matrices and processes them with a compact 1D convolutional architecture equipped with a **mask-guided variable-window pooling** mechanism that learns, end-to-end and per sample, the effective temporal horizon required for anomaly detection.

The same architectural principle operates across both ICS layers within a single framework:
- **IT layer** — supervised multiclass classification on CICIDS2017 network flows
- **OT layer** — unsupervised next-step prediction on SWaT and WADI sensor time series

This repository also includes our reproduction of the two strongest OT baselines, **GTA** and **DuoGAT**, retrained and re-evaluated on our own preprocessing pipeline under a fully harmonized protocol (see [Reproducing GTA and DuoGAT](#reproducing-gta-and-duogat-baselines) below).

---

## Repository Structure

```
.
├── preprocess_cicids.py          # Preprocessing pipeline for CICIDS2017
├── train_cicids.py               # Training script — IT supervised (all d_hist × pooling configs)
├── test_cicids.py                # Evaluation script — IT (paper groupings, F1 global)
│
├── preprocess_ot_unsup.py        # Preprocessing pipeline for SWaT and WADI
├── train_ot_unsup.py             # Training script — OT unsupervised (all configs)
├── test_ot_unsup.py              # Evaluation script — OT (F1_PA, AUC-ROC)
│
├── benchmark_server.py           # Latency benchmark (ours) — GPU + CPU (training server)
├── benchmark_arm.py              # Latency benchmark (ours) — ARM Cortex-A72 (AWS Graviton2)
├── bench_duogat.py               # Latency/RAM benchmark — DuoGAT, ARM Cortex-A72
├── bench_duogat_gpu.py           # Latency benchmark — DuoGAT, GPU + CPU (training server)
├── bench_gta.py                  # Latency/RAM benchmark — GTA, ARM Cortex-A72
├── bench_gta_gpu.py              # Latency benchmark — GTA, GPU + CPU (training server)
│
├── duogat/                       # DuoGAT reproduction (see Reproducing GTA and DuoGAT)
├── gta/                          # GTA reproduction (see Reproducing GTA and DuoGAT)
│
├── cicids_pooling/                # Saved CICIDS2017 models (.pth)
├── swat_unsup_models/             # Saved SWaT models (.pth)
├── wadi_unsup_models/             # Saved WADI models (.pth)
│
├── preprocessed_cicids/          # Preprocessed CICIDS2017 numpy arrays
├── preprocessed_swat_unsup/      # Preprocessed SWaT numpy arrays
└── preprocessed_wadi_unsup/      # Preprocessed WADI numpy arrays
```

---

## Dependencies

```bash
pip install torch numpy pandas scikit-learn tqdm thop
```

Tested with Python 3.10, PyTorch 2.11 (CUDA 13.0). No other dependencies are required for our own model. The `duogat/` and `gta/` reproductions have their own additional dependencies — see their respective sections below.

---

## Datasets

| Dataset | Layer | Task | Source |
|---|---|---|---|
| CICIDS2017 | IT | Supervised classification | [UNB CIC](https://www.unb.ca/cic/datasets/ids-2017.html) |
| SWaT | OT | Unsupervised anomaly detection | [iTrust SUTD](https://itrust.sutd.edu.sg/itrust-labs_datasets/) |
| WADI | OT | Unsupervised anomaly detection | [iTrust SUTD](https://itrust.sutd.edu.sg/itrust-labs_datasets/) |

Place the raw files as follows before preprocessing:

```
dataset/
├── TrafficLabelling/          # CICIDS2017 .csv files
├── SWaT_Dataset_Normal_v1.csv
├── SWaT_Dataset_Attack_v0.csv
├── WADI_14days.csv
└── WADI_attackdataLABLE.csv
```

---

## Usage

### 1. Preprocessing

**CICIDS2017**
```bash
python preprocess_cicids.py --csv_dir ./dataset/TrafficLabelling/ --out_dir ./preprocessed_cicids/
```

**SWaT / WADI** — set `DATASET = "swat"` or `DATASET = "wadi"` at the top of the file, then:
```bash
python preprocess_ot_unsup.py
```

### 2. Training

**CICIDS2017** — trains all 15 configurations (5 window sizes × 3 pooling strategies):
```bash
python train_cicids.py
```

Single configuration:
```bash
python train_cicids.py --d_hist 10 --pooling maskguided
```

**SWaT / WADI** — set `DATASET` at the top of the file, then:
```bash
python train_ot_unsup.py
```

Single configuration:
```bash
python train_ot_unsup.py --dataset swat --d_hist 10 --pooling maskguided
```

### 3. Evaluation

**CICIDS2017** — evaluates all saved models with paper groupings (DoS, Patator, WebAttack):
```bash
python test_cicids.py
```

Single configuration:
```bash
python test_cicids.py --d_hist 10 --pooling maskguided
```

**SWaT / WADI** — reports F1_PA (point-adjust) and AUC-ROC:
```bash
python test_ot_unsup.py --dataset swat
python test_ot_unsup.py --dataset wadi
```

### 4. Benchmarking

**Server (GPU + CPU):**
```bash
python benchmark_server.py
```

**ARM Cortex-A72 (AWS Graviton2 / Raspberry Pi 4-class estimate):**
```bash
python benchmark_arm.py
```

---

## Model Naming Convention

Saved models follow the pattern:

```
model_cicids_{pooling}_d{d_hist}.pth       # CICIDS2017
unsup_{dataset}_{pooling}_d{d_hist}.pth   # SWaT / WADI
```

Examples: `model_cicids_maskguided_d10.pth`, `unsup_swat_avgpool_d5.pth`

---

## Architecture

The model consists of three temporal 1D convolutional layers, two mask-guided pooling stages, and a four-layer MLP head. Convolutions operate exclusively along the temporal axis (kernel shape `1 × q`) to avoid introducing artificial dependencies between feature rows.

| Dataset | d_sw | Params | MACs/inf |
|---|---|---|---|
| CICIDS2017 | 10 | 74,257 | 272,128 |
| SWaT | 10 | 91,221 | 278,656 |
| WADI | 5 | 169,613 | 466,816 |

---

## Experimental Protocol

**CICIDS2017** — stratified random split (70% train / 30% test, `random_state=843`), class-weighted cross-entropy loss (weights ∝ 1/√frequency), 100 epochs with early stopping on validation F1 (patience 15 epochs). Heartbleed and Infiltration classes are excluded from the global F1 computation due to insufficient test samples (3 and 11 respectively).

**SWaT / WADI** — unsupervised protocol: models are trained exclusively on the normal operation period and evaluated on the attack period, under a strictly chronological train/test separation. The anomaly score is the mean squared forecasting error averaged over all features, with no additional per-split standardization. The detection threshold is fixed at the 99th percentile of anomaly scores computed on the normal training data. Performance is reported using the point-adjust protocol (F1_PA) alongside AUC-ROC, consistent with standard practice in the ICS anomaly detection literature.

**GTA and DuoGAT reproductions** use the exact same anomaly score definition and the exact same 99th-percentile thresholding rule as our own model, rather than the method-specific exhaustive grid search over the labeled test set used in their original papers, which is not representative of a deployable detection procedure since it requires test-time attack labels that are not available in practice. DuoGAT's official training pipeline draws its internal train/validation split uniformly at random from overlapping sliding windows; we replace it with a chronological 70/30 split, better suited to time-series data and avoiding the risk of near-duplicate windows leaking across the split. GTA's official implementation already partitions data chronologically by construction. Results were verified stable across repeated training runs with different random seeds. See `duogat/README.md` and `gta/README.md` for full reproduction details.

---

## Reproducing GTA and DuoGAT baselines

We reproduce **GTA**~[[Chen et al., 2021]](https://github.com/zackchen-lb/GTA) and **DuoGAT**~[[Lee et al., 2023]](https://github.com/ByeongtaePark/DuoGAT) because they report among the strongest published OT detection results on SWaT and WADI, and therefore constitute the most relevant competitors for assessing the proposed model. The remaining OT baselines discussed in the paper (OmniAnomaly, USAD, GDN, DE-CNN) are reported from their published results rather than reproduced, since their published detection performance is already lower than that of GTA and DuoGAT.

The `duogat/` and `gta/` directories each contain a modified copy of the corresponding official implementation, adapted to (1) load our own preprocessed SWaT/WADI data, (2) fix a small number of library-compatibility issues unrelated to our contribution, and (3) evaluate under the harmonized protocol described above. Original licenses apply in both directories — see each subdirectory's `LICENSE` file, and please cite the original papers if you use this code.

**DuoGAT** (`duogat/`):
```bash
cd duogat
python train_eval_duogat_pct.py --dataset SWAT --epochs 30 --lookback 5  --our_data_dir ../preprocessed_swat_unsup/
python train_eval_duogat_pct.py --dataset WADI --epochs 30 --lookback 50 --our_data_dir ../preprocessed_wadi_unsup/
```
See `duogat/README.md` for the full list of changes applied to the official repository (chronological train/validation split, harmonized anomaly score and threshold, minor CUDA/CPU device-handling fix).

**GTA** (`gta/`):
```bash
cd gta
python prepare_gta_data.py \
    --swat_normal ../dataset/SWaT_Dataset_Normal_v1.csv \
    --swat_attack ../dataset/SWaT_Dataset_Attack_v0.csv \
    --wadi_normal ../dataset/WADI_14days.csv \
    --wadi_attack ../dataset/WADI_attackdataLABLE.csv \
    --out_dir ./gta_data/ --downsample 10

python run_gta_train_eval.py --dataset swat --data_dir ./gta_data/
python run_gta_train_eval.py --dataset wadi --data_dir ./gta_data/

python compute_auroc_gta_v2.py --dataset swat --data_dir ./gta_data/ --setting gta_SWaT_sl60_ll30_pl24
python compute_auroc_gta_v2.py --dataset wadi --data_dir ./gta_data/ --setting gta_WADI_sl60_ll30_pl24
```
See `gta/README.md` for the full list of changes applied to the official repository (pandas/numpy compatibility fixes, a naming bug fix, and the harmonized anomaly score/threshold evaluation script).

---

## Results

**CICIDS2017** (weighted F1, 7 retained classes):

| Method | F1 (%) | Params |
|---|---|---|
| Han et al. — Transformer | 97.83 | — |
| Sun et al. — CNN+LSTM | — | — |
| **Ours — maskguided, d=10** | **99.77** | **74k** |

**SWaT** (AUC-ROC and F1_PA, point-adjust protocol; GTA and DuoGAT results are our own reproduction under the harmonized 99th-percentile threshold protocol described above):

| Method | AUC-ROC | F1_PA | Params |
|---|---|---|---|
| OmniAnomaly | — | 0.78 | — |
| USAD | — | 0.85 | 3.9M |
| GDN | — | 0.81 | 5k |
| DE-CNN | — | 0.87 | 35,836 |
| GTA | 0.7902 | 0.6097 | 832,407 |
| DuoGAT | 0.8597 | 0.8809 | 170,384 |
| **Ours — maskguided, d=10** | **0.8743** | **0.9153** | **91,221** |

**WADI** (AUC-ROC and F1_PA, point-adjust protocol; GTA and DuoGAT results are our own reproduction under the harmonized 99th-percentile threshold protocol described above):

| Method | AUC-ROC | F1_PA | Params |
|---|---|---|---|
| OmniAnomaly | — | 0.23 | — |
| USAD | — | 0.43 | 3.9M |
| GDN | — | 0.57 | 20k |
| DE-CNN | — | 0.72 | 283,391 |
| GTA | 0.5995 | 0.4697 | 1,086,716 |
| DuoGAT | **0.7356** | 0.7574 | 224,264 |
| **Ours — maskguided, d=5** | 0.6637 | **0.8257** | **169,613** |

Under this harmonized protocol, our model achieves the best F1_PA on both SWaT and WADI among the three reproduced deep-learning-based methods, DuoGAT achieves the best AUC-ROC on WADI, and GTA — despite reporting among the strongest results in its original paper under a test-set grid-search threshold — performs considerably worse than both other methods once evaluated under a common, deployment-realistic thresholding rule.

---

## Deployment

All configurations were benchmarked at batch size 1 on a server workstation (Intel Core Ultra 7 265, NVIDIA RTX 4000 SFF Ada) and on an AWS Graviton2 instance (`t4g.micro`, ARM Cortex-A72 @ 2.5 GHz), which is microarchitecturally identical to the Broadcom BCM2711 of the Raspberry Pi 4. Raspberry Pi 4-class throughput estimates are obtained by scaling the measured Graviton2 throughput by the clock frequency ratio ×0.72 (2.5 GHz → 1.8 GHz); these are estimates derived from direct ARM CPU-class measurements, not direct Raspberry Pi 4 measurements.

**Ours:**

| Dataset | GPU (inf/s) | CPU (inf/s) | RPi4-class est. (inf/s) | Constraint | Margin |
|---|---|---|---|---|---|
| CICIDS2017 | 1,332 | 1,356 | ~420 | 25 flows/s | ×16.8 |
| SWaT | 1,073 | 1,144 | ~374 | 1 Hz | ×374 |
| WADI | 1,212 | 958 | ~366 | 1 Hz | ×366 |

**Comparison with GTA and DuoGAT on SWaT / WADI** (official public implementations, same measurement protocol):

| Dataset | Method | Params | MACs/inf | RPi4-class est. (inf/s) | Margin |
|---|---|---|---|---|---|
| SWaT | GTA | 832,407 | 44,739,040 | ~35.2 | ×35.2 |
| SWaT | DuoGAT | 170,384 | 1,685,890 | ~169.7 | ×169.7 |
| SWaT | **Ours** | **91,221** | **278,656** | **~374** | **×374** |
| WADI | GTA | 1,086,716 | 90,784,768 | ~24.2 | ×24.2 |
| WADI | DuoGAT | 224,264 | 412,382,350 | ~15.6 | ×15.6 |
| WADI | **Ours** | **169,613** | **466,816** | **~366** | **×366** |

DuoGAT's MACs/inference on WADI (412.4 MMACs) is driven by its window size of 50 timesteps combined with the quadratic cost of its attention mechanism over node pairs, compared with 466.8 KMACs for the proposed model, whose cost scales with window size and feature count rather than their product or square.

Runtime memory on Cortex-A72 (full Python/PyTorch process) is ~257–258 MB for our model, well within the 1 GB RAM of the Raspberry Pi 4 B; see `bench_duogat.py` / `bench_gta.py` for the corresponding measurements on the baselines.

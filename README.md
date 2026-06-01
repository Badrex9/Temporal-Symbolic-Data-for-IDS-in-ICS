# Temporal-Symbolic Data Representation for Intrusion Detection in Industrial Control Systems

> **Enzo Zamaï, David Espes, Audrey C. Therrien, Catherine Dezan**
> Université de Bretagne Occidentale, Brest, France — Université de Sherbrooke, Canada
> *Computer Networks* (under review)

---

## Overview

This repository contains the official implementation of the temporal-symbolic IDS framework presented in the paper. The approach transforms raw system observations (network packets for IT environments, sensor snapshots for OT environments) into structured temporal matrices and processes them with a compact 1D convolutional architecture equipped with a **mask-guided variable-window pooling** mechanism that learns, end-to-end and per sample, the effective temporal horizon required for anomaly detection.

The same architectural principle operates across both ICS layers within a single framework:
- **IT layer** — supervised multiclass classification on CICIDS2017 network flows
- **OT layer** — unsupervised next-step prediction on SWaT and WADI sensor time series

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
├── benchmark_server.py           # Latency benchmark — GPU + CPU (training server)
├── benchmark_arm.py              # Latency benchmark — ARM Cortex-A72 (AWS Graviton2)
│
├── cicids_pooling/               # Saved CICIDS2017 models (.pth)
├── swat_unsup_models/            # Saved SWaT models (.pth)
├── wadi_unsup_models/            # Saved WADI models (.pth)
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

Tested with Python 3.10, PyTorch 2.11 (CUDA 13.0). No other dependencies are required.

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

**ARM Cortex-A72 (AWS Graviton2 / Raspberry Pi 4):**
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

**SWaT / WADI** — unsupervised protocol: models are trained exclusively on the normal operation period and evaluated on the attack period. Anomaly scores are derived from per-feature squared prediction errors (top-10% features). Detection threshold set at the 99th percentile of training scores. Performance is reported using the point-adjust protocol (F1_PA), consistent with standard practice in the ICS anomaly detection literature.

---

## Results

**CICIDS2017** (weighted F1, 7 retained classes):

| Method | F1 (%) | Params |
|---|---|---|
| Han et al. — Transformer | 97.83 | — |
| Sun et al. — CNN+LSTM | — | — |
| **Ours — maskguided, d=10** | **99.77** | **74k** |

**SWaT and WADI** (F1_PA, point-adjust protocol):

| Method | SWaT F1_PA | WADI F1_PA | Params |
|---|---|---|---|
| OmniAnomaly | 0.78 | 0.23 | — |
| USAD | 0.85 | 0.43 | 3.9M |
| GDN | 0.81 | 0.57 | 5k–20k |
| DE-CNN | 0.87 | 0.72 | 35k–283k |
| GTA | 0.91 | 0.84 | — |
| DuoGAT | 0.9366 | 0.7380 | 170k–224k |
| **Ours — maskguided** | **0.9153** | **0.8257** | **91–170k** |

---

## Deployment

All configurations were benchmarked at batch size 1 on a server workstation (Intel Core Ultra 7 265, NVIDIA RTX 4000 SFF Ada) and on an AWS Graviton2 instance (`t4g.micro`, ARM Cortex-A72 @ 2.5 GHz), which is microarchitecturally identical to the Broadcom BCM2711 of the Raspberry Pi 4. RPi4 throughput estimates are obtained by scaling Graviton2 measurements by the frequency ratio ×0.72 (2.5 GHz → 1.8 GHz).

| Dataset | GPU (inf/s) | CPU (inf/s) | RPi4 est. (inf/s) | Constraint | Margin |
|---|---|---|---|---|---|
| CICIDS2017 | 1,332 | 1,356 | ~420 | 25 flows/s | ×16.8 |
| SWaT | 1,073 | 1,144 | ~374 | 1 Hz | ×374 |
| WADI | 1,212 | 958 | ~366 | 1 Hz | ×366 |

Runtime memory on Cortex-A72 (full Python/PyTorch process): ~257–258 MB, within the 1 GB RAM of the Raspberry Pi 4 B.

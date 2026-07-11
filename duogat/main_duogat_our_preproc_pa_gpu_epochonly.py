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
from training import Trainer as BaseTrainer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import time

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# ==========================================================
# Load OUR normal-only preprocessing
#
# Expected files:
#   X_train.npy
#   X_test.npy
#   y_test.npy
#
# Put directories at the DuoGAT repo root:
#   ./preprocessed_swat_unsup/
#   ./preprocessed_wadi_unsup/
#
# Or pass --our_data_dir manually.
# ==========================================================
def _progress_iter(iterable, total=None, desc="progress"):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)
    return iterable


def _log_step(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_npy_with_log(path, dtype, name):
    t0 = time.time()
    _log_step(f"Loading {name}: {path}")
    arr = np.load(path)
    _log_step(f"Loaded {name}: shape={arr.shape}, dtype={arr.dtype}, time={time.time() - t0:.1f}s")
    if arr.dtype != dtype:
        t1 = time.time()
        _log_step(f"Casting {name} to {dtype}")
        arr = arr.astype(dtype, copy=False)
        _log_step(f"Cast done for {name}: dtype={arr.dtype}, time={time.time() - t1:.1f}s")
    return arr


# ==========================================================
# Load OUR normal-only preprocessing
#
# Expected files:
#   X_train.npy
#   X_test.npy
#   y_test.npy
#
# Put directories at the DuoGAT repo root:
#   ./preprocessed_swat_unsup/
#   ./preprocessed_wadi_unsup/
#
# Or pass --our_data_dir manually.
# ==========================================================
def load_our_preprocessed(dataset: str, dif_n: int = 1, our_data_dir: str = None):
    d = dataset.lower()

    if our_data_dir is None:
        if d in ["swat", "swat_a1_a2"]:
            our_data_dir = "./preprocessed_swat_unsup/"
        elif d == "wadi":
            our_data_dir = "./preprocessed_wadi_unsup/"
        else:
            raise ValueError(
                f"Unknown dataset '{dataset}'. "
                "Use --our_data_dir to provide the preprocessing directory."
            )

    x_train_path = os.path.join(our_data_dir, "X_train.npy")
    x_test_path = os.path.join(our_data_dir, "X_test.npy")
    y_test_path = os.path.join(our_data_dir, "y_test.npy")

    for path in [x_train_path, x_test_path, y_test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    _log_step("Starting data loading")
    x_train = _load_npy_with_log(x_train_path, np.float32, "X_train")
    x_test = _load_npy_with_log(x_test_path, np.float32, "X_test")
    y_test = _load_npy_with_log(y_test_path, np.uint8, "y_test")

    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError(f"Expected 2D arrays. Got x_train={x_train.shape}, x_test={x_test.shape}")
    if len(x_test) != len(y_test):
        raise ValueError(f"x_test and y_test have different lengths: {len(x_test)} vs {len(y_test)}")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError(f"Feature mismatch: train={x_train.shape[1]}, test={x_test.shape[1]}")

    def make_diff(X, dif_n, name):
        t0 = time.time()
        _log_step(f"Computing diff for {name} with dif_n={dif_n}")
        X_diff = np.zeros_like(X, dtype=np.float32)
        if dif_n > 0:
            X_diff[dif_n:] = X[dif_n:] - X[:-dif_n]
        _log_step(f"Diff done for {name}: shape={X_diff.shape}, time={time.time() - t0:.1f}s")
        return X_diff

    dif_x_train = make_diff(x_train, dif_n, "X_train")
    dif_x_test = make_diff(x_test, dif_n, "X_test")

    print("=" * 80, flush=True)
    print("Loaded OUR preprocessed data", flush=True)
    print("=" * 80, flush=True)
    print("data_dir:", our_data_dir, flush=True)
    print("x_train:", x_train.shape, flush=True)
    print("x_test :", x_test.shape, flush=True)
    print("y_test :", y_test.shape, {0: int((y_test == 0).sum()), 1: int((y_test == 1).sum())}, flush=True)
    print("dif_n  :", dif_n, flush=True)
    print("=" * 80, flush=True)

    return x_train, x_test, y_test, dif_x_train, dif_x_test

# ==========================================================
# Point-adjust protocol
# ==========================================================
def point_adjust_predictions(y_true, y_pred):
    """
    Point-adjust protocol:
    if at least one point inside a contiguous attack segment is detected,
    the whole segment is marked as detected.
    """
    y_true = np.asarray(y_true).astype(np.uint8)
    y_pred = np.asarray(y_pred).astype(np.uint8).copy()

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")

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


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def print_metrics(name, y_true, y_pred):
    m = compute_metrics(y_true, y_pred)
    print("=" * 80)
    print(name)
    print(f"Accuracy : {m['accuracy']:.6f}")
    print(f"Precision: {m['precision']:.6f}")
    print(f"Recall   : {m['recall']:.6f}")
    print(f"F1       : {m['f1']:.6f}")
    print("=" * 80)
    return m


def save_metrics_json(save_path, metrics_dict, filename="point_adjust_metrics.json"):
    path = os.path.join(save_path, filename)
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print("Saved metrics:", path)


# ==========================================================
# Try to extract predictions from Predictor return or saved files
# ==========================================================
def _as_binary_vector(arr, expected_len):
    arr = np.asarray(arr)

    # Flatten common shapes: [N], [N,1], [1,N]
    arr = np.squeeze(arr)

    if arr.ndim != 1:
        return None

    if len(arr) != expected_len:
        return None

    # If probabilities/scores, threshold at 0.5 only if values are in [0, 1].
    # Otherwise reject, because anomaly scores need their own threshold.
    unique = np.unique(arr[~np.isnan(arr)]) if np.issubdtype(arr.dtype, np.number) else np.unique(arr)

    if set(unique.tolist()).issubset({0, 1, 0.0, 1.0, False, True}):
        return arr.astype(np.uint8)

    if np.issubdtype(arr.dtype, np.number) and np.nanmin(arr) >= 0.0 and np.nanmax(arr) <= 1.0:
        return (arr >= 0.5).astype(np.uint8)

    return None


def extract_binary_prediction(result, expected_len):
    """
    Tries to extract a binary anomaly vector from Predictor.predict_anomalies output.

    Supports:
      - dict with y_pred/pred/anomaly_pred/etc.
      - tuple/list containing one vector with expected_len
      - direct ndarray/list with expected_len
    """
    if result is None:
        return None

    possible_keys = [
        "y_pred", "pred", "preds", "prediction", "predictions",
        "anomaly_pred", "anomaly_preds", "anomalies",
        "test_pred", "test_preds", "test_predictions",
    ]

    if isinstance(result, dict):
        for key in possible_keys:
            if key in result:
                pred = _as_binary_vector(result[key], expected_len)
                if pred is not None:
                    print(f"Using prediction returned by Predictor under key '{key}'")
                    return pred

        # Try every dict value as fallback.
        for key, value in result.items():
            pred = _as_binary_vector(value, expected_len)
            if pred is not None:
                print(f"Using prediction returned by Predictor under fallback key '{key}'")
                return pred

    if isinstance(result, (tuple, list)):
        for idx, value in enumerate(result):
            pred = _as_binary_vector(value, expected_len)
            if pred is not None:
                print(f"Using prediction returned by Predictor tuple/list index {idx}")
                return pred

    pred = _as_binary_vector(result, expected_len)
    if pred is not None:
        print("Using prediction returned directly by Predictor.")
        return pred

    return None


def try_load_prediction_from_save_path(save_path, expected_len):
    """
    Some prediction scripts save binary labels instead of returning them.
    This function tries common filenames.
    """
    names = [
        "y_pred.npy",
        "pred.npy",
        "preds.npy",
        "prediction.npy",
        "predictions.npy",
        "anomaly_pred.npy",
        "anomaly_preds.npy",
        "anomalies.npy",
        "test_pred.npy",
        "test_preds.npy",
        "test_predictions.npy",
    ]

    for name in names:
        path = os.path.join(save_path, name)
        if os.path.exists(path):
            arr = np.load(path, allow_pickle=True)
            pred = _as_binary_vector(arr, expected_len)
            if pred is not None:
                print(f"Loaded binary prediction from {path}")
                return pred

    return None


# ==========================================================
# Parameter count
# ==========================================================
def print_model_complexity(model, dataset, n_features, window_size, out_dim, args):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 80)
    print("MODEL COMPLEXITY")
    print(f"Dataset: {dataset}")
    print(f"n_features: {n_features}")
    print(f"window_size: {window_size}")
    print(f"out_dim: {out_dim}")
    print(f"gru_n_layers: {args.gru_n_layers}")
    print(f"gru_hid_dim: {args.gru_hid_dim}")
    print(f"fc_n_layers: {args.fc_n_layers}")
    print(f"fc_hid_dim: {args.fc_hid_dim}")
    print(f"dropout: {args.dropout}")
    print(f"alpha: {args.alpha}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size FP32: {trainable_params * 4 / (1024 ** 2):.3f} MB")
    print(f"Model size INT8: {trainable_params * 1 / (1024 ** 2):.3f} MB")
    print("=" * 80)

    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "model_size_fp32_mb": float(trainable_params * 4 / (1024 ** 2)),
        "model_size_int8_mb": float(trainable_params * 1 / (1024 ** 2)),
    }


# ==========================================================
# Trainer with visible progress bars
# ==========================================================
class ProgressTrainer(BaseTrainer):
    def fit(self, dif_train_loader, dif_val_loader, train_loader, val_loader=None):
        init_train_loss = None
        if val_loader is not None:
            _log_step("Initial validation evaluation")
            init_val_loss = self.evaluate(val_loader, dif_val_loader, desc="init-val-eval")
            print(f"Init total val loss: {init_val_loss[0]:.5f}", flush=True)

        print(f"Training model epoch-by-epoch for {self.n_epochs} epochs on {self.device}..", flush=True)
        train_start = time.time()
        min_loss = 1e+8
        stop_improve_count = 0

        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            self.model.train()
            forecast_b_losses = []

            total_batches = min(len(train_loader), len(dif_train_loader))
            batch_iter = zip(train_loader, dif_train_loader)
            if tqdm is not None:
                batch_iter = tqdm(
                    batch_iter,
                    total=total_batches,
                    desc=f"epoch {epoch + 1}/{self.n_epochs}",
                    dynamic_ncols=True,
                    leave=False,
                )

            for batch_idx, ((x, y), (dif_x, dif_y)) in enumerate(batch_iter, start=1):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                dif_x = dif_x.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                preds = self.model(x, dif_x)

                if self.target_dims is not None:
                    x = x[:, :, self.target_dims]
                    y = y[:, :, self.target_dims].squeeze(-1)

                if preds.ndim == 3:
                    preds = preds.squeeze(1)
                if y.ndim == 3:
                    y = y.squeeze(1)

                criterion = nn.MSELoss()
                forecast_loss = torch.sqrt(criterion(y, preds))
                loss = forecast_loss
                loss.backward()
                self.optimizer.step()

                loss_value = forecast_loss.item()
                forecast_b_losses.append(loss_value)

                if tqdm is not None:
                    batch_iter.set_postfix(loss=f"{loss_value:.5f}")
                elif batch_idx == 1 or batch_idx % max(1, total_batches // 20) == 0 or batch_idx == total_batches:
                    print(
                        f"Epoch {epoch + 1}/{self.n_epochs} "
                        f"batch {batch_idx}/{total_batches} "
                        f"loss={loss_value:.5f}",
                        flush=True,
                    )

            forecast_b_losses = np.array(forecast_b_losses)
            forecast_epoch_loss = np.sqrt((forecast_b_losses ** 2).mean())
            total_epoch_loss = forecast_epoch_loss

            self.losses["train_forecast"].append(forecast_epoch_loss)
            self.losses["train_total"].append(total_epoch_loss)

            forecast_val_loss, total_val_loss = "NA", "NA"
            if val_loader is not None:
                forecast_val_loss, total_val_loss = self.evaluate(
                    val_loader,
                    dif_val_loader,
                    desc=f"val {epoch + 1}/{self.n_epochs}",
                )
                self.losses["val_forecast"].append(forecast_val_loss)
                self.losses["val_total"].append(total_val_loss)

                if total_val_loss < min_loss:
                    self.save("model.pt")
                    min_loss = total_val_loss
                    stop_improve_count = 0
                else:
                    stop_improve_count += 1
                    if stop_improve_count >= self.early_stopping:
                        print("early stop!", flush=True)
                        break

            if self.log_tensorboard:
                self.write_loss(epoch)

            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)

            s = (
                f"[Epoch {epoch + 1}/{self.n_epochs}] "
                f"forecast_loss={forecast_epoch_loss:.5f}, "
                f"total_loss={total_epoch_loss:.5f}"
            )
            if val_loader is not None:
                s += (
                    f" ---- val_forecast_loss={forecast_val_loss:.5f}, "
                    f"val_total_loss={total_val_loss:.5f}"
                )
            s += f" [{epoch_time:.1f}s]"
            print(s, flush=True)

            if val_loader is not None:
                self.scheduler.step(total_val_loss)

        if val_loader is None:
            self.save("model.pt")

        train_time = int(time.time() - train_start)
        if self.log_tensorboard:
            self.writer.add_text("total_train_time", str(train_time))
        print(f"-- Training done in {train_time}s.", flush=True)
        return train_time

    def evaluate(self, data_loader, dif_loader, desc="eval"):
        self.model.eval()
        forecast_losses = []
        total_batches = min(len(data_loader), len(dif_loader))
        eval_iter = zip(data_loader, dif_loader)
        if tqdm is not None:
            eval_iter = tqdm(eval_iter, total=total_batches, desc=desc, dynamic_ncols=True, leave=False)

        with torch.no_grad():
            for batch_idx, ((x, y), (dif_x, dif_y)) in enumerate(eval_iter, start=1):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                dif_x = dif_x.to(self.device, non_blocking=True)

                preds = self.model(x, dif_x)
                if self.target_dims is not None:
                    x = x[:, :, self.target_dims]
                    y = y[:, :, self.target_dims].squeeze(-1)

                if preds.ndim == 3:
                    preds = preds.squeeze(1)
                if y.ndim == 3:
                    y = y.squeeze(1)

                criterion = nn.MSELoss()
                forecast_loss = torch.sqrt(criterion(y, preds))
                loss_value = forecast_loss.item()
                forecast_losses.append(loss_value)

                if tqdm is not None:
                    eval_iter.set_postfix(loss=f"{loss_value:.5f}")
                elif batch_idx == 1 or batch_idx % max(1, total_batches // 10) == 0 or batch_idx == total_batches:
                    print(f"{desc}: batch {batch_idx}/{total_batches} loss={loss_value:.5f}", flush=True)

        forecast_losses = np.array(forecast_losses)
        forecast_loss = np.sqrt((forecast_losses ** 2).mean())
        total_loss = forecast_loss
        return forecast_loss, total_loss


if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    run_id = datetime.now().strftime("%d%m%Y_%H%M%S")

    parser = get_parser()

    # Extra arguments for our preprocessing.
    # Use names unlikely to collide with the original repo.
    parser.add_argument(
        "--our_data_dir",
        type=str,
        default=None,
        help="Directory containing X_train.npy, X_test.npy, y_test.npy. "
             "If omitted, uses ./preprocessed_swat_unsup/ or ./preprocessed_wadi_unsup/."
    )

    parser.add_argument(
        "--skip_predictor_pa",
        action="store_true",
        help="Do not try to compute point-adjust from Predictor outputs."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Device to use. Default: cuda. Use cpu to force CPU, auto to use CUDA if available."
    )

    args = parser.parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "You requested --device cuda, but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch or run with --device cpu."
            )
        use_cuda = True
    elif args.device == "auto":
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = False

    # Keep compatibility with the original DuoGAT Trainer, which expects use_cuda.
    args.use_cuda = use_cuda

    print("=" * 80)
    print("DEVICE CONFIG")
    print("Requested device:", args.device)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("Using CUDA:", use_cuda)
    if use_cuda:
        print("CUDA device:", torch.cuda.get_device_name(0))
    print("=" * 80)

    dataset = args.dataset
    window_size = args.lookback
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    # Set above from --device and copied into args.use_cuda for Trainer compatibility.
    use_cuda = args.use_cuda
    print_every = args.print_every
    log_tensorboard = args.log_tensorboard
    args_summary = str(args.__dict__)
    dif_n = args.dif_n

    print(args_summary)

    output_path = f"output/{dataset}"

    # ======================================================
    # OUR DATA LOADER instead of data_get(dataset, dif_n)
    # ======================================================
    x_train, x_test, y_test, dif_x_train, dif_x_test = load_our_preprocessed(
        dataset=dataset,
        dif_n=dif_n,
        our_data_dir=args.our_data_dir
    )

    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    log_dir = f"{output_path}/logs"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    save_path = f"{output_path}/{run_id}"
    print("save_path:", save_path)
    os.makedirs(save_path)

    _log_step("Converting numpy arrays to torch tensors")
    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()

    dif_x_train = torch.from_numpy(dif_x_train).float()
    dif_x_test = torch.from_numpy(dif_x_test).float()
    _log_step("Tensor conversion done")

    n_features = x_train.shape[1]

    target_dims = get_target_dims(dataset)

    if target_dims is None:
        out_dim = n_features
    elif type(target_dims) == int:
        out_dim = 1
    else:
        out_dim = len(target_dims)

    _log_step("Creating sliding-window datasets")
    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)
    _log_step(f"Sliding-window datasets ready: train={len(train_dataset)}, test={len(test_dataset)}")

    _log_step("Creating data loaders")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset,
        batch_size,
        val_split,
        shuffle_dataset,
        test_dataset=test_dataset
    )
    _log_step(f"Data loaders ready: train_batches={len(train_loader)}, val_batches={len(val_loader) if val_loader is not None else 0}, test_batches={len(test_loader)}")

    _log_step("Creating diff sliding-window datasets")
    dif_train_dataset = SlidingWindowDataset(dif_x_train, window_size, target_dims)
    dif_test_dataset = SlidingWindowDataset(dif_x_test, window_size, target_dims)
    _log_step(f"Diff sliding-window datasets ready: train={len(dif_train_dataset)}, test={len(dif_test_dataset)}")

    _log_step("Creating diff data loaders")
    dif_train_loader, dif_val_loader, dif_test_loader = create_data_loaders(
        dif_train_dataset,
        batch_size,
        val_split,
        shuffle_dataset,
        test_dataset=dif_test_dataset
    )
    _log_step(f"Diff data loaders ready: train_batches={len(dif_train_loader)}, val_batches={len(dif_val_loader) if dif_val_loader is not None else 0}, test_batches={len(dif_test_loader)}")

    model = DuoGAT(
        n_features,
        window_size,
        out_dim,
        batch_size=args.bs,
        gru_n_layers=args.gru_n_layers,
        gru_hid_dim=args.gru_hid_dim,
        forecast_n_layers=args.fc_n_layers,
        forecast_hid_dim=args.fc_hid_dim,
        dropout=args.dropout,
        alpha=args.alpha
    )

    complexity = print_model_complexity(
        model=model,
        dataset=dataset,
        n_features=n_features,
        window_size=window_size,
        out_dim=out_dim,
        args=args
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=6,
        factor=0.8
    )

    forecast_criterion = nn.MSELoss()
    early_stopping = 10

    trainer = ProgressTrainer(
        model,
        optimizer,
        scheduler,
        early_stopping,
        window_size,
        n_features,
        target_dims,
        n_epochs,
        batch_size,
        init_lr,
        forecast_criterion,
        use_cuda,
        save_path,
        log_dir,
        print_every,
        log_tensorboard,
        args_summary
    )

    train_time = trainer.fit(
        dif_train_loader,
        dif_val_loader,
        train_loader,
        val_loader
    )

    # Check test loss
    test_loss = trainer.evaluate(test_loader, dif_test_loader)
    print(f"Test forecast loss: {test_loss[0]:.5f}")
    print(f"Test total loss: {test_loss[1]:.5f}")

    trainer.load(f"{save_path}/model.pt")

    prediction_args = {
        "dataset": dataset,
        "target_dims": target_dims,
        "save_path": save_path
    }

    best_model = trainer.model

    predictor = Predictor(
        best_model,
        window_size,
        n_features,
        prediction_args,
    )

    # Important alignment: DuoGAT produces predictions only after lookback samples.
    label = y_test[window_size:] if y_test is not None else None

    predictor_result = predictor.predict_anomalies(x_test, label, dif_x_test)

    # ======================================================
    # Point-adjust metrics, if Predictor exposes/saves binary predictions
    # ======================================================
    pa_metrics = None

    if not args.skip_predictor_pa and label is not None:
        y_pred = extract_binary_prediction(predictor_result, expected_len=len(label))

        if y_pred is None:
            y_pred = try_load_prediction_from_save_path(save_path, expected_len=len(label))

        if y_pred is not None:
            raw_metrics = print_metrics("Raw point-wise metrics", label, y_pred)

            y_pred_pa = point_adjust_predictions(label, y_pred)
            adjusted_metrics = print_metrics("Point-adjust metrics", label, y_pred_pa)

            pa_metrics = {
                "raw": raw_metrics,
                "point_adjust": adjusted_metrics,
            }

            np.save(os.path.join(save_path, "y_true_aligned.npy"), label.astype(np.uint8))
            np.save(os.path.join(save_path, "y_pred_raw.npy"), y_pred.astype(np.uint8))
            np.save(os.path.join(save_path, "y_pred_point_adjust.npy"), y_pred_pa.astype(np.uint8))

            save_metrics_json(save_path, pa_metrics)

        else:
            print("=" * 80)
            print("WARNING: could not compute point-adjust metrics from main.py alone.")
            print("Reason: Predictor.predict_anomalies did not return a binary prediction")
            print("and no common prediction .npy file was found in save_path.")
            print("")
            print("Fix: open prediction.py and make predict_anomalies return the final")
            print("binary anomaly vector, for example:")
            print("")
            print("    return y_pred")
            print("")
            print("or save it as:")
            print("")
            print("    np.save(os.path.join(self.save_path, 'y_pred.npy'), y_pred)")
            print("=" * 80)

    args.__dict__["train_time"] = train_time
    args.__dict__["complexity"] = complexity
    args.__dict__["point_adjust_metrics"] = pa_metrics

    # Save config
    args_path = f"{save_path}/config.txt"
    print("args_path", args_path)

    with open(args_path, "w") as f:
        print(args.__dict__)
        json.dump(args.__dict__, f, indent=2)

import os
import json
import pickle
import numpy as np
from eval_methods import *
from utils import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


class Predictor:
    def __init__(self, model, window_size, n_features, pred_args, summary_file_name="summary.txt"):
        self.model = model
        self.window_size = window_size
        self.n_features = n_features
        self.dataset = pred_args["dataset"]
        self.target_dims = pred_args["target_dims"]
        self.save_path = pred_args["save_path"]
        self.batch_size = 256
        self.use_cuda = True
        self.pred_args = pred_args
        self.summary_file_name = summary_file_name

    def get_score(self, values, dif_test):
        print("Predicting and calculating anomaly scores..")
        data = SlidingWindowDataset(values, self.window_size, self.target_dims)
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=False)
        device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"

        # differencing test
        dif_data = SlidingWindowDataset(dif_test, self.window_size, self.target_dims)
        dif_loader = torch.utils.data.DataLoader(dif_data, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        preds = []

        n_batches = min(len(loader), len(dif_loader))
        print(f"Inference batches: {n_batches}")

        with torch.no_grad():
            for batch_idx, ((x, y), (dif_x, dif_y)) in enumerate(zip(loader, dif_loader), start=1):
                x = x.to(device)
                dif_x = dif_x.to(device)

                y_hat = self.model(x, dif_x)
                preds.append(y_hat.detach().cpu().numpy())

                # Clean epoch-style progress, not tqdm spam.
                if batch_idx == 1 or batch_idx == n_batches or batch_idx % 200 == 0:
                    print(f"  inference batch {batch_idx}/{n_batches}")

        preds = np.concatenate(preds, axis=0)
        actual = values.detach().cpu().numpy()[self.window_size:]

        if self.target_dims is not None:
            actual = actual[:, self.target_dims]

        anomaly_scores = np.zeros_like(actual)
        df = pd.DataFrame()

        for i in range(preds.shape[1]):
            df[f"Forecast_{i}"] = preds[:, i]
            df[f"True_{i}"] = actual[:, i]
            a_score = np.sqrt((preds[:, i] - actual[:, i]) ** 2)
            anomaly_scores[:, i] = a_score
            df[f"A_Score_{i}"] = a_score

        # Calculate anomaly scores.
        scaler = StandardScaler()
        scaled_anomaly_scores = scaler.fit_transform(anomaly_scores)
        scaled_anomaly_scores_max = np.max(scaled_anomaly_scores, 1)
        df["scaled_max"] = scaled_anomaly_scores_max

        return df

    def predict_anomalies(self, test, true_anomalies, dif_test, save_output=True):
        test_pred_df = self.get_score(test, dif_test)
        test_anomaly_scores_smax = test_pred_df["scaled_max"].values.astype(np.float32)
        test_pred_df["scaled_max"] = test_anomaly_scores_smax

        if true_anomalies is not None:
            y_true = np.asarray(true_anomalies).astype(np.uint8)
            if len(y_true) != len(test_anomaly_scores_smax):
                raise ValueError(
                    f"Length mismatch for metrics: y_true={len(y_true)}, "
                    f"scores={len(test_anomaly_scores_smax)}"
                )

            # AUC metrics from continuous scores, before thresholding.
            auc_metrics = {
                "roc_auc": float(roc_auc_score(y_true, test_anomaly_scores_smax)),
                "pr_auc": float(average_precision_score(y_true, test_anomaly_scores_smax)),
            }

            print("=" * 80)
            print("AUC metrics from continuous anomaly scores")
            print(f"ROC-AUC: {auc_metrics['roc_auc']:.6f}")
            print(f"PR-AUC : {auc_metrics['pr_auc']:.6f}")
            print("=" * 80)

            print("Finding best f1-score by searching for threshold..")
            bf_point_smax = bf_search_point(
                test_anomaly_scores_smax,
                y_true,
                start=np.quantile(test_anomaly_scores_smax, 0.80),
                end=np.quantile(test_anomaly_scores_smax, 1),
                step_num=1000,
                verbose=False,
            )
        else:
            y_true = None
            auc_metrics = {}
            bf_point_smax = {}

        print(f"Results using best f1 score search:\n {bf_point_smax}")

        for k, v in bf_point_smax.items():
            bf_point_smax[k] = float(v)

        # Save scores, labels, AUC, and thresholded predictions.
        if save_output:
            os.makedirs(self.save_path, exist_ok=True)

            summary = {
                "bf_point_max": bf_point_smax,
                "auc_metrics": auc_metrics,
            }

            with open(os.path.join(self.save_path, self.summary_file_name), "wb") as f:
                pickle.dump(summary, f)

            with open(os.path.join(self.save_path, "metrics_auc_f1.json"), "w") as f:
                json.dump(summary, f, indent=2)

            np.save(os.path.join(self.save_path, "anomaly_scores.npy"), test_anomaly_scores_smax)

            if y_true is not None:
                np.save(os.path.join(self.save_path, "y_true.npy"), y_true)

            # Save binary prediction using the best threshold if available.
            if "threshold" in bf_point_smax:
                threshold = float(bf_point_smax["threshold"])
                y_pred = (test_anomaly_scores_smax > threshold).astype(np.uint8)
                np.save(os.path.join(self.save_path, "y_pred.npy"), y_pred)
            else:
                y_pred = None
        else:
            y_pred = None

        print("-- Done.")

        # Return the binary vector so main.py can compute point-adjust metrics too.
        return y_pred

"""
bench_duogat_ram.py
====================
Mesure RAM isolee par dataset pour DuoGAT (code officiel).
Contrairement a bench_duogat_official.py qui teste SWaT ET WADI dans le
meme process (biaisant /usr/bin/time -v qui ne capture que le PIC sur
toute la duree), ce script ne teste qu'UN SEUL dataset par execution.

Usage :
  python bench_duogat_ram.py --dataset swat
  python bench_duogat_ram.py --dataset wadi

Pour la mesure memoire isolee :
  /usr/bin/time -v python bench_duogat_ram.py --dataset swat 2>&1 | grep "Maximum resident"
  /usr/bin/time -v python bench_duogat_ram.py --dataset wadi 2>&1 | grep "Maximum resident"
"""

import sys, os, argparse, warnings
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DuoGAT"))
from duogat import DuoGAT  # noqa: E402

CONFIGS = {
    "swat": dict(n_features=51,  window_size=5),
    "wadi": dict(n_features=123, window_size=50),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["swat", "wadi"])
    args = parser.parse_args()

    cfg = CONFIGS[args.dataset]
    model = DuoGAT(
        n_features=cfg["n_features"],
        window_size=cfg["window_size"],
        out_dim=cfg["n_features"],
        batch_size=1,
        gru_n_layers=1,
        gru_hid_dim=150,
        forecast_n_layers=1,
        forecast_hid_dim=150,
        dropout=0.0,
        alpha=0.2,
    )
    model.eval()

    x     = torch.randn(1, cfg["window_size"], cfg["n_features"])
    dif_x = torch.randn(1, cfg["window_size"], cfg["n_features"])

    with torch.no_grad():
        for _ in range(100):
            _ = model(x, dif_x)

    print(f"done -- {args.dataset}")


if __name__ == "__main__":
    main()

"""
bench_gta_ram.py
=================
Mesure RAM isolee par dataset pour GTA (code officiel).
Contrairement a bench_gta_gpu.py qui teste SWaT ET WADI dans le meme
process (biaisant /usr/bin/time -v qui ne capture que le PIC sur toute
la duree), ce script ne teste qu'UN SEUL dataset par execution.

PREREQUIS : identiques a bench_gta.py (repo clone + torch_geometric +
patch sed du bug gc_modules -> gc_module).

Usage :
  python bench_gta_ram.py --dataset swat
  python bench_gta_ram.py --dataset wadi

Pour la mesure memoire isolee :
  /usr/bin/time -v python bench_gta_ram.py --dataset swat 2>&1 | grep "Maximum resident"
  /usr/bin/time -v python bench_gta_ram.py --dataset wadi 2>&1 | grep "Maximum resident"
"""

import sys, os, argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GTA"))
from models.gta import GTA  # noqa: E402

CONFIGS = {
    "swat": dict(num_nodes=51),
    "wadi": dict(num_nodes=112),
}
SEQ_LEN, LABEL_LEN, PRED_LEN = 60, 30, 24


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["swat", "wadi"])
    args = parser.parse_args()

    cfg = CONFIGS[args.dataset]
    device = torch.device("cpu")

    model = GTA(
        num_nodes=cfg["num_nodes"], seq_len=SEQ_LEN, label_len=LABEL_LEN, out_len=PRED_LEN,
        num_levels=3, factor=5, d_model=128, n_heads=8,
        e_layers=3, d_layers=2, d_ff=128, dropout=0.05,
        attn="prob", embed="fixed", data="SWaT", activation="gelu",
        device=device,
    ).double().to(device)
    model.eval()

    x      = torch.randn(1, SEQ_LEN, cfg["num_nodes"], dtype=torch.double)
    y      = torch.randn(1, LABEL_LEN + PRED_LEN, cfg["num_nodes"], dtype=torch.double)
    x_mark = torch.randn(1, SEQ_LEN, 4, dtype=torch.double)
    y_mark = torch.randn(1, LABEL_LEN + PRED_LEN, 4, dtype=torch.double)

    with torch.no_grad():
        for _ in range(20):
            _ = model(x, y, x_mark, y_mark)

    print(f"done -- {args.dataset}")


if __name__ == "__main__":
    main()

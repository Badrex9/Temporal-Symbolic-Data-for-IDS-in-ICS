"""
bench_gta_gpu.py
=================
Benchmark complet GTA (Chen et al., IEEE IoT-J 2021) sur GPU + CPU serveur.
Poids aleatoires -- pas d'entrainement, pas de perfs, uniquement le cout
d'inference et l'empreinte memoire.

PREREQUIS (identiques a bench_gta.py) :
  git clone https://github.com/zackchen-lb/GTA.git
  pip install torch_geometric --break-system-packages
  sed -i 's/self\.gc_modules\[0\]/self.gc_module/' GTA/models/gta.py

Lancer depuis le dossier PARENT du dossier GTA/ :
  python bench_gta_gpu.py
"""

import sys, os, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GTA"))
from models.gta import GTA  # noqa: E402

CONFIGS = [
    dict(name="SWaT", num_nodes=51,  constraint_hz=1, constraint_label="1 Hz"),
    dict(name="WADI", num_nodes=112, constraint_hz=1, constraint_label="1 Hz"),
]
SEQ_LEN, LABEL_LEN, PRED_LEN = 60, 30, 24


def build_model(num_nodes, device):
    return GTA(
        num_nodes=num_nodes, seq_len=SEQ_LEN, label_len=LABEL_LEN, out_len=PRED_LEN,
        num_levels=3, factor=5, d_model=128, n_heads=8,
        e_layers=3, d_layers=2, d_ff=128, dropout=0.05,
        attn="prob", embed="fixed", data="SWaT", activation="gelu",
        device=device,
    ).double().to(device)


def make_inputs(num_nodes, device):
    x      = torch.randn(1, SEQ_LEN, num_nodes, dtype=torch.double, device=device)
    y      = torch.randn(1, LABEL_LEN + PRED_LEN, num_nodes, dtype=torch.double, device=device)
    x_mark = torch.randn(1, SEQ_LEN, 4, dtype=torch.double, device=device)
    y_mark = torch.randn(1, LABEL_LEN + PRED_LEN, 4, dtype=torch.double, device=device)
    return x, y, x_mark, y_mark


def benchmark_cpu(model, num_nodes, n_warmup=20, n_runs=200):
    model.eval()
    x, y, x_mark, y_mark = make_inputs(num_nodes, torch.device("cpu"))
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x, y, x_mark, y_mark)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(x, y, x_mark, y_mark)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e6)
    times = np.array(times)
    return dict(median_us=np.median(times), p95_us=np.percentile(times, 95),
                p99_us=np.percentile(times, 99), throughput=1e6 / np.median(times))


def benchmark_gpu(model_cpu_state, num_nodes, n_warmup=50, n_runs=500):
    device = torch.device("cuda")
    model = build_model(num_nodes, device)
    model.load_state_dict(model_cpu_state)
    model.eval().to(device)
    x, y, x_mark, y_mark = make_inputs(num_nodes, device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x, y, x_mark, y_mark)
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(x, y, x_mark, y_mark)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000)  # ms -> us
    times = np.array(times)
    return dict(median_us=np.median(times), p95_us=np.percentile(times, 95),
                p99_us=np.percentile(times, 99), throughput=1e6 / np.median(times))


def main():
    has_gpu = torch.cuda.is_available()
    print(f"\n{'='*65}")
    print(f"  GTA -- Benchmark serveur (GPU + CPU)  |  batch=1")
    print(f"  PyTorch {torch.__version__}")
    print(f"  GPU : {torch.cuda.get_device_name(0) if has_gpu else 'N/A'}")
    print(f"{'='*65}\n")

    for cfg in CONFIGS:
        print(f"-- GTA / {cfg['name']}  (num_nodes={cfg['num_nodes']}) --")

        model_cpu = build_model(cfg["num_nodes"], torch.device("cpu"))
        n_params = sum(p.numel() for p in model_cpu.parameters() if p.requires_grad)

        # Taille memoire (poids seuls -- double precision dans ce repo, donc f32/INT8
        # sont des tailles theoriques si le modele etait quantifie/reduit en precision)
        f32_mb  = n_params * 4  / (1024 ** 2)
        int8_mb = n_params * 1  / (1024 ** 2)

        rc = benchmark_cpu(model_cpu, cfg["num_nodes"])
        rg = benchmark_gpu(model_cpu.state_dict(), cfg["num_nodes"]) if has_gpu else None

        print(f"  Params        : {n_params:,}")
        print(f"  f32 (MB)      : {f32_mb:.2f}")
        print(f"  INT8 (MB)     : {int8_mb:.2f}")
        print(f"  CPU latence   : median={rc['median_us']:.1f} us  "
              f"p95={rc['p95_us']:.1f} us  p99={rc['p99_us']:.1f} us")
        print(f"  CPU throughput: {rc['throughput']:.2f} inf/s")
        if rg:
            print(f"  GPU latence   : median={rg['median_us']:.1f} us  "
                  f"p95={rg['p95_us']:.1f} us  p99={rg['p99_us']:.1f} us")
            print(f"  GPU throughput: {rg['throughput']:.2f} inf/s")
        print(f"  Contrainte    : {cfg['constraint_label']}")
        print(f"  Marge CPU     : x{rc['throughput']/cfg['constraint_hz']:.2f}")
        if rg:
            print(f"  Marge GPU     : x{rg['throughput']/cfg['constraint_hz']:.2f}")
        print()

        # MACs -- thop ne supporte pas les couches torch_geometric (message passing
        # custom AdaGCNConv), donc on tente et on retombe proprement sinon.
        try:
            from thop import profile
            x, y, x_mark, y_mark = make_inputs(cfg["num_nodes"], torch.device("cpu"))
            macs, _ = profile(model_cpu, inputs=(x, y, x_mark, y_mark), verbose=False)
            print(f"  MACs/inf      : {int(macs):,}  ({macs/1e6:.3f} MMACs)")
        except Exception as e:
            print(f"  MACs/inf      : indisponible (thop ne supporte pas "
                  f"les couches torch_geometric -- {type(e).__name__})")
        print()

if __name__ == "__main__":
    main()

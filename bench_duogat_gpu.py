"""
bench_duogat_official_gpu.py
=============================
Benchmark GPU + CPU serveur du VRAI code officiel DuoGAT
(ByeongtaePark/DuoGAT, CIKM'23) -- import direct de duogat.py + modules.py.
Poids aleatoires, pas d'entrainement, pas de perfs.

PREREQUIS :
  1. Cloner le repo officiel a cote de ce script :
       git clone https://github.com/ByeongtaePark/DuoGAT.git
  2. PATCH CPU/GPU-safe obligatoire -- le repo hardcode `.to('cuda')` dans
     modules.py (TemporalAttentionLayer.forward, 2 occurrences). Pour
     pouvoir tester CPU ET GPU avec le meme code, on route ce device
     dynamiquement plutot que de le retirer completement :
       sed -i "s/weight_matrix = weight_matrix.to('cuda')/weight_matrix = weight_matrix.to(x.device)/" DuoGAT/modules.py
       sed -i "s/weight_matrix2 = weight_matrix2.to('cuda')/weight_matrix2 = weight_matrix2.to(x.device)/" DuoGAT/modules.py
  3. pip install pandas --break-system-packages   (requis par utils.py)
  4. Lancer depuis le dossier PARENT du dossier DuoGAT/ :
       python bench_duogat_official_gpu.py

Config -- fenetre temporelle fidele au texte de l'article CIKM'23
(section 4.4): "sliding window size among {5, 50, 150, 30}" pour
{SWaT, WADI, SMAP, MSL}.
  SWaT : window_size = 5
  WADI : window_size = 50
Hyperparametres modele (args.py, valeurs par defaut, communes aux datasets):
  gru_hid_dim=150, fc_hid_dim=150, gru_n_layers=1, fc_n_layers=1
"""

import sys, os, time, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DuoGAT"))
from duogat import DuoGAT  # noqa: E402

try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

CONFIGS = [
    dict(name="SWaT", n_features=51,  window_size=5,  constraint_hz=1, constraint_label="1 Hz"),
    dict(name="WADI", n_features=123, window_size=50, constraint_hz=1, constraint_label="1 Hz"),
]


def build_model(cfg, batch_size=1):
    return DuoGAT(
        n_features=cfg["n_features"],
        window_size=cfg["window_size"],
        out_dim=cfg["n_features"],
        batch_size=batch_size,
        gru_n_layers=1,
        gru_hid_dim=150,
        forecast_n_layers=1,
        forecast_hid_dim=150,
        dropout=0.0,
        alpha=0.2,
    )


def make_inputs(cfg, device):
    x     = torch.randn(1, cfg["window_size"], cfg["n_features"], device=device)
    dif_x = torch.randn(1, cfg["window_size"], cfg["n_features"], device=device)
    return x, dif_x


def benchmark_cpu(model, cfg, n_warmup=50, n_runs=500):
    model.eval().cpu()
    x, dif_x = make_inputs(cfg, torch.device("cpu"))
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x, dif_x)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(x, dif_x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e6)
    times = np.array(times)
    return dict(median_us=np.median(times), p95_us=np.percentile(times, 95),
                p99_us=np.percentile(times, 99), throughput=1e6 / np.median(times))


def benchmark_gpu(state_dict, cfg, n_warmup=200, n_runs=2000):
    device = torch.device("cuda")
    model = build_model(cfg)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    x, dif_x = make_inputs(cfg, device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x, dif_x)
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(x, dif_x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000)  # ms -> us
    times = np.array(times)
    return dict(median_us=np.median(times), p95_us=np.percentile(times, 95),
                p99_us=np.percentile(times, 99), throughput=1e6 / np.median(times))


def main():
    has_gpu = torch.cuda.is_available()
    print(f"\n{'='*65}")
    print(f"  DuoGAT (code officiel) -- Benchmark serveur (GPU + CPU)")
    print(f"  batch=1  |  PyTorch {torch.__version__}")
    print(f"  GPU : {torch.cuda.get_device_name(0) if has_gpu else 'N/A'}")
    print(f"{'='*65}\n")

    for cfg in CONFIGS:
        print(f"-- DuoGAT / {cfg['name']}  "
              f"(n_features={cfg['n_features']}, window={cfg['window_size']}) --")

        model_cpu = build_model(cfg)
        n_params = sum(p.numel() for p in model_cpu.parameters() if p.requires_grad)
        f32_mb  = n_params * 4 / (1024 ** 2)
        int8_mb = n_params * 1 / (1024 ** 2)

        rc = benchmark_cpu(model_cpu, cfg)
        rg = benchmark_gpu(model_cpu.state_dict(), cfg) if has_gpu else None

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

        if HAS_THOP:
            try:
                x, dif_x = make_inputs(cfg, torch.device("cpu"))
                macs, _ = profile(model_cpu, inputs=(x, dif_x), verbose=False)
                print(f"  MACs/inf      : {int(macs):,}  ({macs/1e6:.3f} MMACs)")
            except Exception as e:
                print(f"  MACs/inf      : echec ({type(e).__name__})")
        else:
            print(f"  MACs/inf      : thop non installe")
        print()

    print(f"{'='*65}")
    print("  Code officiel ByeongtaePark/DuoGAT, import direct (duogat.py")
    print("  + modules.py). Patch: .to('cuda') -> .to(x.device) dans")
    print("  modules.py pour compatibilite CPU/GPU sans autre changement")
    print("  de logique. window_size selon texte article CIKM'23 sec. 4.4.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()

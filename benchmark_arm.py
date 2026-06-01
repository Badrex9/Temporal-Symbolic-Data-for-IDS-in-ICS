"""
benchmark_arm.py
================
Mesure de latence + MACs sur ARM Cortex-A72 (AWS Graviton2 / RPi4-class).
Modèles : CICIDS2017 (maskguided, d=10), SWaT (maskguided, d=10), WADI (maskguided, d=5)
CPU uniquement, batch=1.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

# ================================================================
# Pooling modules
# ================================================================
class TemporalAvgPool(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((None, K))
    def forward(self, x):
        return self.pool(x)

class TemporalWeightedPool(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K
    def forward(self, x):
        B, C, H, W = x.shape
        if W == 1:
            return x
        raw = torch.tensor([1./(i+1) for i in range(W)], device=x.device, dtype=x.dtype)
        w = (raw / raw.sum()).view(1,1,1,W)
        return F.adaptive_avg_pool2d(x * w, (H, self.K))

class MaskGuidedVariablePooling(nn.Module):
    def __init__(self, K, tau=1.0):
        super().__init__()
        self.K, self.tau = K, tau
        self.h_mlp = nn.Sequential(
            nn.LazyLinear(32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, x):
        B, C, H, W = x.shape; eps = 1e-8; dev = x.device
        h_k = self.h_mlp(x.mean(dim=[2,3])).squeeze(1)
        t = torch.linspace(0., 1., W, device=dev).unsqueeze(0)
        m = torch.sigmoid(self.tau * (h_k.unsqueeze(1) - t))
        if self.K == 1:
            wt = m / (m.sum(1, keepdim=True) + eps)
            return (x * wt.unsqueeze(1).unsqueeze(2)).sum(3, keepdim=True)
        s = torch.relu(m[:,:-1] - m[:,1:])
        pi = (s+eps)/(s.sum(1,keepdim=True)+eps)
        cdf = torch.cumsum(pi, 1).clamp(max=1.)
        q = torch.linspace(0,1,self.K+1,device=dev)[1:-1]
        cuts = torch.stack([((cdf>=qi).int().argmax(1)) for qi in q], 1) \
               if self.K > 1 else torch.empty(B,0,dtype=torch.long,device=dev)
        idx = torch.arange(W, device=dev).unsqueeze(0)
        prev = torch.zeros(B, dtype=torch.long, device=dev)
        outs = []
        for r in range(self.K):
            curr = cuts[:,r] if r<self.K-1 else torch.full((B,),W-1,dtype=torch.long,device=dev)
            seg = ((idx>=prev.unsqueeze(1))&(idx<=curr.unsqueeze(1))).float()
            w = m * seg; w = w/(w.sum(1,keepdim=True)+eps)
            outs.append((x * w.unsqueeze(1).unsqueeze(2)).sum(3))
            prev = curr + 1
        return torch.stack(outs, 3)

def make_pool(method, K, tau=1.0):
    if method == "avgpool":      return TemporalAvgPool(K)
    if method == "weightedpool": return TemporalWeightedPool(K)
    if method == "maskguided":   return MaskGuidedVariablePooling(K, tau)

# ================================================================
# Configs architecture
# ================================================================
ARCH_CFG_CICIDS = {
    10: dict(conv1_k=3, pool1_K=4, conv2_k=2, pool2_K=2, conv3_k=2),
}
ARCH_CFG_OT = {
    5:  dict(conv1_k=2, pool1_K=4, conv2_k=2, pool2_K=3, conv3_k=3),
    10: dict(conv1_k=3, pool1_K=6, conv2_k=2, pool2_K=3, conv3_k=3),
}
FC_HIDDEN_CICIDS = [32, 128, 128, 64]
FC_HIDDEN_OT     = [64, 128, 128, 64]
N_FILTERS = 16

# ================================================================
# Modèle CICIDS
# ================================================================
class CICIDSModel(nn.Module):
    def __init__(self, d_hist, pooling, n_features=82, n_classes=15):
        super().__init__()
        cfg = ARCH_CFG_CICIDS[d_hist]; nf = N_FILTERS
        self.conv1 = nn.Conv2d(1,  nf, (1, cfg["conv1_k"]))
        self.pool1 = make_pool(pooling, cfg["pool1_K"])
        self.conv2 = nn.Conv2d(nf, nf, (1, cfg["conv2_k"]))
        self.pool2 = make_pool(pooling, cfg["pool2_K"])
        self.conv3 = nn.Conv2d(nf, nf, (1, cfg["conv3_k"]))
        self.flat  = nn.Flatten()
        with torch.no_grad():
            d = torch.zeros(1, 1, n_features, d_hist)
            d = torch.relu(self.conv1(d)); d = self.pool1(d)
            d = torch.relu(self.conv2(d)); d = self.pool2(d)
            d = torch.relu(self.conv3(d))
            fsz = self.flat(d).shape[1]
        layers, in_d = [], fsz
        for out_d in FC_HIDDEN_CICIDS:
            layers += [nn.Linear(in_d, out_d), nn.ReLU()]; in_d = out_d
        layers.append(nn.Linear(in_d, n_classes))
        self.fc = nn.Sequential(*layers)
    def forward(self, x):
        x = torch.relu(self.conv1(x)); x = self.pool1(x)
        x = torch.relu(self.conv2(x)); x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        return self.fc(self.flat(x))

# ================================================================
# Modèle OT
# ================================================================
class OTUnsupModel(nn.Module):
    def __init__(self, n_features, d_hist, pooling):
        super().__init__()
        cfg = ARCH_CFG_OT[d_hist]; nf = N_FILTERS
        self.conv1 = nn.Conv2d(1,  nf, (1, cfg["conv1_k"]))
        self.pool1 = make_pool(pooling, cfg["pool1_K"])
        self.conv2 = nn.Conv2d(nf, nf, (1, cfg["conv2_k"]))
        self.pool2 = make_pool(pooling, cfg["pool2_K"])
        self.conv3 = nn.Conv2d(nf, nf, (1, cfg["conv3_k"]))
        self.flat  = nn.Flatten()
        with torch.no_grad():
            d = torch.zeros(1, 1, n_features, d_hist)
            d = torch.relu(self.conv1(d)); d = self.pool1(d)
            d = torch.relu(self.conv2(d)); d = self.pool2(d)
            d = torch.relu(self.conv3(d))
            fsz = self.flat(d).shape[1]
        layers, in_d = [], fsz
        for out_d in FC_HIDDEN_OT:
            layers += [nn.Linear(in_d, out_d), nn.ReLU()]; in_d = out_d
        layers.append(nn.Linear(in_d, n_features))
        self.fc = nn.Sequential(*layers)
    def forward(self, x):
        x = torch.relu(self.conv1(x)); x = self.pool1(x)
        x = torch.relu(self.conv2(x)); x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        return self.fc(self.flat(x))

# ================================================================
# Benchmark latence
# ================================================================
def benchmark(model, input_shape, n_warmup=200, n_runs=2000):
    model.eval()
    x = torch.randn(1, 1, *input_shape)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e6)
    times = np.array(times)
    return dict(
        median_us=np.median(times),
        p95_us=np.percentile(times, 95),
        p99_us=np.percentile(times, 99),
        throughput=1e6 / np.median(times)
    )

# ================================================================
# MACs via thop
# ================================================================
def compute_macs(model, input_shape):
    model.eval()
    x = torch.randn(1, 1, *input_shape)
    macs, _ = profile(model, inputs=(x,), verbose=False)
    return int(macs)

# ================================================================
# MAIN
# ================================================================
def main():
    print(f"\n{'='*65}")
    print(f"  Benchmark ARM Cortex-A72 (AWS Graviton2 / RPi4-class)")
    print(f"  PyTorch {torch.__version__}  |  CPU only  |  batch=1")
    print(f"{'='*65}\n")

    configs = [
        dict(name="CICIDS2017", model_cls="cicids",
             n_features=82,  d_hist=10, pooling="maskguided",
             pth="model_cicids_maskguided_d10_s1.pth",
             constraint_hz=25, constraint_label="25 flows/s"),
        dict(name="SWaT", model_cls="ot",
             n_features=51,  d_hist=10, pooling="maskguided",
             pth="unsup_swat_maskguided_d10_s0.pth",
             constraint_hz=1, constraint_label="1 Hz"),
        dict(name="WADI", model_cls="ot",
             n_features=123, d_hist=5,  pooling="maskguided",
             pth="unsup_wadi_maskguided_d5_s0.pth",
             constraint_hz=1, constraint_label="1 Hz"),
    ]

    for cfg in configs:
        print(f"── {cfg['name']}  (d={cfg['d_hist']}, {cfg['pooling']}) ──")

        if cfg["model_cls"] == "cicids":
            model = CICIDSModel(cfg["d_hist"], cfg["pooling"], cfg["n_features"])
        else:
            model = OTUnsupModel(cfg["n_features"], cfg["d_hist"], cfg["pooling"])

        state = torch.load(cfg["pth"], map_location="cpu")
        model.load_state_dict(state)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        shape    = (cfg["n_features"], cfg["d_hist"])

        # MACs
        macs = compute_macs(model, shape)

        # Latence ARM
        r = benchmark(model, shape)
        margin = r["throughput"] / cfg["constraint_hz"]

        print(f"  Params     : {n_params:,}")
        print(f"  MACs/inf   : {macs:,}  ({macs/1e6:.3f} MMACs)")
        print(f"  Latence    : median={r['median_us']:.1f} µs  p95={r['p95_us']:.1f} µs  p99={r['p99_us']:.1f} µs")
        print(f"  Throughput : {r['throughput']:.0f} inf/s")
        print(f"  Contrainte : {cfg['constraint_label']}")
        print(f"  Marge      : x{margin:.1f}")
        print()

    print(f"{'='*65}")
    print("  Graviton2 = Cortex-A72 @ ~2.5 GHz — architecturalement")
    print("  identique au Raspberry Pi 4 (Cortex-A72 @ 1.8 GHz).")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    main()

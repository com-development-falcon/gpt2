"""
Plot a GPT-2 124M training run from log/log.txt (and log/perf.csv).
Run: python plot_log.py log_full_run
Saves <logdir>/training_curves.png and <logdir>/throughput.png
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

log_dir = sys.argv[1] if len(sys.argv) > 1 else "log"

# baselines (same reference lines Andrej uses)
GPT2_124M_VAL = 3.2924  # OpenAI GPT-2 124M checkpoint, FineWeb-Edu val loss
GPT2_124M_HELLA = 0.294463  # OpenAI GPT-2 124M HellaSwag acc_norm
GPT3_124M_HELLA = 0.337  # GPT-3 paper, 124M (Small) HellaSwag

# ---- parse log.txt: "<step> <kind> <value>" ----
streams = {"train": {}, "val": {}, "hella": {}}
with open(os.path.join(log_dir, "log.txt")) as f:
    for line in f:
        parts = line.split()
        if len(parts) != 3:
            continue
        step, kind, val = int(parts[0]), parts[1], float(parts[2])
        if kind in streams:
            streams[kind][step] = val

def xy(d):
    s = sorted(d)
    return np.array(s), np.array([d[k] for k in s])

tr_x, tr_y = xy(streams["train"])
va_x, va_y = xy(streams["val"])
he_x, he_y = xy(streams["hella"])

# smooth train loss with a moving average so the trend is readable
def smooth(y, k=100):
    if len(y) < k:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    s = (c[k:] - c[:-k]) / k
    return np.concatenate([y[: k - 1], s])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(tr_x, tr_y, alpha=0.25, label="train loss (raw)")
ax1.plot(tr_x, smooth(tr_y), label="train loss (100-step avg)")
ax1.plot(va_x, va_y, "o-", ms=3, label="val loss")
ax1.axhline(GPT2_124M_VAL, color="r", ls="--", label=f"OpenAI GPT-2 124M val ({GPT2_124M_VAL})")
ax1.set_xlabel("step")
ax1.set_ylabel("loss")
ax1.set_ylim(top=min(6.0, ax1.get_ylim()[1]))
ax1.set_title(f"Loss (final val {va_y[-1]:.4f})" if len(va_y) else "Loss")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(he_x, he_y, "o-", ms=3, label="this run")
ax2.axhline(GPT2_124M_HELLA, color="r", ls="--", label=f"OpenAI GPT-2 124M ({GPT2_124M_HELLA:.3f})")
ax2.axhline(GPT3_124M_HELLA, color="g", ls="--", label=f"GPT-3 124M ({GPT3_124M_HELLA})")
ax2.axhline(0.25, color="gray", ls=":", label="random (0.25)")
ax2.set_xlabel("step")
ax2.set_ylabel("HellaSwag acc_norm")
ax2.set_title(f"HellaSwag (final {he_y[-1]:.4f})" if len(he_y) else "HellaSwag")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(log_dir, "training_curves.png")
plt.savefig(out, dpi=150)
print(f"saved {out}")
if len(va_y):
    print(f"final val loss {va_y[-1]:.4f} (min {va_y.min():.4f} @ step {va_x[va_y.argmin()]}) vs GPT-2 {GPT2_124M_VAL}")
if len(he_y):
    print(f"final HellaSwag {he_y[-1]:.4f} (max {he_y.max():.4f} @ step {he_x[he_y.argmax()]}) vs GPT-2 {GPT2_124M_HELLA:.4f}")

# ---- throughput from perf.csv ----
perf_path = os.path.join(log_dir, "perf.csv")
if os.path.exists(perf_path):
    data = np.genfromtxt(perf_path, delimiter=",", skip_header=1, names=True)  # line 1 is "# run_tag"
    steps, tok = data["step"], data["tok_per_sec"]
    keep = steps >= 3  # skip compile warm-up
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps[keep], tok[keep], lw=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("tokens / sec")
    ax.set_title(f"Throughput (median {np.median(tok[keep]):,.0f} tok/s)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(log_dir, "throughput.png")
    plt.savefig(out, dpi=150)
    print(f"saved {out}")
    print(f"median dt {np.median(data['dt_ms'][keep]):.1f} ms, median {np.median(tok[keep]):,.0f} tok/s")

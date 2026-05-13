"""Track per-segment yaw amplitude vs time across multiple captures.

Pass ``.npz`` paths on the command line, or rely on default filenames in the
current working directory (for example after copying captures out of ``/tmp``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    d = np.load(path, allow_pickle=True)
    yaw_idx = d["yaw_idx"]
    yaws = d["ja"][:, yaw_idx]
    ticks = d["ticks"]
    sec = (ticks - ticks[0]) * 0.002
    return sec, yaws


def amp_per_window(sec, yaws, win_sec=2.0):
    bin_idx = (sec // win_sec).astype(int)
    n_w = int(bin_idx.max()) + 1
    amp = np.zeros((n_w, yaws.shape[1]))
    times = (np.arange(n_w) + 0.5) * win_sec
    for w in range(n_w):
        m = bin_idx == w
        if m.any():
            col = yaws[m]
            amp[w] = col.max(0) - col.min(0)
    return times, amp


def main():
    paths = sys.argv[1:]
    if not paths:
        paths = [
            ("worm_30.npz", "a=0.30 ρ=2000 (default)"),
            ("worm_a040_d0200.npz", "a=0.40 ρ=200"),
            ("worm_a040_d1000.npz", "a=0.40 ρ=1000"),
            ("worm_a040_d2000.npz", "a=0.40 ρ=2000"),
            ("worm_a050_d0200.npz", "a=0.50 ρ=200"),
        ]
    else:
        paths = [(p, p) for p in paths]

    fig, axes = plt.subplots(len(paths), 1, figsize=(14, 2.5*len(paths)), sharex=False)
    if len(paths) == 1: axes = [axes]
    for ax, (p, label) in zip(axes, paths):
        try:
            sec, yaws = load(p)
        except Exception as e:
            ax.text(0.5, 0.5, f"FAIL: {e}", ha="center", transform=ax.transAxes)
            continue
        times, amp = amp_per_window(sec, yaws, win_sec=2.0)
        # Plot per-segment amp as colored lines
        cmap = plt.cm.viridis
        for s in range(amp.shape[1]):
            ax.plot(times, amp[:, s], c=cmap(s/12), lw=1, alpha=0.8)
        ax.set_title(f"{label}  yaw amp per segment over 2-s windows")
        ax.set_ylabel("amp (rad)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(0.3, amp.max()*1.1))
        # Mark head, mid, tail in legend
        ax.plot([], [], c=cmap(1/12), label="head (j12)")
        ax.plot([], [], c=cmap(6/12), label="mid (j67)")
        ax.plot([], [], c=cmap(10/12), label="tail (jab)")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("sim time (s)")
    plt.tight_layout()
    out_png = Path(os.environ.get("WORM_OUTDIR", ".")) / "decay_comparison.png"
    plt.savefig(out_png, dpi=120)
    print(f"saved {out_png}")

    # Also print per-window per-segment amp summary
    for p, label in paths:
        try:
            sec, yaws = load(p)
            times, amp = amp_per_window(sec, yaws, win_sec=10.0)
            print(f"\n{label}: {len(times)} windows of 10s")
            for w in range(len(times)):
                head = amp[w, 1]; mid = amp[w, 6]; tail = amp[w, 10]
                print(f"  t={times[w]:5.1f}s  head={head:.2f}  mid={mid:.2f}  tail={tail:.2f}")
        except Exception as e:
            print(f"  {label}: {e}")


if __name__ == "__main__":
    main()

"""Instability diagnostics for a high ``angle_max`` capture (``worm_35.npz`` style)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("WORM_NPZ", "worm_instability.npz")
    outdir = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("WORM_OUTDIR", "worm_plots_instability"))
    outdir.mkdir(parents=True, exist_ok=True)

    d = np.load(inp, allow_pickle=True)
    ticks = d["ticks"]
    ja = d["ja"]; jv = d["jv"]; ma = d["ma"]; S = d["S"]; fired = d["fired"]
    joint_range = d["joint_range"]   # (24, 2)
    yaw_idx = d["yaw_idx"]
    neuron_names = list(d["neuron_names"])
    SIM_DT = 0.002
    n = len(ticks)
    sec = (ticks - ticks[0]) * SIM_DT

    yaws = ja[:, yaw_idx]            # (n, 12)
    yaw_velocities = jv[:, yaw_idx]  # (n, 12)
    yaw_lim_max = float(joint_range[yaw_idx[0], 1])  # all yaws share range
    print(f"loaded {n} frames; sim_t {sec[0]:.1f}–{sec[-1]:.1f}s; yaw range ±{yaw_lim_max:.3f}")

    n_seg = 12
    ma3 = ma.reshape(n, n_seg, 4)
    dorsal = ma3[..., :2].mean(axis=-1)
    ventral = ma3[..., 2:].mean(axis=-1)
    d_minus_v = dorsal - ventral

    name_to_id = {nm: i for i, nm in enumerate(neuron_names)}
    motor_db = [f"DB{i}" for i in range(1, 8) if f"DB{i}" in name_to_id]
    motor_vb = [f"VB{i}" for i in range(1, 12) if f"VB{i}" in name_to_id]
    motor = motor_db + motor_vb
    motor_S = np.stack([S[:, name_to_id[m]] for m in motor], axis=0)  # (n_motors, n)

    # ============================== 1. Lockup timeline ==========================
    # Per 1-second windows of SIM TIME (not frame count). Broadcast samples
    # at ~104 Hz, not 500 Hz, so use sec-based bins.
    win_sec = 1.0
    bin_idx = (sec // win_sec).astype(int)
    n_wins = int(bin_idx.max()) + 1
    times = (np.arange(n_wins) + 0.5) * win_sec
    # For each window, per-segment amplitude and per-segment time-at-soft-limit
    seg_amp = np.zeros((n_wins, n_seg))
    seg_atlim = np.zeros((n_wins, n_seg))
    seg_meanv = np.zeros((n_wins, n_seg))
    s_lockup = np.zeros((n_wins,))  # fraction of frames with all yaws |y| > 0.9*lim
    head_amp = np.zeros(n_wins)
    mid_amp = np.zeros(n_wins)
    tail_amp = np.zeros(n_wins)
    for w in range(n_wins):
        mask = bin_idx == w
        if not mask.any():
            continue
        for s_idx in range(n_seg):
            col = yaws[mask, s_idx]
            seg_amp[w, s_idx] = col.max() - col.min() if col.size else 0.0
            seg_atlim[w, s_idx] = (np.abs(col) > 0.9 * yaw_lim_max).mean() if col.size else 0.0
            seg_meanv[w, s_idx] = np.abs(yaw_velocities[mask, s_idx]).mean() if col.size else 0.0
        s_lockup[w] = ((np.abs(yaws[mask]) > 0.9 * yaw_lim_max).all(axis=1)).mean() if mask.any() else 0.0
        head_amp[w] = seg_amp[w, 1]
        mid_amp[w] = seg_amp[w, 6]
        tail_amp[w] = seg_amp[w, 10]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(times, head_amp, label="head amp", c="C0")
    axes[0].plot(times, mid_amp, label="mid amp", c="C1")
    axes[0].plot(times, tail_amp, label="tail amp", c="C2")
    axes[0].set_ylabel("yaw amp (rad)\nper-segment over 1s")
    axes[0].axhline(0.05, ls="--", c="r", alpha=0.4, label="lockup threshold")
    axes[0].legend(loc="upper right"); axes[0].grid(True, alpha=0.3)

    im = axes[1].imshow(seg_amp.T, aspect="auto", cmap="viridis",
                         extent=[times[0], times[-1], n_seg+0.5, 0.5], vmin=0, vmax=2*yaw_lim_max)
    axes[1].set_ylabel("seg, yaw amp\n(rad over 1s)")
    plt.colorbar(im, ax=axes[1])

    im2 = axes[2].imshow(seg_atlim.T, aspect="auto", cmap="hot",
                          extent=[times[0], times[-1], n_seg+0.5, 0.5], vmin=0, vmax=1)
    axes[2].set_ylabel("frac time |yaw|>0.9·lim\nper segment")
    plt.colorbar(im2, ax=axes[2])

    axes[3].plot(times, s_lockup, "r-")
    axes[3].set_ylabel("frac frames\nALL yaws at limit")
    axes[3].set_xlabel("sim time (s)")
    axes[3].grid(True, alpha=0.3)
    axes[0].set_title(f"Lockup timeline at angle_max={yaw_lim_max:.2f} — amp collapse + saturation")
    plt.tight_layout()
    plt.savefig(outdir / "I01_lockup_timeline.png", dpi=120)
    plt.close()

    # ============================== 2. Joint kymograph ==========================
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    im = ax.imshow(yaws.T, aspect="auto", cmap="RdBu_r",
                   extent=[sec[0], sec[-1], n_seg+0.5, 0.5],
                   vmin=-yaw_lim_max, vmax=yaw_lim_max)
    ax.set_xlabel("sim time (s)"); ax.set_ylabel("yaw segment")
    ax.set_title(f"Joint-yaw kymograph at angle_max={yaw_lim_max:.2f}")
    plt.colorbar(im, ax=ax, label="yaw (rad)")
    plt.tight_layout()
    plt.savefig(outdir / "I02_yaw_kymograph.png", dpi=120)
    plt.close()

    # ============================== 3. D−V kymograph =========================
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    im = ax.imshow(d_minus_v.T, aspect="auto", cmap="RdBu_r",
                   extent=[sec[0], sec[-1], n_seg+0.5, 0.5], vmin=-1, vmax=1)
    ax.set_xlabel("sim time (s)"); ax.set_ylabel("segment")
    ax.set_title(f"D−V muscle drive at angle_max={yaw_lim_max:.2f}")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(outdir / "I03_dv_kymograph.png", dpi=120)
    plt.close()

    # ============================== 4. Motor neuron S =========================
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    im = ax.imshow(motor_S, aspect="auto", cmap="RdBu_r",
                   extent=[sec[0], sec[-1], len(motor)-0.5, -0.5], vmin=-1.5, vmax=1.5)
    ax.set_yticks(range(len(motor))); ax.set_yticklabels(motor)
    ax.set_xlabel("sim time (s)"); ax.set_title(f"Motor neuron S at angle_max={yaw_lim_max:.2f}")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(outdir / "I04_motor_S.png", dpi=120)
    plt.close()

    # ============================== 5. Joint velocity vs joint angle (phase plot) ===
    # When body locks up, |jv| → 0 and |yaw| → max
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey=True)
    for i, ax_ in enumerate(axes.flat):
        if i >= n_seg: break
        ax_.scatter(yaws[:, i], yaw_velocities[:, i], s=0.5, c=sec, cmap="viridis", alpha=0.5)
        ax_.set_title(f"joint{i+1}", fontsize=8)
        ax_.axvline(yaw_lim_max, ls="--", c="r", alpha=0.3)
        ax_.axvline(-yaw_lim_max, ls="--", c="r", alpha=0.3)
        ax_.grid(True, alpha=0.3)
    fig.text(0.5, 0.04, "yaw (rad)", ha="center")
    fig.text(0.04, 0.5, "yaw velocity (rad/s)", va="center", rotation="vertical")
    fig.suptitle(f"Phase-space (yaw vs ω) per segment, color=time at angle_max={yaw_lim_max:.2f}")
    plt.tight_layout(rect=[0.04, 0.04, 1, 0.96])
    plt.savefig(outdir / "I05_phase_plot.png", dpi=120)
    plt.close()

    # ============================== 6. Power balance estimate =================
    # At each tick, the body's net torque per joint ≈ I·α + drag·ω, and we can compare
    # angular velocity ω to the velocity that the muscle FORCE could sustain.
    # Estimate the velocity profile during a "successful" swing window vs. the
    # locked window.

    # Find a 4-s "alive" window early on (high amp) and a 4-s "locked" window late.
    early_idx = 0
    late_idx = n_wins - 4
    while early_idx < n_wins and head_amp[early_idx] < 0.3:
        early_idx += 1
    early_idx = min(early_idx, n_wins - 4)
    early_mask = (bin_idx >= early_idx) & (bin_idx < early_idx + 4)
    late_mask = (bin_idx >= late_idx) & (bin_idx < late_idx + 4)
    # Convert back to slices for compactness in the alive-vs-locked plot
    early_sl = np.where(early_mask)[0]
    late_sl = np.where(late_mask)[0]
    print(f"  alive window: t={times[early_idx]:.1f}-{times[early_idx]+4:.1f}s")
    print(f"  locked window: t={times[late_idx]:.1f}-{times[late_idx]+4:.1f}s")
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex="col")
    for col, idxs, title in [(0, early_sl, f"EARLY (t={times[early_idx]:.0f}s)"),
                              (1, late_sl, f"LATE (t={times[late_idx]:.0f}s)")]:
        sec_w = sec[idxs]
        axes[0, col].plot(sec_w, yaws[idxs, 0], "C0", lw=0.5, label="head")
        axes[0, col].plot(sec_w, yaws[idxs, 6], "C1", lw=0.5, label="mid")
        axes[0, col].plot(sec_w, yaws[idxs, 11], "C2", lw=0.5, label="tail")
        axes[0, col].axhline(yaw_lim_max, ls="--", c="r", alpha=0.4)
        axes[0, col].axhline(-yaw_lim_max, ls="--", c="r", alpha=0.4)
        axes[0, col].set_ylabel("yaw"); axes[0, col].legend(fontsize=7)
        axes[0, col].set_title(title)
        axes[0, col].grid(True, alpha=0.3)
        axes[1, col].plot(sec_w, yaw_velocities[idxs, 0], "C0", lw=0.5)
        axes[1, col].plot(sec_w, yaw_velocities[idxs, 6], "C1", lw=0.5)
        axes[1, col].plot(sec_w, yaw_velocities[idxs, 11], "C2", lw=0.5)
        axes[1, col].set_ylabel("yaw velocity")
        axes[1, col].grid(True, alpha=0.3)
        axes[2, col].plot(sec_w, motor_S[motor.index("DB1"), idxs], "C0", lw=0.5, label="DB1")
        axes[2, col].plot(sec_w, motor_S[motor.index("VB1"), idxs], "C2", lw=0.5, label="VB1")
        axes[2, col].set_ylabel("motor S"); axes[2, col].legend(fontsize=7)
        axes[2, col].set_xlabel("sim time (s)")
        axes[2, col].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "I06_alive_vs_locked.png", dpi=120)
    plt.close()

    # ============================== 7. summary stats ==========================
    pre_lock_idx = np.where(s_lockup > 0.5)[0]
    if len(pre_lock_idx) > 0:
        lockup_t = times[pre_lock_idx[0]]
        print(f"  >>> lockup detected at t≈{lockup_t:.1f}s (window {pre_lock_idx[0]})")
    else:
        print(f"  no full lockup detected in {n_wins}s capture")
    print(f"  early head/mid/tail amp avg: {head_amp[early_idx:early_idx+4].mean():.2f}/{mid_amp[early_idx:early_idx+4].mean():.2f}/{tail_amp[early_idx:early_idx+4].mean():.2f}")
    print(f"  late head/mid/tail amp avg: {head_amp[late_idx:].mean():.2f}/{mid_amp[late_idx:].mean():.2f}/{tail_amp[late_idx:].mean():.2f}")
    print(f"  saved plots → {outdir}")


if __name__ == "__main__":
    main()

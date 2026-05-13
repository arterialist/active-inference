"""Continuous metrics on a long-run ``worm_*.npz`` (Welch, peaks, D–V, etc.)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch, find_peaks


def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("WORM_NPZ", "worm_long_continuous.npz")
    outdir = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("WORM_OUTDIR", "worm_long_plots"))
    outdir.mkdir(parents=True, exist_ok=True)
    d = np.load(inp, allow_pickle=True)
    ticks = d["ticks"]; ja = d["ja"]; jv = d["jv"]; ma = d["ma"]; S = d["S"]; fired = d["fired"]
    com = d["com"]; sm = d["sm"]
    fe = d["fe"] if "fe" in d.files else np.zeros(len(ticks), dtype=np.float32)
    nm01 = d["nm01"] if "nm01" in d.files else np.zeros((len(ticks), 2), dtype=np.float32)
    yaw_idx = d["yaw_idx"]
    SIM_DT = 0.002
    sec = (ticks - ticks[0]) * SIM_DT
    n = len(ticks)
    seg = sm.reshape(n, -1, 3)
    fps = 1.0 / float(np.median(np.diff(ticks))) / SIM_DT
    print(f"Loaded {n} frames, sim_t {sec[-1]:.0f}s, broadcast fps≈{fps:.0f}")

    yaws = ja[:, yaw_idx]
    n_seg = 12
    ma3 = ma.reshape(n, n_seg, 4)
    dorsal = ma3[..., :2].mean(-1)
    ventral = ma3[..., 2:].mean(-1)

    head = seg[:, 0, :2]; tail = seg[:, -1, :2]
    forward = (head - tail) / (np.linalg.norm(head - tail, axis=1, keepdims=True) + 1e-9)
    heading_rad = np.unwrap(np.arctan2(forward[:, 1], forward[:, 0]))
    com_vel = np.diff(com[:, :2], axis=0) / np.diff(sec).reshape(-1, 1)
    fwd_speed_um_s = np.einsum('ij,ij->i', com_vel, forward[:-1]) * 1000

    # ---- Continuous event detection ----
    # 1. Reversal: smoothed forward speed < 0 for ≥1s
    win = max(1, int(round(2.0 * fps)))
    fwd_smooth = np.convolve(fwd_speed_um_s, np.ones(win)/win, mode='same')
    is_reversed = fwd_smooth < -10  # µm/s threshold
    # find sustained reversal episodes
    diffs = np.diff(is_reversed.astype(int))
    rev_starts = np.where(diffs == 1)[0]
    rev_ends = np.where(diffs == -1)[0]
    if len(rev_ends) and len(rev_starts) and rev_ends[0] < rev_starts[0]:
        rev_ends = rev_ends[1:]
    if len(rev_starts) > len(rev_ends):
        rev_starts = rev_starts[:len(rev_ends)]
    rev_durations = (rev_ends - rev_starts) / fps
    sustained_rev = rev_durations[rev_durations >= 1.0]

    # 2. Omega bend: ≥6 consecutive joints all bent same sign with magnitude > 0.7×limit
    # We test per-frame
    yaw_lim = 0.30  # default
    omega = np.zeros(n, dtype=bool)
    for i in range(n):
        sgn = np.sign(yaws[i])
        # Look for run of ≥6 same sign with |yaw| > 0.7*lim
        big_same = (np.abs(yaws[i]) > 0.7 * yaw_lim)
        if big_same.sum() < 6:
            continue
        # Check if they're consecutive same-sign
        signs = sgn[big_same]
        if len(signs) >= 6 and (np.all(signs == 1) or np.all(signs == -1)):
            omega[i] = True
    # Find sustained omega episodes (≥0.5s)
    diffs = np.diff(omega.astype(int))
    om_starts = np.where(diffs == 1)[0]
    om_ends = np.where(diffs == -1)[0]
    if len(om_ends) and len(om_starts) and om_ends[0] < om_starts[0]:
        om_ends = om_ends[1:]
    if len(om_starts) > len(om_ends):
        om_starts = om_starts[:len(om_ends)]
    om_durations = (om_ends - om_starts) / fps
    sustained_om = om_durations[om_durations >= 0.5]

    # 3. Per-frame zero-crossings of yaw vector
    zc_per_frame = np.zeros(n, dtype=int)
    for i in range(n):
        cnt = 0; prev = None
        for y in yaws[i]:
            if abs(y) < 0.04: continue
            s = 1 if y > 0 else -1
            if prev is not None and s != prev: cnt += 1
            prev = s
        zc_per_frame[i] = cnt

    # 4. Heading angular velocity
    heading_vel_deg_s = np.diff(np.degrees(heading_rad)) * fps

    # ---- Print summary ----
    print(f"\n=== EVENT INVENTORY ({sec[-1]:.0f}s sim time) ===")
    print(f"  Total reversal frames:    {is_reversed.sum()} ({is_reversed.mean()*100:.1f}% of run)")
    print(f"  Sustained reversals (≥1s): {len(sustained_rev)}  rate {len(sustained_rev)/sec[-1]*60:.1f}/min")
    if len(sustained_rev):
        print(f"    duration: mean {sustained_rev.mean():.2f}s  max {sustained_rev.max():.2f}s")
    print(f"  Omega-bend frames:        {omega.sum()} ({omega.mean()*100:.1f}% of run)")
    print(f"  Sustained Ω-bends (≥0.5s): {len(sustained_om)}  rate {len(sustained_om)/sec[-1]*60:.1f}/min")
    if len(sustained_om):
        print(f"    duration: mean {sustained_om.mean():.2f}s  max {sustained_om.max():.2f}s")
    print(f"  Zero-crossings:")
    for k in range(5):
        cnt = (zc_per_frame == k).sum()
        if cnt: print(f"    zc={k}: {cnt} frames ({cnt/n*100:.1f}%)")
    print(f"  Mean fwd speed: {fwd_speed_um_s.mean():+.1f}µm/s  fwd_smooth max={fwd_smooth.max():+.0f}  min={fwd_smooth.min():+.0f}")
    print(f"  Heading drift: {np.degrees(heading_rad[-1]-heading_rad[0]):+.0f}°  mean angular velocity {heading_vel_deg_s.mean():+.2f}°/s")
    com_disp = float(np.linalg.norm(com[-1, :2] - com[0, :2]))
    com_path = float(np.sum(np.linalg.norm(np.diff(com[:, :2], axis=0), axis=1)))
    print(f"  COM disp/path: {com_disp:.2f}/{com_path:.2f} mm")
    body_len = float(np.median(np.linalg.norm(head - tail, axis=1)))
    print(f"  Body length: {body_len:.2f}mm  → {com_disp/body_len:.2f} body-lengths net  ({com_disp/body_len * 60/sec[-1]:.1f} BL/min)")

    # ---- Plots ----
    # 1. Full kymograph (continuous, no down-sampling unless huge)
    skip = max(1, n // 10000)  # max ~10000 columns
    fig, ax = plt.subplots(1, 1, figsize=(20, 4))
    im = ax.imshow(yaws[::skip].T, aspect='auto', cmap='RdBu_r',
                   extent=[sec[0], sec[-1], 12.5, 0.5], vmin=-0.35, vmax=0.35)
    ax.set_xlabel('sim time (s)'); ax.set_ylabel('yaw segment')
    ax.set_title(f'Continuous yaw kymograph — {n} frames ≈ {sec[-1]:.0f}s')
    plt.colorbar(im, ax=ax, label='yaw (rad)')
    plt.tight_layout()
    plt.savefig(outdir / "01_full_yaw_kymograph.png", dpi=120)
    plt.close()

    # 2. Continuous metric timeline: forward speed, heading, zc, reversals
    fig, axes = plt.subplots(5, 1, figsize=(18, 12), sharex=True)
    axes[0].plot(sec[:-1], fwd_speed_um_s, 'g-', lw=0.3, alpha=0.5, label='instant')
    axes[0].plot(sec[:-1], fwd_smooth, 'darkgreen', lw=0.8, label='2-s smoothed')
    axes[0].axhline(0, color='k', lw=0.5)
    axes[0].set_ylabel('fwd speed (µm/s)')
    axes[0].fill_between(sec[:-1], -300, 300, where=is_reversed, alpha=0.2, color='red', label='reversal')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sec, np.degrees(heading_rad), 'b-', lw=0.5)
    axes[1].set_ylabel('heading (°, unwrapped)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(sec, zc_per_frame, 'k-', lw=0.3, alpha=0.5)
    axes[2].fill_between(sec, 0, zc_per_frame, alpha=0.3)
    axes[2].set_ylabel('zero-crossings')
    axes[2].set_yticks([0, 1, 2, 3, 4])
    axes[2].grid(True, alpha=0.3)

    axes[3].fill_between(sec, 0, omega.astype(int), alpha=0.5, color='purple')
    axes[3].set_ylabel('omega-bend')
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].set_yticks([0, 1])
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(sec[:-1], np.abs(heading_vel_deg_s), 'orange', lw=0.3)
    axes[4].set_ylabel('|angular vel| (°/s)')
    axes[4].set_xlabel('sim time (s)')
    axes[4].grid(True, alpha=0.3)
    axes[4].set_yscale('log')

    plt.suptitle(f'Continuous behavior timeline — {sec[-1]:.0f}s')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(outdir / "02_continuous_timeline.png", dpi=120)
    plt.close()

    # 3. Per-segment yaw amplitude over time (windowed but small windows)
    win_sec = 5.0
    bin_idx = (sec // win_sec).astype(int)
    n_w = int(bin_idx.max()) + 1
    seg_amp_per_window = np.zeros((n_w, 12))
    for w in range(n_w):
        m = bin_idx == w
        if m.any():
            seg_amp_per_window[w] = yaws[m].max(0) - yaws[m].min(0)
    times = (np.arange(n_w) + 0.5) * win_sec
    fig, ax = plt.subplots(1, 1, figsize=(18, 4))
    cmap = plt.cm.viridis
    for s in range(12):
        ax.plot(times, np.degrees(seg_amp_per_window[:, s]), color=cmap(s/12), lw=0.8, alpha=0.7)
    ax.plot([], [], color=cmap(1/12), label='head (j12)')
    ax.plot([], [], color=cmap(6/12), label='mid (j67)')
    ax.plot([], [], color=cmap(10/12), label='tail (jab)')
    ax.set_xlabel('sim time (s)'); ax.set_ylabel('yaw amp (°) over 5-s windows')
    ax.set_title('Per-segment amplitude over time — does any segment die?')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "03_seg_amp_timeline.png", dpi=120)
    plt.close()

    # 4. Full COM trajectory (entire run)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sc = ax.scatter(com[:, 0], com[:, 1], c=sec, cmap='viridis', s=0.3)
    ax.scatter([com[0,0]], [com[0,1]], c='green', s=100, edgecolor='black', label='start', zorder=5)
    ax.scatter([com[-1,0]], [com[-1,1]], c='red', s=100, edgecolor='black', label='end', zorder=5)
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    ax.set_aspect('equal'); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title(f'Full COM trajectory ({sec[-1]:.0f}s, {com_disp:.2f}mm net displacement)')
    plt.colorbar(sc, ax=ax, label='sim time (s)')
    plt.tight_layout()
    plt.savefig(outdir / "04_full_com.png", dpi=120)
    plt.close()

    # 5. Body shape gallery (16 evenly-spaced frames)
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    for k, ax in enumerate(axes.flat):
        i = int(n * (k + 0.5) / 16)
        cum = np.concatenate([[0], np.cumsum(yaws[i])])
        x = np.cumsum(np.cos(cum)); y = np.cumsum(np.sin(cum))
        ax.plot(x, y, 'b-', lw=4, alpha=0.8)
        ax.scatter([x[0]], [y[0]], c='green', s=50, edgecolor='black', zorder=5)
        ax.scatter([x[-1]], [y[-1]], c='red', s=50, edgecolor='black', zorder=5)
        ax.set_aspect('equal')
        ax.set_title(f't={sec[i]:.0f}s zc={zc_per_frame[i]} {"Ω" if omega[i] else ""}')
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'16 sampled body shapes across {sec[-1]:.0f}s run')
    plt.tight_layout()
    plt.savefig(outdir / "05_body_shapes.png", dpi=120)
    plt.close()

    # ---- Behavioral state classification per frame ----
    # Five state labels:
    #   "S-wave"     : zc≥2, oscillating, low |mean yaw|
    #   "C-fwd"      : zc≤1, body in single C, moving (forward speed > 30 µm/s along path)
    #   "Omega"      : ≥6 segs at >0.7 lim same sign, brief duration
    #   "stuck-curl" : ≥6 segs at >0.7 lim same sign, sustained ≥3s
    #   "transition" : everything else
    win = max(1, int(round(2.0 * fps)))
    fwd_smooth_full = np.convolve(np.concatenate([fwd_speed_um_s, [fwd_speed_um_s[-1]]]),
                                   np.ones(win)/win, mode='same')
    yaw_lim = 0.30
    state_labels = []
    for i in range(n):
        sgn = np.sign(yaws[i])
        big_same = (np.abs(yaws[i]) > 0.7 * yaw_lim)
        big_count = big_same.sum()
        if big_count >= 6:
            signs = sgn[big_same]
            if np.all(signs == 1) or np.all(signs == -1):
                state_labels.append("curl")  # might be omega or stuck
                continue
        if zc_per_frame[i] >= 2:
            state_labels.append("S-wave")
        elif zc_per_frame[i] == 1 and abs(fwd_smooth_full[i]) > 20:
            state_labels.append("C-fwd")
        else:
            state_labels.append("transition")
    state_labels = np.array(state_labels)
    # Sub-classify "curl" by duration
    curl_runs = []
    in_curl = False; cs = 0
    for i in range(n):
        if state_labels[i] == "curl" and not in_curl:
            in_curl = True; cs = i
        elif state_labels[i] != "curl" and in_curl:
            curl_runs.append((cs, i))
            in_curl = False
    if in_curl: curl_runs.append((cs, n))
    for s, e in curl_runs:
        dur = (e - s) / fps
        # Omega: brief curl (<3s), low |fe| → intended turn
        # Stuck-curl: long curl (>3s)
        if dur >= 3.0:
            state_labels[s:e] = "stuck-curl"
        else:
            state_labels[s:e] = "Omega"

    print(f"\n=== BEHAVIORAL STATE INVENTORY ({sec[-1]:.0f}s) ===")
    states, counts = np.unique(state_labels, return_counts=True)
    for st, ct in zip(states, counts):
        print(f"  {st:12s}: {ct} frames ({ct/n*100:.1f}%)  total time {ct/fps:.1f}s")

    # FE per state
    if (fe > 0).any():
        print(f"\n=== FREE ENERGY by behavioral state ===")
        for st in states:
            mask = state_labels == st
            if mask.any():
                fe_mask = fe[mask]
                print(f"  {st:12s}  FE: mean={fe_mask.mean():.3f} median={np.median(fe_mask):.3f} std={fe_mask.std():.3f}  "
                      f"min={fe_mask.min():.3f} max={fe_mask.max():.3f}")
        # Test hypothesis: "stuck-curl" has BIMODAL FE — some low (intended omega), some high (failed)
        sc_mask = state_labels == "stuck-curl"
        if sc_mask.sum() > 100:
            sc_fe = fe[sc_mask]
            print(f"\n  stuck-curl FE distribution: 25th={np.percentile(sc_fe,25):.3f} 50th={np.percentile(sc_fe,50):.3f} 75th={np.percentile(sc_fe,75):.3f}")
            # Detect bimodality via gap test
            sorted_fe = np.sort(sc_fe)
            median_gap = np.median(np.diff(sorted_fe))
            big_gaps = np.where(np.diff(sorted_fe) > 5 * median_gap)[0]
            print(f"  bimodality check: {len(big_gaps)} large gaps in sorted FE (suggesting distinct modes)")

    # Plot state timeline + FE
    fig, axes = plt.subplots(4, 1, figsize=(20, 10), sharex=True)
    state_codes = {st: i for i, st in enumerate(['S-wave', 'C-fwd', 'transition', 'Omega', 'stuck-curl'])}
    state_int = np.array([state_codes.get(s, 5) for s in state_labels])
    axes[0].imshow(state_int.reshape(1, -1), aspect='auto', cmap='tab10',
                    extent=[sec[0], sec[-1], 0, 1], vmin=0, vmax=5)
    axes[0].set_yticks([])
    axes[0].set_title(f'Behavioral state timeline')
    # Legend
    for st, code in state_codes.items():
        axes[0].plot([], [], 's', color=plt.cm.tab10(code/10), label=st)
    axes[0].legend(loc='upper right', ncol=5)

    axes[1].plot(sec, fe, 'purple', lw=0.4)
    axes[1].set_ylabel('free energy')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(sec[:-1], fwd_smooth, 'darkgreen', lw=0.4)
    axes[2].axhline(0, color='k', lw=0.5)
    axes[2].set_ylabel('fwd speed (µm/s, 2-s smoothed)')
    axes[2].grid(True, alpha=0.3)

    axes[3].imshow(yaws[::max(1, n // 5000)].T, aspect='auto', cmap='RdBu_r',
                    extent=[sec[0], sec[-1], 12.5, 0.5], vmin=-yaw_lim, vmax=yaw_lim)
    axes[3].set_ylabel('yaw segment')
    axes[3].set_xlabel('sim time (s)')

    plt.tight_layout()
    plt.savefig(outdir / "07_state_FE_timeline.png", dpi=120)
    plt.close()

    # ---- Special focus: ticks 80000-100000 (user-flagged lockup region) ----
    focus_start_tick = 80000
    focus_end_tick = 100000
    focus_mask = (ticks >= focus_start_tick) & (ticks <= focus_end_tick)
    if focus_mask.any():
        print(f"\n=== FOCUS: ticks {focus_start_tick}-{focus_end_tick} (sim t={ticks[focus_mask][0]*SIM_DT:.0f}-{ticks[focus_mask][-1]*SIM_DT:.0f}s) ===")
        f_yaws = yaws[focus_mask]
        f_sec = sec[focus_mask]
        f_dorsal = dorsal[focus_mask]
        f_ventral = ventral[focus_mask]
        f_zc = zc_per_frame[focus_mask]
        f_omega = omega[focus_mask]
        # Per-segment amp in window
        per_amp = np.degrees(f_yaws.max(0) - f_yaws.min(0))
        per_mean = np.degrees(f_yaws.mean(0))
        print(f"  per-seg yaw amp:  {per_amp.round(0).astype(int).tolist()}")
        print(f"  per-seg yaw mean: {per_mean.round(1).tolist()}")
        # Mean dorsal/ventral muscle drive per segment
        d_mean = f_dorsal.mean(0)
        v_mean = f_ventral.mean(0)
        print(f"  mean D drive:    {d_mean.round(2).tolist()}")
        print(f"  mean V drive:    {v_mean.round(2).tolist()}")
        # Lockup metric: fraction of time |yaw| > 0.7×limit (saturated)
        for_lim = 0.30
        sat_frac_per_seg = (np.abs(f_yaws) > 0.7 * for_lim).mean(0)
        print(f"  fraction saturated per seg: {sat_frac_per_seg.round(2).tolist()}")
        # Omega bend % in this window
        print(f"  omega-bend frames in focus: {f_omega.sum()}/{f_omega.size} ({f_omega.mean()*100:.1f}%)")
        print(f"  zero-crossings distribution:")
        for k in range(5):
            cnt = (f_zc == k).sum()
            if cnt: print(f"    zc={k}: {cnt} frames ({cnt/f_zc.size*100:.0f}%)")
        # Forward speed in focus
        focus_idx_start = np.where(focus_mask)[0][0]
        focus_idx_end = min(np.where(focus_mask)[0][-1], len(fwd_speed_um_s)-1)
        f_fwd = fwd_speed_um_s[focus_idx_start:focus_idx_end]
        print(f"  forward speed in focus: mean={f_fwd.mean():+.1f}µm/s  std={f_fwd.std():.0f}")

        # Plot focus window in detail
        fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)
        # Yaw kymograph (this window only)
        im = axes[0].imshow(f_yaws.T, aspect='auto', cmap='RdBu_r',
                             extent=[f_sec[0], f_sec[-1], 12.5, 0.5], vmin=-0.35, vmax=0.35)
        axes[0].set_ylabel('yaw seg')
        axes[0].set_title(f"FOCUS: yaw kymograph, ticks {focus_start_tick}-{focus_end_tick}")
        plt.colorbar(im, ax=axes[0])

        # D-V drive kymograph
        dv = f_dorsal - f_ventral
        im2 = axes[1].imshow(dv.T, aspect='auto', cmap='RdBu_r',
                              extent=[f_sec[0], f_sec[-1], 12.5, 0.5], vmin=-1, vmax=1)
        axes[1].set_ylabel('D-V drive seg')
        plt.colorbar(im2, ax=axes[1])

        # Zero-crossings + omega
        axes[2].plot(f_sec, f_zc, 'k-', lw=0.5)
        axes[2].fill_between(f_sec, 0, f_omega.astype(int)*4, alpha=0.3, color='purple', label='omega')
        axes[2].set_ylabel('zero crossings')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Mean yaw per segment over time
        for s in [1, 6, 10]:
            axes[3].plot(f_sec, np.degrees(f_yaws[:, s]), label=f'seg{s+1}', alpha=0.7)
        axes[3].axhline(0, color='k', lw=0.5)
        axes[3].set_ylabel('yaw (°)')
        axes[3].set_xlabel('sim time (s)')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(outdir / "06_focus_80k_100k.png", dpi=120)
        plt.close()

    print(f"\nplots → {outdir}")


if __name__ == "__main__":
    main()

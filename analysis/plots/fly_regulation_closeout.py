"""Render the completed regulatory comparisons from checked PAULA recordings.

No simulation, fitted decoder, interpolation or new viewer. The compact NPZ
retains the exact plotted spike events, commands and terminal gate fractions.
Run from active-inference with python -m analysis.plots.fly_regulation_closeout
RECORDING_ROOT GRAPH OUTPUT. OUTPUT must not already exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulations.drosophila.connectome import Subgraph, sha256
from simulations.drosophila.electrical_analysis import prefix
from simulations.drosophila.paired_recording import PN

COLORS = ("#222222", "#007487")


def extract(path, graph):
    metadata = json.loads((path / "analysis.json").read_text())
    metadata, data, structure = prefix(path, metadata["ticks"], current_source=False)
    roots = structure["roots"].tolist()
    columns = [i for i, root in enumerate(roots)
               if graph.nodes[root]["annotation"]["cell_class"] == "ALLN"]
    spikes = data["soma"][:, columns, 1]
    pn = data["soma"][:, roots.index(PN), 1]
    if not np.isin(spikes, [0, 1]).all() or not np.isin(pn, [0, 1]).all():
        raise ValueError("Expected binary somatic spike events")
    return {
        "ln_spikes": spikes.astype(bool), "pn_spikes": pn.astype(bool),
        "ln_roots": structure["roots"][columns],
        "commands": data["command"],
        "release_fractions": data["inhibition"][:, :, 1],
    }, metadata


def bins(values, width=25):
    starts = np.arange(0, len(values), width)
    return starts, np.array([values[i:i + width].mean() for i in starts])


def raster(ax, spikes, color):
    ticks, cells = np.nonzero(spikes)
    ax.scatter(ticks, cells, s=0.35, c=color, linewidths=0, rasterized=True)
    ax.set_ylim(-0.6, max(0.6, spikes.shape[1] - 0.4))
    ax.set_yticks([0] if spikes.shape[1] == 1 else [0, spikes.shape[1] - 1],
                  ["1"] if spikes.shape[1] == 1 else ["1", str(spikes.shape[1])])


def decorate(axes, stop, end, switch=None):
    for ax in axes.flat:
        ax.axvspan(stop, end, color="0.95", zorder=-5)
        for x in (200, stop):
            ax.axvline(x, color="0.55", linewidth=0.6)
        if switch is not None:
            ax.axvline(switch, color="0.55", linewidth=0.6, linestyle=":")
        ax.set_xlim(0, end)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8, length=3)
    for ax in axes[-1]:
        ax.set_xlabel("Model tick")


def render_history(records, output):
    fig, axes = plt.subplots(5, 3, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [0.75, 1.25, 1.25, 1, 1]})
    for col, (scope, title) in enumerate((
        ("isolated", "Isolated circuit: 44 cells"),
        ("original", "Antennal: 424 cells, original signs"),
        ("gaba", "Antennal: 424 cells, GABA control"),
    )):
        axes[0, col].set_title(title, fontsize=10, loc="left")
        for j, history in enumerate(("low", "preconditioned")):
            data = records[f"history_{scope}_{history}"]
            color, style = COLORS[j], ("-", "--")[j]
            x, y = bins(data["commands"][:, :-1].mean(axis=1))
            axes[0, col].step(x, y, where="post", color=color, linestyle=style)
            raster(axes[1 + j, col], data["ln_spikes"], color)
            cumulative = np.cumsum(data["pn_spikes"][600:])
            axes[3, col].plot(np.arange(600, 2600), cumulative, color=color, linestyle=style)
            axes[4, col].plot(data["release_fractions"].mean(axis=1), color=color,
                             linestyle=style, linewidth=1)
        axes[0, col].set_ylim(0, 5)
        axes[3, col].set_ylim(0, 450)
        axes[4, col].set_ylim(0, 1.02)
    for row, label in enumerate(("Mean ORN command\n25-tick bins, model current",
                                "Constant low\nLN identity", "High then low\nLN identity",
                                "Target PN spikes\ncumulative since tick 600",
                                "Mean terminal\nrelease fraction")):
        axes[row, 0].set_ylabel(label, fontsize=9)
    decorate(axes, 1600, 2600, 600)
    fig.suptitle("Different sensory histories, identical future input", x=.08, ha="left", fontsize=16)
    fig.text(.08, .932, "Black solid: constant low. Blue dashed: high then low. "
             "Common input begins at 600; input ends at 1600.", fontsize=9)
    fig.text(.08, .018, "Raw LN spike events; fixed neuron order within each comparison. "
             "Gray: no-input recovery. One timing seed; native plasticity remains active.", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .92), h_pad=1.1)
    save(fig, output / "sensory-history")


def render_branch(records, output):
    fig, axes = plt.subplots(3, 4, figsize=(12, 7.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1, 1]})
    for col, key in enumerate(("original_intact", "original_lnblock", "gaba_intact", "gaba_lnblock")):
        data = records[f"branch_{key}"]
        sign, condition = key.split("_")
        axes[0, col].set_title(("Original signs" if sign == "original" else "GABA control") +
                              ("\nIntact" if condition == "intact" else "\nRegulator → LN blocked"),
                              fontsize=10, loc="left")
        raster(axes[0, col], data["ln_spikes"], COLORS[0])
        axes[1, col].plot(np.arange(200, 2200), np.cumsum(data["pn_spikes"][200:]), color=COLORS[0])
        axes[1, col].set_ylim(0, 600)
        axes[2, col].plot(data["release_fractions"].mean(axis=1), color=COLORS[0], linewidth=1)
        axes[2, col].set_ylim(0, 1.02)
    for row, label in enumerate(("LN identity", "Target PN spikes\ncumulative since tick 200",
                                "Mean terminal\nrelease fraction")):
        axes[row, 0].set_ylabel(label, fontsize=9)
    decorate(axes, 1200, 2200)
    fig.suptitle("An active sensory gate does not guarantee network recovery", x=.08, ha="left", fontsize=16)
    fig.text(.08, .917, "Same high-input course; only the declared regulator-to-LN forward pathway is blocked. "
             "Input ends at 1200.", fontsize=9)
    fig.text(.08, .018, "All 208 LNs shown in the same order. Gray: no-input recovery. "
             "One timing seed. The block leaves regulator-to-PN delivery intact.", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .90), h_pad=1.1)
    save(fig, output / "regulatory-branch")


def save(fig, stem):
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".svg"), dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_root", type=Path)
    parser.add_argument("graph", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    graph = Subgraph.load(args.graph)
    paths = {f"history_{scope}_{history}": f"dl5-regulator-history-{scope}-{history}-20260910"
             for scope in ("isolated", "original", "gaba") for history in ("low", "preconditioned")}
    paths.update({"branch_original_intact": "dl5-regulator-afferent2-antennal-20260910",
                  "branch_original_lnblock": "dl5-regulator-branch-ln-20260910",
                  "branch_gaba_intact": "dl5-regulator-gaba-high-20260910",
                  "branch_gaba_lnblock": "dl5-regulator-gaba-high-lnblock-20260910"})
    records, provenance = {}, {}
    for key, relative in paths.items():
        path = args.recording_root / relative
        records[key], metadata = extract(path, graph)
        provenance[key] = {"directory": relative, "record_manifest_sha256": sha256(path / "analysis.json"),
                           "record_manifest": metadata}
    for scope in ("isolated", "original", "gaba"):
        low, high = (records[f"history_{scope}_{h}"] for h in ("low", "preconditioned"))
        np.testing.assert_array_equal(low["commands"][600:], high["commands"][600:])
        np.testing.assert_array_equal(low["ln_roots"], high["ln_roots"])
    reference_roots = records["branch_original_intact"]["ln_roots"]
    for key in paths:
        if key.startswith("branch_"):
            np.testing.assert_array_equal(reference_roots, records[key]["ln_roots"])
    np.savez_compressed(args.output / "plotted-records.npz",
                        **{f"{key}__{field}": value for key, fields in records.items()
                           for field, value in fields.items()})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "svg.fonttype": "none", "axes.linewidth": .7})
    render_history(records, args.output)
    render_branch(records, args.output)
    provenance["renderer_sha256"] = sha256(Path(__file__))
    provenance["derived_artifacts"] = {p.name: sha256(p) for p in args.output.iterdir() if p.is_file()}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()

"""Paired PN-drive interventions with bounded, full-tick port recordings.

Not an odor or fluorescence model. Compare the literal count-mapped circuit's
operating range before fitting physiology. Release blockade is an experimental
filter on forward events only; native soma/learning/return dynamics still run.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import replace
import json
import inspect
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import typer

from .connectome import Subgraph, sha256
from .execution_probe import SOMA_FIELDS
from .paula import Dynamics, Neuron, GradedNeuron, LocalCableGradedNeuron, build_paula, _assemble_network

CONDITIONS = ("intact", "kc_release_block", "apl_release_block", "apl_activation")


def source_hashes():
    from neuron.network import NeuronNetwork
    from .spatial_paula import configure_local_apl
    paths = {Path(__file__)} | {
        Path(inspect.getfile(obj))
        for obj in (Subgraph, build_paula, Neuron, GradedNeuron, LocalCableGradedNeuron,
                    configure_local_apl, NeuronNetwork, _assemble_network)
    }
    return {str(path.resolve()): sha256(path) for path in sorted(paths)}


SOURCE_FILES_AT_IMPORT = source_hashes()


def make_course(preparation, graph):
    """Four increasing drive levels, with pre/post zero input. No odor labels."""
    roots = tuple(preparation.root_to_id)
    classes = [graph.nodes[r]["annotation"]["cell_class"] for r in roots]
    kc_rows = np.flatnonzero(np.asarray(classes) == "Kenyon_Cell")
    pn_rows = np.flatnonzero(np.asarray(classes) == "ALPN")
    apl_rows = [i for i, r in enumerate(roots) if graph.nodes[r]["annotation"]["hemibrain_type"] == "APL"]
    if len(apl_rows) != 1 or not len(kc_rows) or not len(pn_rows):
        raise ValueError("The PN intervention course requires PN, KC and exactly one APL")
    drive = np.zeros((224, len(roots)), dtype=np.float64)
    epochs = []
    for i, amplitude in enumerate((1.25, 2.5, 5.0, 10.0)):
        start = 32 + i * 40
        stop = start + 40
        drive[start:stop, pn_rows] = amplitude
        epochs.append({"start": start, "stop": stop, "PN_drive": amplitude})
    return drive, kc_rows, pn_rows, apl_rows[0], epochs


class TickRecorder:
    """Dense local chunks, never a whole-course RAM allocation.

    It records all instantiated input ports including undriven boundary slots.
    Initial/final coefficients per chunk plus full every-tick coefficients make
    weight evolution inspectable. This recording is not an executable snapshot.
    """
    def __init__(self, preparation, output: Path, chunk_size=16):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.p = preparation
        self.output = output
        output.mkdir(parents=True, exist_ok=False)
        self.chunk_size = chunk_size
        self.cells = list(preparation.network.network.neurons.values())
        self.cell_rows = {c.id: i for i, c in enumerate(self.cells)}
        self.offsets = np.cumsum([0] + [c.params.num_inputs for c in self.cells])
        self.ports = [(c.id, s) for c in self.cells for s in c.postsynaptic_points]
        self.terminals = [(c.id, t) for c in self.cells for t in c.presynaptic_points]
        self.post_objects = [s for c in self.cells for s in c.postsynaptic_points.values()]
        self.pre_objects = [t for c in self.cells for t in c.presynaptic_points.values()]
        self.cable_cells = [c for c in self.cells if isinstance(c, LocalCableGradedNeuron)]
        self.chunks = []
        self.start = preparation.network.current_tick
        self.count = 0
        self.arrays = {}
        cable_columns = {}
        if self.cable_cells:
            cable_columns = {
                "cable_nodes": np.array([(c.id, int(node)) for c in self.cable_cells
                    for node in c.cable_node_ids], dtype=np.int64),
                "cable_capacity": np.concatenate([c.cable.capacity for c in self.cable_cells]),
                "cable_terminals": np.array([(c.id, int(t)) for c in self.cable_cells for t in c.terminal_ids], dtype=np.int64),
                "cable_input_ports": np.array([(c.id, sid) for c in self.cable_cells
                    for sid in range(c.params.num_inputs)], dtype=np.int64),
            }
        with (output / "columns.npz").open("xb") as out:
            np.savez_compressed(out, cell_ids=np.asarray([c.id for c in self.cells]),
                                root_ids=np.asarray(tuple(preparation.root_to_id)),
                                postsynaptic_ports=np.asarray(self.ports, dtype=np.int64),
                                terminals=np.asarray(self.terminals, dtype=np.int64).reshape(-1, 2),
                                input_delays=np.asarray([c.distances[s] for c in self.cells for s in c.postsynaptic_points]),
                                edge_bindings=preparation.edge_bindings,
                                incoming_boundary_ports=preparation.incoming_boundary_ports,
                                outgoing_boundary_terminals=preparation.outgoing_boundary_terminals,
                                **cable_columns)

    def begin(self):
        n, ports, terminals, k = len(self.cells), len(self.ports), len(self.terminals), self.chunk_size
        self.arrays = {
            "soma": np.zeros((k + 1, n, len(SOMA_FIELDS))),
            "M": np.zeros((k + 1, n, 2)),
            "inputs": np.zeros((k, ports, 4), dtype=np.float32),
            "local_potential": np.zeros((k, ports)),
            "post_weight": np.zeros((k + 1, ports)),
            "terminal_info": np.zeros((k + 1, terminals)),
            "external": np.zeros((k, n)),
            "emitted": np.zeros((k, n), dtype=np.int64),
            "blocked": np.zeros((k, n), dtype=np.int64),
            "returned": np.zeros((k, n), dtype=np.int64),
        }
        if self.cable_cells:
            nc = sum(len(c.cable.voltage) for c in self.cable_cells)
            nt = sum(len(c.terminal_ids) for c in self.cable_cells)
            ni = sum(c.params.num_inputs for c in self.cable_cells)
            self.arrays.update(cable_voltage=np.zeros((k + 1, nc)), cable_current=np.zeros((k, nc)),
                cable_terminal_release=np.zeros((k + 1, nt)), cable_arrived_current=np.zeros((k, ni)),
                cable_checks=np.zeros((k, len(self.cable_cells), 2)))
        self.snapshot(0)

    def snapshot(self, row):
        a = self.arrays
        for i, c in enumerate(self.cells):
            a["soma"][row, i] = [getattr(c, f) for f in SOMA_FIELDS]
            a["M"][row, i] = c.M_vector
        a["post_weight"][row] = [s.u_i.info for s in self.post_objects]
        a["terminal_info"][row] = [s.u_o.info for s in self.pre_objects]
        if self.cable_cells:
            a["cable_voltage"][row] = np.concatenate([c.cable.voltage for c in self.cable_cells])
            a["cable_terminal_release"][row] = np.concatenate([c.terminal_release for c in self.cable_cells])

    def finish_tick(self, external):
        self.arrays["external"][self.count] = external
        if self.cable_cells:
            self.arrays["cable_current"][self.count] = np.concatenate([c.cable.last_current for c in self.cable_cells])
            self.arrays["cable_arrived_current"][self.count] = np.concatenate([c.arrived_port_current for c in self.cable_cells])
            self.arrays["cable_checks"][self.count] = [[c.cable.last_mass_residual, c.native_mean_error] for c in self.cable_cells]
        self.count += 1
        self.snapshot(self.count)
        if self.count == self.chunk_size:
            self.flush()

    def flush(self):
        if not self.count:
            return
        filename = f"ticks-{self.start:06d}-{self.start + self.count:06d}.npz"
        states = {"soma", "M", "post_weight", "terminal_info", "cable_voltage", "cable_terminal_release"}
        arrays = {key: value[:self.count + (key in states)] for key, value in self.arrays.items()}
        with (self.output / filename).open("xb") as out:
            np.savez_compressed(out, **arrays)
        self.chunks.append({"file": filename, "start": self.start, "stop": self.start + self.count,
                            "sha256": sha256(self.output / filename)})
        self.start += self.count
        self.count = 0
        self.arrays.clear()

    @contextmanager
    def observe(self, blocked_ids: set[int]):
        """Blocks release, not O or learning. Use in a dedicated simulation process.

        Tick patches are process-global while the context is entered. Do not
        run another network concurrently in the same process.
        """
        stack = ExitStack()
        original_base, original_graded = Neuron.tick, GradedNeuron.tick
        original_cable = LocalCableGradedNeuron.tick
        record = self

        def filter_events(cell, events):
            # Native PAULA and GradedNeuron emit tuple forward events and
            # RetrogradeSignalEvent objects. Refuse an unrecognized event API.
            from neuron.neuron import RetrogradeSignalEvent
            forward = [e for e in events if isinstance(e, tuple) and len(e) == 3]
            if any(not isinstance(e, RetrogradeSignalEvent) and not (isinstance(e, tuple) and len(e) == 3) for e in events):
                raise TypeError("Unrecognized neural event format")
            row, i = record.count, record.cell_rows[cell.id]
            record.arrays["emitted"][row, i] = len(forward)
            record.arrays["returned"][row, i] = len(events) - len(forward)
            if cell.id in blocked_ids:
                record.arrays["blocked"][row, i] = len(forward)
                return [e for e in events if isinstance(e, RetrogradeSignalEvent)]
            return events

        def base_tick(cell, external_inputs, current_tick, dt=1.0):
            if not record.arrays:
                raise RuntimeError("Begin the recording chunk before network arrivals are processed")
            if current_tick != record.start + record.count:
                raise RuntimeError("Recording tick is out of alignment with the network")
            i = record.cell_rows[cell.id]
            section = slice(record.offsets[i], record.offsets[i + 1])
            record.arrays["inputs"][record.count, section] = cell.input_buffer
            active = np.flatnonzero(cell.input_buffer[:, 0] > 0)
            events = original_base(cell, external_inputs, current_tick, dt)
            for sid in active:
                record.arrays["local_potential"][record.count, record.offsets[i] + sid] = cell.postsynaptic_points[sid].potential
            # Graded release is appended outside this super() call. Filter it
            # only at the outermost call, after its real O has been produced.
            return events if isinstance(cell, GradedNeuron) else filter_events(cell, events)

        def graded_tick(cell, external_inputs, current_tick, dt=1.0):
            return filter_events(cell, original_graded(cell, external_inputs, current_tick, dt))

        def cable_tick(cell, external_inputs, current_tick, dt=1.0):
            return filter_events(cell, original_cable(cell, external_inputs, current_tick, dt))

        with stack:
            stack.enter_context(patch.object(Neuron, "tick", base_tick))
            stack.enter_context(patch.object(GradedNeuron, "tick", graded_tick))
            stack.enter_context(patch.object(LocalCableGradedNeuron, "tick", cable_tick))
            yield


def run_intervention(graph: Subgraph, output: Path, condition: str, weight_per_count: float,
                     *, spatial: Path | None = None, apl_representation="global_graded",
                     apl_cable_rm_over_ra_um: float = 25000.0):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    started = time.perf_counter()
    sources_at_start = source_hashes()
    dynamics = replace(Dynamics(), weight_per_count=weight_per_count, apl_representation=apl_representation,
                       apl_cable_rm_over_ra_um=apl_cable_rm_over_ra_um)
    p = build_paula(graph, dynamics, spatial=spatial)
    drive, kc_rows, pn_rows, apl_row, epochs = make_course(p, graph)
    ids = np.asarray(list(p.root_to_id.values()))
    blocked_ids = set(map(int, ids[kc_rows])) if condition == "kc_release_block" else {int(ids[apl_row])} if condition == "apl_release_block" else set()
    if condition == "apl_activation":
        drive[32:192, apl_row] = 50.0
    recorder = TickRecorder(p, output)
    soma_course = []
    with recorder.observe(blocked_ids):
        for tick, row in enumerate(drive):
            if not recorder.arrays:
                recorder.begin()
            for i in np.flatnonzero(row):
                root = graph.selected[int(i)]
                p.stimulate(root, float(row[i]))
            p.network.run_tick()
            soma_course.append([[getattr(c, f) for f in SOMA_FIELDS] for c in recorder.cells])
            recorder.finish_tick(row)
            if (tick + 1) % 64 == 0:
                print(f"{condition} gain={weight_per_count:g}: {tick + 1}/{len(drive)} ticks", flush=True)
    recorder.flush()
    soma = np.asarray(soma_course)
    with (output / "soma.npz").open("xb") as out:
        np.savez_compressed(out, soma=soma)
    epoch_results = []
    for epoch in epochs:
        start, stop = epoch["start"], epoch["stop"]
        kc_output = soma[start:stop, kc_rows, 1]
        epoch_results.append({**epoch, "kc_cells_fired": int(np.any(kc_output > 0, axis=0).sum()),
                              "kc_spikes": int((kc_output > 0).sum()),
                              "apl_output_min": float(soma[start:stop, apl_row, 1].min()),
                              "apl_output_max": float(soma[start:stop, apl_row, 1].max()),
                              "pn_spikes": int((soma[start:stop, pn_rows, 1] > 0).sum())})
    manifest = {
        "claim": "operating-range/intervention diagnostic; not odor, calcium, behavioral or learning replication",
        "condition": condition, "assumptions": p.assumptions,
        "anatomical_provenance": graph.provenance, "anatomy": graph.summary(),
        "protocol": {"ticks": len(drive), "epochs": epochs, "kc_rows": kc_rows.tolist(),
                     "pn_rows": pn_rows.tolist(), "apl_row": apl_row,
                     "input": "uniform current to all actual ALPN providers, not an odor pattern",
                     "block": "remove all forward release events from named cells throughout course; membrane, weights, inputs and native return events remain active",
                     "blocked_ids": sorted(blocked_ids),
                     "activation": "extra APL current 50 during ticks 32..191 only in apl_activation"},
        "recording": {"soma_fields": SOMA_FIELDS, "chunks": recorder.chunks,
                      "input_fields": ["info", "plast", "mod0", "mod1"],
                      "state_time": "initial and after each chunk tick",
                      "input_time": "native entry after arrivals; local_potential is native pre-update product",
                      "emitted": "attempted forward terminal events before experimental block",
                      **({"cable": {"voltage": "initial and after each tick, columns keyed by exact tree-node IDs",
                                    "current": "per-node current after native delay, before cable update",
                                    "terminal_release": "mean local site release before multiplying native terminal coefficient or applying lesion",
                                    "arrived_current": "per-port delayed native potentials",
                                    "checks": ["conservative_mass_residual", "cable_mean_minus_native_float32_mean"]}}
                         if recorder.cable_cells else {}),
                      "omitted": ["full in-flight queues", "individual retrograde error vectors", "terminal modulation coefficients"],
                      "not_a_checkpoint": True,
                      "columns_sha256": sha256(output / "columns.npz"),
                      "soma_sha256": sha256(output / "soma.npz")},
        "epoch_summaries_not_acceptance": epoch_results,
        "source_files": {"at_import": SOURCE_FILES_AT_IMPORT,
                         "at_start": sources_at_start, "at_finish": source_hashes(),
                         "scope": "on-disk files at three times; not a loaded-bytecode attestation"},
        "runtime_seconds": time.perf_counter() - started,
    }
    with (output / "manifest.json").open("x") as out:
        json.dump(manifest, out, indent=2)
    return manifest


def main(source: Path, output: Path, condition: str = "intact", weight_per_count: float = 0.02,
         spatial: Path | None = None, apl_representation: str = "global_graded",
         apl_cable_rm_over_ra_um: float = 25000.0):
    result = run_intervention(Subgraph.load(source), output, condition, weight_per_count,
                              spatial=spatial, apl_representation=apl_representation,
                              apl_cable_rm_over_ra_um=apl_cable_rm_over_ra_um)
    print(json.dumps(result["epoch_summaries_not_acceptance"], indent=2), flush=True)


if __name__ == "__main__":
    typer.run(main)

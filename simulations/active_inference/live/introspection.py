"""Bounded, read-only tick traces for the embodied PAULA microscope.

The live brain already emits a compact spike raster.  That is enough to make a
neuron light up, but not enough to answer *why* it lit up after the viewer has
paused or sought backwards.  ``TraceStore`` keeps a finite, in-memory window of
the actual post-tick PAULA state and the physical transducer state.  It is an
observer only: it never writes to a neuron, a synapse, or the body.

The trace deliberately separates three layers:

* static neuron/synapse manifests (parameters, metadata, graph endpoints),
* per-tick neuron state (membrane/output/adaptation variables), and
* per-tick body and synaptic-point state (the MuJoCo pose, afferents, and
  postsynaptic potential/weight values).

This gives the web client a real tick microscope without putting a JSON copy of
the entire graph on the websocket for every tick.  The ring is bounded because
an indefinitely running embodied simulation must remain RAM-flat.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading
from typing import Any, Iterable, Mapping

import numpy as np


STATE_FIELDS = (
    "S",
    "O",
    "F_avg",
    "r",
    "b",
    "t_ref",
    "t_last_fire",
    "M0",
    "M1",
)
SYNAPSE_FIELDS = ("info", "plast", "potential")


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _json_value(value: Any) -> Any:
    """Convert numpy/dataclass-ish values into safe JSON values."""
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _finite(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _finite(value) if isinstance(value, float) else value
    return str(value)


def _params(unit: Any) -> dict[str, Any]:
    params = getattr(unit, "params", None)
    if params is None:
        return {}
    names = getattr(params, "__dataclass_fields__", {})
    if names:
        return {name: _json_value(getattr(params, name, None)) for name in names}
    return {
        str(name): _json_value(value)
        for name, value in vars(params).items()
        if not str(name).startswith("_")
    }


@dataclass(frozen=True)
class EdgeManifest:
    source: int
    target: int
    synapse: int
    terminal: int
    distance: int

    def public(self) -> dict[str, int]:
        return {
            "source": self.source,
            "target": self.target,
            "synapse": self.synapse,
            "terminal": self.terminal,
            "distance": self.distance,
        }


@dataclass
class TraceFrame:
    tick: int
    step: int
    states: bytes
    synapses: bytes
    spikes: bytes
    body: dict[str, Any]
    state_shape: tuple[int, int]
    synapse_shape: tuple[int, int]


class TraceStore:
    """A bounded post-tick state ring and immutable graph manifest."""

    def __init__(self, ids: Iterable[int], units: Iterable[Any], network: Any, *, max_ticks: int = 4096):
        self.ids = tuple(int(value) for value in ids)
        self.id_to_index = {value: index for index, value in enumerate(self.ids)}
        self.max_ticks = max(1, int(max_ticks))
        self.lock = threading.RLock()
        self.frames: deque[TraceFrame] = deque(maxlen=self.max_ticks)
        self.total_captured = 0
        self.error: str | None = None
        self.neuron_manifest = tuple(self._neuron_manifest(nid, unit) for nid, unit in zip(self.ids, units))
        self.edges = self._edge_manifest(network)
        self.edge_index = {(edge.source, edge.target, edge.synapse): index for index, edge in enumerate(self.edges)}

    @staticmethod
    def _neuron_manifest(nid: int, unit: Any) -> dict[str, Any]:
        metadata = getattr(unit, "metadata", getattr(unit, "meta", {})) or {}
        return {
            "id": int(nid),
            "metadata": _json_value(metadata),
            "params": _params(unit),
            "intrinsic": {
                "t_ref": _finite(getattr(unit, "t_ref", 0.0)),
                "r": _finite(getattr(unit, "r", 0.0)),
                "b": _finite(getattr(unit, "b", 0.0)),
                "upper_t_ref_bound": _finite(getattr(unit, "upper_t_ref_bound", 0.0)),
                "lower_t_ref_bound": _finite(getattr(unit, "lower_t_ref_bound", 0.0)),
            },
        }

    @staticmethod
    def _edge_manifest(network: Any) -> tuple[EdgeManifest, ...]:
        edges: list[EdgeManifest] = []
        cache = getattr(network, "connection_cache", {})
        neurons = getattr(network, "neurons", {})
        for (source, terminal), targets in sorted(cache.items(), key=lambda item: item[0]):
            for target, synapse in sorted(targets, key=lambda item: (item[0], item[1])):
                target_unit = neurons.get(target)
                distance = int(getattr(target_unit, "distances", {}).get(synapse, 1)) if target_unit else 1
                edges.append(EdgeManifest(int(source), int(target), int(synapse), int(terminal), distance))
        return tuple(edges)

    @property
    def neuron_count(self) -> int:
        return len(self.ids)

    @property
    def synapse_count(self) -> int:
        return len(self.edges)

    def manifest(self) -> dict[str, Any]:
        with self.lock:
            return {
                "protocol": "aif-introspection/1",
                "max_ticks": self.max_ticks,
                "state_fields": list(STATE_FIELDS),
                "synapse_fields": list(SYNAPSE_FIELDS),
                "neuron_count": self.neuron_count,
                "synapse_count": self.synapse_count,
                "ids": list(self.ids),
                "neurons": list(self.neuron_manifest),
                "edges": [edge.public() for edge in self.edges],
                "trace": self.range_public(),
            }

    def range_public(self) -> dict[str, Any]:
        with self.lock:
            ticks = [frame.tick for frame in self.frames]
            return {
                "first": ticks[0] if ticks else None,
                "last": ticks[-1] if ticks else None,
                "count": len(ticks),
                "total_captured": self.total_captured,
                "ticks": ticks,
            }

    def timeline(self, *, start: int | None = None, end: int | None = None,
                 stride: int = 1, limit: int = 512) -> dict[str, Any]:
        """Return a light-weight seek model; detailed state stays behind ``/api/tick``."""
        stride = max(1, int(stride))
        limit = max(1, min(int(limit), self.max_ticks))
        with self.lock:
            frames = [frame for frame in self.frames
                      if (start is None or frame.tick >= int(start))
                      and (end is None or frame.tick <= int(end))]
        frames = frames[::stride]
        if len(frames) > limit:
            # Preserve both ends when a browser asks for a coarse overview of a long ring.
            frames = frames[:limit]
        rows = []
        for frame in frames:
            states = self._matrix(frame, len(STATE_FIELDS))
            rows.append({
                "tick": frame.tick,
                "step": frame.step,
                "firing_count": int(np.count_nonzero(states[:, 1] > 0)),
                "body": frame.body,
                "fired": [self.ids[index] for index, value in enumerate(states[:, 1]) if value > 0],
            })
        return {"protocol": "aif-introspection/1", "trace": self.range_public(), "frames": rows}

    def _body(self, ag: Any) -> dict[str, Any]:
        world = getattr(ag, "world", None)
        pose = [0.0, 0.0, 0.0]
        if world is not None and hasattr(world, "pose"):
            try:
                pose = [_finite(value) for value in world.pose()]
            except Exception:
                pass
        result: dict[str, Any] = {
            "pose": pose,
            "speed": _finite(world.speed()) if world is not None and hasattr(world, "speed") else 0.0,
            "yaw_rate": _finite(world.yaw_rate()) if world is not None and hasattr(world, "yaw_rate") else 0.0,
            "home_distance": _finite(world.dist_home()) if world is not None and hasattr(world, "dist_home") else 0.0,
            "eaten": int(getattr(world, "eaten", 0)) if world is not None else 0,
            "toxin_hits": int(getattr(world, "tox_hits", 0)) if world is not None else 0,
            "event": getattr(world, "event", None) if world is not None else None,
            "pending_event": getattr(world, "pending_event", None) if world is not None else None,
            "food_remaining": sum(1 for food in getattr(world, "foods", ()) if food is not None) if world else 0,
            "toxin_count": len(getattr(world, "toxins", ())) if world else 0,
            "arena": _finite(getattr(world, "arena", 0.0)) if world is not None else 0.0,
            "foods": [[_finite(point[0]), _finite(point[1])] for point in getattr(world, "foods", ())]
            if world is not None else [],
            "toxins": [[_finite(point[0]), _finite(point[1])] for point in getattr(world, "toxins", ())]
            if world is not None else [],
            "barriers": _json_value(getattr(world, "barriers", ())) if world is not None else [],
            "sensors": getattr(ag, "last_sensor_drives", {}),
            "obstacle": getattr(ag, "last_obstacle_afferents", {}),
            "metabolic": getattr(ag, "last_metabolic_afferents", {}),
        }
        if world is not None and hasattr(world, "metabolic_state"):
            try:
                result["metabolic"] = world.metabolic_state()
            except Exception:
                pass
        return _json_value(result)

    def capture(self, *, tick: int, step: int, ag: Any, units: Iterable[Any]) -> None:
        """Capture one completed neural tick.  Any observer failure is isolated."""
        try:
            units = tuple(units)
            n = len(self.ids)
            states = np.zeros((n, len(STATE_FIELDS)), dtype="<f4")
            spikes = np.zeros(n, dtype=np.uint8)
            for index, unit in enumerate(units):
                states[index, 0] = _finite(getattr(unit, "S", 0.0))
                states[index, 1] = _finite(getattr(unit, "O", 0.0))
                states[index, 2] = _finite(getattr(unit, "F_avg", 0.0))
                states[index, 3] = _finite(getattr(unit, "r", 0.0))
                states[index, 4] = _finite(getattr(unit, "b", 0.0))
                states[index, 5] = _finite(getattr(unit, "t_ref", 0.0))
                states[index, 6] = _finite(getattr(unit, "t_last_fire", 0.0), -1.0)
                mod = getattr(unit, "M_vector", ())
                if len(mod):
                    states[index, 7] = _finite(mod[0])
                if len(mod) > 1:
                    states[index, 8] = _finite(mod[1])
                spikes[index] = 1 if states[index, 1] > 0 else 0
            synapses = np.zeros((len(self.edges), len(SYNAPSE_FIELDS)), dtype="<f4")
            neurons = getattr(getattr(ag, "net", None), "network", None)
            neurons = getattr(neurons, "neurons", {})
            for index, edge in enumerate(self.edges):
                target = neurons.get(edge.target)
                point = getattr(target, "postsynaptic_points", {}).get(edge.synapse) if target else None
                vector = getattr(point, "u_i", None)
                synapses[index, 0] = _finite(getattr(vector, "info", 0.0))
                synapses[index, 1] = _finite(getattr(vector, "plast", 0.0))
                synapses[index, 2] = _finite(getattr(point, "potential", 0.0))
            frame = TraceFrame(
                tick=int(tick), step=int(step), states=states.tobytes(), synapses=synapses.tobytes(),
                spikes=np.packbits(spikes).tobytes(), body=self._body(ag),
                state_shape=states.shape, synapse_shape=synapses.shape,
            )
            with self.lock:
                self.frames.append(frame)
                self.total_captured += 1
                self.error = None
        except Exception as exc:  # instrumentation must never stop the animal
            with self.lock:
                self.error = f"{type(exc).__name__}: {exc}"

    def _frame(self, tick: int | None = None) -> TraceFrame | None:
        with self.lock:
            if not self.frames:
                return None
            if tick is None:
                return self.frames[-1]
            for frame in self.frames:
                if frame.tick == int(tick):
                    return frame
            return None

    @staticmethod
    def _matrix(frame: TraceFrame, field_count: int, *, synapse: bool = False) -> np.ndarray:
        shape = frame.synapse_shape if synapse else frame.state_shape
        data = frame.synapses if synapse else frame.states
        return np.frombuffer(data, dtype="<f4").reshape(shape)

    def tick_public(self, tick: int | None = None, *, include_neurons: bool = True, include_synapses: bool = False) -> dict[str, Any]:
        frame = self._frame(tick)
        if frame is None:
            raise KeyError(tick)
        states = self._matrix(frame, len(STATE_FIELDS))
        result: dict[str, Any] = {
            "protocol": "aif-introspection/1",
            "tick": frame.tick,
            "step": frame.step,
            "body": frame.body,
            "fired": [self.ids[index] for index, value in enumerate(states[:, 1]) if value > 0],
            "trace": self.range_public(),
        }
        if include_neurons:
            result["neurons"] = [
                {"id": nid, **{field: _finite(states[index, column], -1.0 if field == "t_last_fire" else 0.0)
                                for column, field in enumerate(STATE_FIELDS)}}
                for index, nid in enumerate(self.ids)
            ]
        if include_synapses:
            values = self._matrix(frame, len(SYNAPSE_FIELDS), synapse=True)
            result["synapses"] = [
                {**edge.public(), **{field: _finite(values[index, column])
                                     for column, field in enumerate(SYNAPSE_FIELDS)}}
                for index, edge in enumerate(self.edges)
            ]
        return result

    def neuron_public(self, nid: int, tick: int | None = None) -> dict[str, Any]:
        nid = int(nid)
        index = self.id_to_index.get(nid)
        if index is None:
            raise KeyError(nid)
        result = dict(self.neuron_manifest[index])
        frame = self._frame(tick)
        if tick is not None and frame is None:
            raise KeyError(tick)
        if frame is not None:
            states = self._matrix(frame, len(STATE_FIELDS))
            result["tick"] = frame.tick
            result["step"] = frame.step
            result["body"] = frame.body
            result["state"] = {field: _finite(states[index, column], -1.0 if field == "t_last_fire" else 0.0)
                                for column, field in enumerate(STATE_FIELDS)}
            result["firing"] = bool(states[index, 1] > 0)
        return result

    def neuron_history(self, nid: int, *, start: int | None = None, end: int | None = None,
                       stride: int = 1, limit: int = 4096) -> dict[str, Any]:
        nid = int(nid)
        index = self.id_to_index.get(nid)
        if index is None:
            raise KeyError(nid)
        stride = max(1, int(stride)); limit = max(1, min(int(limit), self.max_ticks))
        with self.lock:
            frames = [frame for frame in self.frames
                      if (start is None or frame.tick >= int(start))
                      and (end is None or frame.tick <= int(end))][::stride]
        if len(frames) > limit:
            frames = frames[:limit]
        rows = []
        for frame in frames:
            values = self._matrix(frame, len(STATE_FIELDS))[index]
            rows.append({"tick": frame.tick, "step": frame.step,
                         **{field: _finite(values[column], -1.0 if field == "t_last_fire" else 0.0)
                            for column, field in enumerate(STATE_FIELDS)}})
        return {"protocol": "aif-introspection/1", "id": nid, "fields": list(STATE_FIELDS), "frames": rows,
                "trace": self.range_public()}

    def synapse_public(self, *, source: int, target: int, synapse: int, tick: int | None = None) -> dict[str, Any]:
        edge_index = self.edge_index.get((int(source), int(target), int(synapse)))
        if edge_index is None:
            raise KeyError((source, target, synapse))
        edge = self.edges[edge_index]
        result = edge.public()
        frame = self._frame(tick)
        if tick is not None and frame is None:
            raise KeyError(tick)
        if frame is not None:
            values = self._matrix(frame, len(SYNAPSE_FIELDS), synapse=True)
            result.update({field: _finite(values[edge_index, column])
                           for column, field in enumerate(SYNAPSE_FIELDS)})
            states = self._matrix(frame, len(STATE_FIELDS))
            result.update({"tick": frame.tick, "source_firing": bool(states[self.id_to_index[edge.source], 1] > 0),
                           "target_firing": bool(states[self.id_to_index[edge.target], 1] > 0)})
        return result

    def synapse_history(self, *, source: int, target: int, synapse: int, start: int | None = None,
                        end: int | None = None, stride: int = 1, limit: int = 4096) -> dict[str, Any]:
        edge_index = self.edge_index.get((int(source), int(target), int(synapse)))
        if edge_index is None:
            raise KeyError((source, target, synapse))
        stride = max(1, int(stride)); limit = max(1, min(int(limit), self.max_ticks))
        with self.lock:
            frames = [frame for frame in self.frames
                      if (start is None or frame.tick >= int(start))
                      and (end is None or frame.tick <= int(end))][::stride]
        if len(frames) > limit:
            frames = frames[:limit]
        rows = []
        source_index = self.id_to_index.get(int(source)); target_index = self.id_to_index.get(int(target))
        for frame in frames:
            values = self._matrix(frame, len(SYNAPSE_FIELDS), synapse=True)[edge_index]
            states = self._matrix(frame, len(STATE_FIELDS))
            rows.append({"tick": frame.tick, "step": frame.step,
                         **{field: _finite(values[column]) for column, field in enumerate(SYNAPSE_FIELDS)},
                         "source_firing": bool(source_index is not None and states[source_index, 1] > 0),
                         "target_firing": bool(target_index is not None and states[target_index, 1] > 0)})
        return {"protocol": "aif-introspection/1", "edge": self.edges[edge_index].public(),
                "fields": list(SYNAPSE_FIELDS), "frames": rows, "trace": self.range_public()}

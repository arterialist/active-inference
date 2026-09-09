"""Display adapters kept outside the PAULA brain.

These adapters make the browser's language explicit: the simulator remains the
source of truth, while the UI receives stable JSON suitable for a graph,
timeline, or intracellular inspector.  They never write to a neuron.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

from .protocol import PROTOCOL_VERSION


class TopologyDisplayAdapter:
    """Validate and annotate a packed topology payload for display clients."""

    def adapt(self, payload: Mapping) -> dict:
        required = {"regions", "systems", "nn", "ne", "id0", "idd", "reg", "grp", "pa", "pw", "cnt", "ev", "edx", "edp"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"topology payload missing fields: {', '.join(sorted(missing))}")
        out = dict(payload)
        out["protocol"] = PROTOCOL_VERSION
        out["display"] = {"coordinate_layouts": ["anatomy", "wiring"], "synapse_edges": True}
        return out


class NeuronDisplayAdapter:
    """Convert one live PAULA unit to JSON without exposing the object itself."""

    def adapt(self, neuron_id: int, unit, *, incoming: list[dict], outgoing: list[dict]) -> dict:
        last_fire = getattr(unit, "t_last_fire", 0)
        if not math.isfinite(float(last_fire)):
            last_fire = None
        params = getattr(unit, "params", None)
        param_fields = getattr(params, "__dataclass_fields__", {})
        if param_fields:
            raw_params = {name: getattr(params, name, None) for name in param_fields}
        else:
            raw_params = {name: value for name, value in vars(params).items()
                          if not str(name).startswith("_")} if params is not None else {}
        def safe(value):
            if hasattr(value, "tolist"):
                return value.tolist()
            if isinstance(value, (int, float, str, bool)) or value is None:
                return value
            return str(value)
        return {
            "protocol": PROTOCOL_VERSION,
            "id": int(neuron_id),
            "intracellular": {
                "S": float(getattr(unit, "S", 0.0)),
                "O": float(getattr(unit, "O", 0.0)),
                "t_last_fire": None if last_fire is None else int(last_fire),
                "t_ref": float(getattr(unit, "t_ref", 0.0)),
                "metadata": dict(getattr(unit, "metadata", getattr(unit, "meta", {})) or {}),
                "parameters": {str(key): safe(value) for key, value in raw_params.items()},
            },
            "synapses": {"incoming": incoming, "outgoing": outgoing},
        }

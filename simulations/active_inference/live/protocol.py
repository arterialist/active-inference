"""Versioned wire contracts for live neural inspection.

JSON is used for control/state/topology metadata.  The raster stream is a
small binary protocol so every neural tick can be observed without converting
the whole graph to JSON.  Clients must consume the ``hello`` contract instead
of assuming a neuron count or fixed IDs.
"""

from __future__ import annotations

from typing import Iterable, Mapping


PROTOCOL_VERSION = "aif-live/1"
INTROSPECTION_PROTOCOL_VERSION = "aif-introspection/1"
BINARY_FRAME_VERSION = 1
BINARY_FRAME_HEADER = {
    "version": BINARY_FRAME_VERSION,
    "tick": "uint32-le",
    "spikes": "ceil(neuron_count / 8) bytes, MSB-first",
    "graded": "four float32-le membrane values",
}
STATE_SCHEMA = {
    "ready": "bool",
    "version": "agent version id",
    "step": "MuJoCo agent step",
    "ticks": "PAULA neural tick",
    "pose": "[x, y, yaw]",
    "world": "selected world id",
    "barrier": "null or physical challenge id",
    "barriers": "physical obstacle rectangles for display",
    "obstacle": "body transducer snapshot (left/right/onset/contact)",
    "dec": "read-only display rows",
    "rate": "agent steps per second",
    "trace": "bounded tick trace range; see /api/introspection and /api/tick",
}


def hello(version: Mapping, ids: Iterable[int], graded: Iterable[int]) -> dict:
    ids = [int(i) for i in ids]
    graded = [int(i) for i in graded]
    return {
        "protocol": PROTOCOL_VERSION,
        "hello": True,
        "version": dict(version),
        "neuron_count": len(ids),
        "ids": ids,
        "graded": graded,
        "binary": dict(BINARY_FRAME_HEADER),
        "state_schema": dict(STATE_SCHEMA),
    }


def session(*, version: Mapping, ids: Iterable[int], graded: Iterable[int], http_port: int, ws_port: int) -> dict:
    msg = hello(version, ids, graded)
    msg.update({
        "service": "active-inference-live-brain",
        "endpoints": {
            "state": "/api/state",
            "command": "/api/command",
            "topology": "/api/topology",
            "neuron": "/api/neuron/{id}",
            "neuron_history": "/api/neuron/{id}/history",
            "introspection": "/api/introspection",
            "tick": "/api/tick/{tick}",
            "synapse": "/api/synapse?source=<id>&target=<id>&synapse=<slot>",
            "schema": "/api/schema",
            "health": "/api/health",
            "lab": "/lab",
        },
        "http": {"host": "127.0.0.1", "port": int(http_port)},
        "websocket": {"host": "127.0.0.1", "port": int(ws_port), "path": "/ws"},
    })
    return msg


def schema() -> dict:
    return {
        "protocol": PROTOCOL_VERSION,
        "binary_frame": dict(BINARY_FRAME_HEADER),
        "state": dict(STATE_SCHEMA),
        "commands": {
            "run": "GET/POST {c:run,n:<ticks|-1>}",
            "pause": "GET/POST {c:pause}",
            "step": "GET/POST {c:step}",
            "world": "GET/POST {c:world,w:<meadow|minefield|sparse|obstacle_detour|obstacle_corner|obstacle_chicane|obstacle_maze>}",
            "params": "GET/POST {c:params,p:<json>}",
            "reset": "GET/POST {c:reset}",
        },
        "introspection": {
            "protocol": INTROSPECTION_PROTOCOL_VERSION,
            "manifest": "/api/introspection",
            "range": "/api/trace",
            "tick": "/api/tick/{tick}?detail=neurons|synapses|all",
            "neuron": "/api/neuron/{id}?tick=<tick>",
            "neuron_history": "/api/neuron/{id}/history?from=<tick>&to=<tick>&stride=<n>",
            "synapse": "/api/synapse?source=<id>&target=<id>&synapse=<slot>&tick=<tick>",
            "semantics": (
                "post-tick PAULA state plus physical transducers; the trace is read-only and bounded, "
                "so a long-running server remains RAM-flat"
            ),
        },
    }

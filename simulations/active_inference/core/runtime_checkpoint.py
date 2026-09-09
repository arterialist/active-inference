"""Exact, trusted-local PAULA runtime checkpoints for counterfactual research.

Run with `uv run --no-sync --with cloudpickle==3.1.2 python ...`.
These executable Python checkpoints are ONLY for
files created and kept in a trusted research workspace, never uploaded media,
downloaded models or network input. Hash checks detect corruption and runtime
changes; they do not authenticate a malicious file. Loading requires explicit
trusted=True before any unpickling occurs.

The object graph preserves numpy scalar dtypes, queued events, extension state
and cache/buffer aliases. Logging handlers are omitted and rebound to the local
logger, preserving context but not log formatting or sinks. This stores the
neural runtime only, not a physical body or external environment. Branch.step()
isolates Python and legacy numpy RNG streams from other branches. It is not
thread-safe because those generators are process-global; use one worker process
per independently executing branch.
"""
from dataclasses import dataclass
import hashlib
import inspect
import io
import json
from pathlib import Path
import platform
import random
import sys
import zipfile

import numpy as np
from loguru import logger


def _logger_context(extra):
    return logger.bind(**extra)


def _serializer():
    try:
        import cloudpickle
    except ImportError as exc:
        raise ImportError('Run with uv run --no-sync --with cloudpickle==3.1.2 for trusted runtime checkpoints') from exc
    class CheckpointPickler(cloudpickle.CloudPickler):
        def reducer_override(self, obj):
            if type(obj) is type(logger):
                return _logger_context, (dict(obj._options[-1]),)
            return super().reducer_override(obj)
    return cloudpickle, CheckpointPickler


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _versions():
    cloudpickle, _ = _serializer()
    return dict(python=sys.version, implementation=platform.python_implementation(),
                numpy=np.__version__, cloudpickle=cloudpickle.__version__)


def _runtime_sources(net, extra):
    paths = {Path(__file__).resolve(), *(Path(p).resolve() for p in extra)}
    classes = {type(net), type(net.network)}
    classes.update(type(n) for n in net.network.neurons.values())
    for cls in classes:
        for base in cls.__mro__:
            if base is object:
                continue
            paths.add(Path(inspect.getfile(base)).resolve())
    return {str(p): _digest(p) for p in sorted(paths)}


def check_buffer_aliases(net):
    """A restored array with equal values is insufficient if caches point away."""
    topo = net.network
    if set(topo.connection_cache) != set(topo.fast_connection_cache):
        raise ValueError('Connection cache keys disagree')
    for key, targets in topo.connection_cache.items():
        direct = topo.fast_connection_cache[key]
        if len(targets) != len(direct):
            raise ValueError('Connection cache fan-out differs')
        for (nid, sid), (buffer, actual_sid) in zip(targets, direct, strict=True):
            if sid != actual_sid or buffer is not topo.neurons[nid].input_buffer:
                raise ValueError('Disconnected neural input-buffer alias')
    vec = getattr(topo, '_ext_vec', None)
    if vec is not None:
        for buffer, synapses, rows in vec['groups']:
            for sid, row in zip(synapses, rows, strict=True):
                nid, expected_sid = vec['keys'][int(row)]
                if int(sid) != expected_sid or buffer is not topo.neurons[nid].input_buffer:
                    raise ValueError('Disconnected external-input-buffer alias')


@dataclass
class RuntimeBranch:
    network: object
    python_rng: object
    numpy_rng: object
    manifest: dict

    def step(self):
        """Advance neural equations while retaining this branch's RNG history."""
        ambient_python, ambient_numpy = random.getstate(), np.random.get_state()
        random.setstate(self.python_rng); np.random.set_state(self.numpy_rng)
        try:
            return self.network.run_tick()
        finally:
            self.python_rng, self.numpy_rng = random.getstate(), np.random.get_state()
            random.setstate(ambient_python); np.random.set_state(ambient_numpy)


def save_checkpoint(net, path, *, sources=()):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(path)
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    if isinstance(net, RuntimeBranch):
        python_rng, numpy_rng = net.python_rng, net.numpy_rng
        net = net.network
    check_buffer_aliases(net)
    _, pickler = _serializer()
    stream = io.BytesIO()
    pickler(stream, protocol=5).dump(dict(network=net, python_rng=python_rng, numpy_rng=numpy_rng))
    payload = stream.getvalue()
    manifest = dict(format='trusted-paula-runtime-1', versions=_versions(), sources=_runtime_sources(net, sources),
                    tick=net.current_tick, neurons=len(net.network.neurons),
                    payload_sha256=hashlib.sha256(payload).hexdigest(),
                    scope='Neural runtime only. Executable trusted-local checkpoint. Logging sinks are rebound.')
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        z.writestr('manifest.json', json.dumps(manifest, sort_keys=True, allow_nan=False))
        z.writestr('runtime.pkl', payload)
    return manifest


def load_checkpoint(path, *, trusted=False):
    if trusted is not True:
        raise ValueError('Executable checkpoint: trusted=True is required for your own trusted local file')
    cloudpickle, _ = _serializer()
    with zipfile.ZipFile(path) as z:
        if set(z.namelist()) != {'manifest.json', 'runtime.pkl'} or len(z.namelist()) != 2:
            raise ValueError('Unexpected checkpoint entries')
        manifest = json.loads(z.read('manifest.json'))
        if manifest['format'] != 'trusted-paula-runtime-1' or manifest['versions'] != _versions():
            raise ValueError('Checkpoint runtime version mismatch')
        if any(_digest(p) != h for p, h in manifest['sources'].items()):
            raise ValueError('Checkpoint source mismatch')
        payload = z.read('runtime.pkl')
        if hashlib.sha256(payload).hexdigest() != manifest['payload_sha256']:
            raise ValueError('Checkpoint payload changed')
    data = cloudpickle.loads(payload)
    net = data['network']
    if net.current_tick != manifest['tick'] or len(net.network.neurons) != manifest['neurons']:
        raise ValueError('Checkpoint state disagrees with manifest')
    check_buffer_aliases(net)
    return RuntimeBranch(net, data['python_rng'], data['numpy_rng'], manifest)

"""Compatibility shim; the PAULA arbiter now lives in components.arbitration.

The path-based loader keeps old standalone probes and direct module imports
working even when the repository package is not on ``sys.path``.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_SOURCE = Path(__file__).resolve().parent / "components" / "arbitration" / "paula.py"
_SPEC = spec_from_file_location("_aif_arbiter_component", _SOURCE)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import failure guard
    raise ImportError(f"Cannot load moved arbiter implementation: {_SOURCE}")
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
globals().update({name: value for name, value in vars(_MODULE).items() if not name.startswith("_")})
del module_from_spec, spec_from_file_location, Path, _SPEC, _MODULE, _SOURCE

if __name__ == "__main__":
    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).resolve().parent / "components" / "arbitration" / "paula.py"), run_name="__main__")

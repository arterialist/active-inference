"""Compatibility shim; the embodied body now lives in components.body.world.

Some maintained probes load this file directly with ``spec_from_file_location``
instead of importing the repository as a package.  Resolve the sibling source
by path so that both entry styles execute the same implementation.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_SOURCE = Path(__file__).resolve().parent / "components" / "body" / "world.py"
_SPEC = spec_from_file_location("_aif_world3d_component", _SOURCE)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import failure guard
    raise ImportError(f"Cannot load moved World3D implementation: {_SOURCE}")
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
globals().update({name: value for name, value in vars(_MODULE).items() if not name.startswith("_")})
del module_from_spec, spec_from_file_location, Path, _SPEC, _MODULE, _SOURCE

if __name__ == "__main__":
    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).resolve().parent / "components" / "body" / "world.py"), run_name="__main__")

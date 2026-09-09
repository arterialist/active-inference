"""Single-topology composer for PAULA component fragments.

The legacy builder still owns the large, proven core while extraction proceeds
without changing neuron IDs or synapse semantics.  New fragments are added as
explicit build contributions here; the composer never runs during a tick.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

try:
    from simulations.active_inference.core.components import registry
except ModuleNotFoundError:
    # ``aif_agent3d.py`` is also loaded directly by long-lived probes.  Keep
    # the build seam usable in that mode without manufacturing a second
    # registry implementation.
    _source = Path(__file__).resolve().parent / "components.py"
    _spec = spec_from_file_location("_aif_component_registry", _source)
    if _spec is None or _spec.loader is None:  # pragma: no cover - import failure guard
        raise ImportError(f"Cannot load component registry: {_source}")
    _registry_module = module_from_spec(_spec)
    sys.modules["_aif_component_registry"] = _registry_module
    _spec.loader.exec_module(_registry_module)
    registry = _registry_module.registry
    del _source, _spec, _registry_module


def compose_brain(
    legacy_builder: Callable[..., tuple[list, list, list, list]],
    build_kwargs: Mapping,
    *,
    components: Sequence[str] | None = None,
) -> tuple[list, list, list, list]:
    """Build one PAULA topology from an explicit component selection.

    The existing core remains the compatibility baseline when ``components``
    is omitted.  An explicit selection is structural: the legacy builder gets
    a profile keyword and omits populations outside it.  All communication
    after this function returns is ordinary PAULA synaptic propagation.
    """
    kwargs = dict(build_kwargs)
    if components is not None:
        selected = tuple(components)
        registry.validate(selected)
        # ``components`` is a topology contract, not just documentation.  The
        # legacy builder accepts the profile through an explicit keyword so it
        # can omit populations which are not part of the selected version.
        # Keeping the keyword out of the default path preserves the historical
        # full-brain construction for callers that do not opt into a version.
        kwargs["enabled_components"] = selected
        if "arbitration.metabolic_sleep" in selected:
            kwargs["metabolic_sleep"] = True
    return legacy_builder(**kwargs)

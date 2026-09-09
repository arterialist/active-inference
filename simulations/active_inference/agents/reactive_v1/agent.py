"""Composable V1 entrypoint, backed by the proven existing agent implementation."""

from __future__ import annotations

from ...aif_agent3d import AIFAgent3D
from ...core import AgentBlueprint


blueprint = AgentBlueprint(
    name="reactive_v1",
    inherited=("sensory.olfactory_valence", "motor.cpg_muscle"),
)
blueprint.validate()


class ReactiveV1Agent(AIFAgent3D):
    """Current compatible reactive composition.

    The accepted CPG/muscle substrate is inherited rather than removed.  This
    class is a composition boundary, not a fork of the whole-brain script.
    """

    def __init__(self, *args, **kwargs):
        # Versioned entrypoints opt into the strict topology profile.  The
        # legacy full builder remains available only to callers that invoke
        # AIFAgent3D directly without ``components``.
        kwargs.setdefault("components", blueprint.components)
        super().__init__(*args, **kwargs)

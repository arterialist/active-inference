"""V2 composes learned mushroom-body valence on top of V1."""

from ..reactive_v1.agent import ReactiveV1Agent, blueprint as v1_blueprint
from ...core import AgentBlueprint
from ...aif_agent3d import AIFAgent3D


blueprint = AgentBlueprint(
    name="memory_v2",
    inherited=v1_blueprint.components,
    added=("learning.mushroom_body",),
)
blueprint.validate()


class MemoryV2Agent(ReactiveV1Agent):
    """Current compatible MB composition; its causal harness supplies teaching."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("components", blueprint.components)
        # Call the shared builder directly so V2 does not inherit V1's
        # defaulted component tuple through the compatibility wrapper.
        AIFAgent3D.__init__(self, *args, **kwargs)

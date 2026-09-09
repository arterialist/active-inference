"""Prototype composition makes non-default component activation explicit."""

from __future__ import annotations

from dataclasses import dataclass

from ...aif_agent3d import AIFAgent3D
from ...core.components import ComponentStatus, registry


@dataclass(frozen=True)
class PrototypeSelection:
    components: tuple[str, ...]
    allow_nonaccepted: bool = False

    def validate(self) -> None:
        registry.validate(self.components)
        if not self.allow_nonaccepted:
            pending = [name for name in self.components if registry.get(name).status is not ComponentStatus.ACCEPTED]
            if pending:
                raise ValueError("Experimental/quarantined components require allow_nonaccepted=True: " + ", ".join(pending))


class PrototypeAgent(AIFAgent3D):
    """Build the established agent with an auditable optional-component selection."""

    def __init__(self, *, selection: PrototypeSelection, **kwargs):
        selection.validate()
        self.selection = selection
        super().__init__(components=selection.components, **kwargs)

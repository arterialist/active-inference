"""V3 metabolic interoception composed on the existing one-brain builder."""

from ...core import AgentBlueprint, ChallengeSpec
from ...aif_agent3d import AIFAgent3D
from ...embodied_config import DEFAULT_EMBODIED_CONFIG, EmbodiedAgentConfig


challenge = ChallengeSpec(
    name="metabolic_rest",
    prior_failure="Immediate hunger drain makes food contact equivalent to immediate usable energy; "
                  "the prior agent cannot trade active search against digestion/rest.",
    minimum_body_delta=("body.metabolic_organs",),
    required_neural_delta=("arbitration.metabolic_sleep",),
    acceptance_harness="experiments/embodied_v3_trajectory_causal.py",
    ablation_harness="experiments/embodied_v3_trajectory_causal.py",
)

blueprint = AgentBlueprint(
    name="interoceptive_v3",
    inherited=(
        "sensory.olfactory_valence", "motor.cpg_muscle", "learning.mushroom_body",
        "arbitration.foraging_exploration",
    ),
    added=("body.metabolic_organs", "arbitration.metabolic_sleep"),
    challenge=challenge,
)
blueprint.validate()


class InteroceptiveV3Agent(AIFAgent3D):
    """One PAULA topology with metabolic body afferents and SLEEP WTA."""

    def __init__(self, *args, config: EmbodiedAgentConfig | None = None, **kwargs):
        config=(config or DEFAULT_EMBODIED_CONFIG).with_overrides(metabolic_sleep=True)
        # In the strict no-compass topology, EXPLORE has no uncertainty
        # afferent.  The original shared 0.5 mode-drive weight left its
        # spontaneous WTA activity marginally stronger than the hunger-driven
        # FORAGE population, so the embodied arbiter could not establish its
        # pre-meal state.  This is a circuit calibration, not a host-side mode
        # choice: it strengthens the declared hunger afferent for V3/V4 while
        # leaving the legacy full-brain builder byte-compatible.
        kwargs.setdefault("w_unc_mode", 0.8)
        kwargs.setdefault("components", blueprint.components)
        super().__init__(*args, config=config, **kwargs)

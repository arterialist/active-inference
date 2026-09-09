"""One auditable configuration for the embodied PAULA agent.

The MuJoCo runner, direct ``AIFAgent3D.tick`` harness, live laboratory, and
new experiment runners all use this object.  It deliberately contains only
the shared, mechanism-relevant settings; experiment-specific stimuli and
uncommon exploratory knobs remain explicit call-site overrides.

``DEFAULT_EMBODIED_CONFIG`` preserves the formerly hard-coded closed-loop
episode configuration: a 1.0 ring-maintenance current on every neural tick.
``LEGACY_DIRECT_TICK_CONFIG`` retains the old direct/live default (tonic off)
for controlled comparisons.  Neither is a behavioural-success claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EmbodiedAgentConfig:
    """Build-time and runtime settings whose disagreement previously split runners."""

    # Ring maintenance and the core compass settings.
    tonic_amp: float = 1.0
    # ``0`` means an ungated tonic, exactly matching the former run_episode path.
    # A positive value linearly suppresses tonic drive as |CCW| + |CW| reaches it.
    tonic_gate: float = 0.0
    w_tonic: float = 0.12
    r_conj_lo: float = 1.35
    r_conj_hi: float = 1.75
    d7: bool = True

    # Sensorimotor and visual-motion settings used by the composed agent.
    k_prop: float = 2.0
    k_ang: float = 0.65
    w_cap: float = 1.2
    # Angular proprioception is filtered in ``World3D.yaw_rate_net`` before
    # it reaches the P-EN shift populations.  Keep its historical runtime
    # defaults explicit and shared so a live run and an evidence runner cannot
    # silently compare different vestibular signals.
    wz_tau: float = 60.0
    wz_comp: float = 0.0
    # Experimental PAULA opponent vestibular filter.  When selected it
    # replaces the host-side filtered current with two graded neural cells
    # driven by the signed physical yaw-rate transducer.
    vestibular_opponent: bool = False
    vop_tau: float = 60.0
    vop_gain: float = 1.0
    d_emd: int = 16
    w_hs_shift: float = 0.0
    w_hs_anti: float = 1.6

    # The speed-gated path-integration input.  These remain the existing
    # construction defaults; the object makes their actual values observable.
    r_pg: float = 1.7
    w_pg: float = 1.4
    w_pg_s: float = 1.4
    k_pi: float = 1.0
    cd_neg: float = 0.0
    cd_cut: float = 0.05

    # Structural options are explicit so a trace records what brain was built.
    # The toxin temporal-rise bank prevents the opponent turn circuit from
    # cancelling a symmetric, head-on hazard.  Its physical causal ablation
    # is maintained in embodied_headon_toxin_causal.py.
    trise: bool = True
    w_trise: float = 3.0
    vac: bool = False
    accum: bool = False
    w_peg: float = 0.0
    w_eff: float = 0.0
    w_rly_eff: float = 0.0
    sh_bank: bool = False
    w_lgi: float = 0.0

    # Learned olfactory valence reaches the motor decision through a
    # lateral-horn-like population.  The LH retains a weak innate sensory
    # contribution; physical toxin teaching recruits its additional
    # MBON/AVOID input and changes the embodied outcome.
    mb_lh: bool = True
    w_lh_avoid: float = 3.0
    w_lh_sensor: float = 1.2
    r_lh: float = 0.8
    w_lh_turn: float = 10.0

    # FORAGE is an interoceptive state, not merely a decoded label: it vetoes
    # the PAULA background SEARCH generator.  Hazard escape has a separate
    # STEER population and therefore remains available while hungry.
    w_forage_search: float = -2.0

    # V3 metabolic interoception.  Disabled in the stable defaults so V1/V2
    # evidence remains bit-for-bit comparable; the V3 package opts in.
    metabolic_sleep: bool = False
    w_sleep_veto: float = -8.0
    w_sleep_gut: float = 2.5
    w_sleep_energy: float = 0.5
    w_sleep_low: float = 0.0
    w_forage_low: float = 0.5
    # V3 has no visual/compass uncertainty ladder.  Usable energy therefore
    # provides the missing *readiness to explore* afferent: low energy drives
    # FORAGE, while a replenished store can recruit EXPLORE.  This is a body
    # signal, not a host-side mode selector.
    w_explore_energy: float = 2.0
    w_sleep_hazard: float = -4.0

    # Runtime conversion of the physical metabolic state into the existing
    # PAULA hunger ladder.  Gut load is the delayed meal signal; it is not an
    # immediate reward pulse.  Keeping these values explicit prevents the
    # closed-loop runner and direct/live probes from silently disagreeing.
    metabolic_hunger_fill_base: float = 0.35
    metabolic_hunger_low_gain: float = 0.35
    metabolic_hunger_gut_gain: float = 3.0
    # Energy afferents use a slightly higher receptor gain than the other
    # interoceptive channels so a mid-range store is represented by spikes,
    # not only subthreshold membrane drift.
    metabolic_energy_afferent_gain: float = 4.0
    # Toxin is a local no-go cue in V3.  A peripheral gain below the broad
    # food-odor gain prevents a distant plume from cancelling every food
    # approach, while contact/near-field concentrations still recruit the
    # crossed turn and TRISE escape pathways.
    metabolic_toxin_gain: float = 0.5
    metabolic_toxin_threshold: float = 0.8

    def with_overrides(self, **overrides: Any) -> "EmbodiedAgentConfig":
        """Return a validated config with named settings replaced.

        Rejecting misspellings is intentional: silent allowlist drops were a
        recurring source of false circuit conclusions in this research stream.
        """
        valid = set(self.__dataclass_fields__)
        unknown = sorted(set(overrides) - valid)
        if unknown:
            raise ValueError(f"Unknown embodied config setting(s): {', '.join(unknown)}")
        return replace(self, **overrides)

    def build_kwargs(self) -> dict[str, Any]:
        """Parameters supplied while the PAULA network is constructed."""
        return {
            "w_tonic": self.w_tonic,
            "r_conj_lo": self.r_conj_lo,
            "r_conj_hi": self.r_conj_hi,
            "d7": self.d7,
            "d_emd": self.d_emd,
            "w_hs_shift": self.w_hs_shift,
            "w_hs_anti": self.w_hs_anti,
            "r_pg": self.r_pg,
            "w_pg": self.w_pg,
            "w_pg_s": self.w_pg_s,
            "k_pi": self.k_pi,
            "cd_neg": self.cd_neg,
            "cd_cut": self.cd_cut,
            "trise": self.trise,
            "w_trise": self.w_trise,
            "vac": self.vac,
            "accum": self.accum,
            "w_peg": self.w_peg,
            "w_eff": self.w_eff,
            "w_rly_eff": self.w_rly_eff,
            "sh_bank": self.sh_bank,
            "w_lgi": self.w_lgi,
            "mb_lh": self.mb_lh,
            "w_lh_avoid": self.w_lh_avoid,
            "w_lh_sensor": self.w_lh_sensor,
            "r_lh": self.r_lh,
            "w_lh_turn": self.w_lh_turn,
            "w_forage_steer": self.w_forage_search,
            "metabolic_sleep": self.metabolic_sleep,
            "w_sleep_veto": self.w_sleep_veto,
            "w_sleep_gut": self.w_sleep_gut,
            "w_sleep_energy": self.w_sleep_energy,
            "w_sleep_low": self.w_sleep_low,
            "w_forage_low": self.w_forage_low,
            "w_explore_energy": self.w_explore_energy,
            "w_sleep_hazard": self.w_sleep_hazard,
            "vestibular_opponent": self.vestibular_opponent,
            "vop_tau": self.vop_tau,
            "vop_gain": self.vop_gain,
        }

    def manifest(self) -> dict[str, Any]:
        """Stable, JSON-ready record for an experiment manifest or live snapshot."""
        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "tonic": {
                "amplitude": self.tonic_amp,
                "velocity_gate": self.tonic_gate,
                "synapse_weight": self.w_tonic,
            },
            "compass": {
                "k_ang": self.k_ang,
                "drive_cap": self.w_cap,
                "yaw_filter_tau": self.wz_tau,
                "yaw_filter_compensation": self.wz_comp,
                "opponent_filter": {
                    "enabled": self.vestibular_opponent,
                    "tau": self.vop_tau,
                    "gain": self.vop_gain,
                },
                "shift_thresholds": [self.r_conj_lo, self.r_conj_hi],
                "delta7": self.d7,
                "emd_delay": self.d_emd,
                "hs_shift_weight": self.w_hs_shift,
            },
            "path_integration": {
                "proprioceptive_gain": self.k_prop,
                "pg": {"threshold": self.r_pg, "ring_weight": self.w_pg,
                       "speed_weight": self.w_pg_s},
                "analog_gain": self.k_pi,
                "signed_cd_weight": self.cd_neg,
                "cd_cut": self.cd_cut,
            },
            "structural_gates": {
                "delta7": self.d7,
                "trise": self.trise,
                "vac": self.vac,
                "accum": self.accum,
                "peg_weight": self.w_peg,
                "efference_weight": self.w_eff,
                "relay_efference_weight": self.w_rly_eff,
                "shift_bank": self.sh_bank,
                "ladder_global_inhibition": self.w_lgi,
                "mb_lateral_horn": self.mb_lh,
                "forage_search_veto": self.w_forage_search,
            },
            "metabolic": {
                "sleep": self.metabolic_sleep,
                "forage_low_weight": self.w_forage_low,
                "explore_energy_weight": self.w_explore_energy,
                "energy_afferent_gain": self.metabolic_energy_afferent_gain,
                "hunger_fill_base": self.metabolic_hunger_fill_base,
                "hunger_low_gain": self.metabolic_hunger_low_gain,
                "hunger_gut_gain": self.metabolic_hunger_gut_gain,
                "toxin_gain": self.metabolic_toxin_gain,
                "toxin_threshold": self.metabolic_toxin_threshold,
            },
            "raw": asdict(self),
        }


# Former run_episode behaviour: tonic current 1.0 was injected every neural
# tick.  Making it explicit changes no closed-loop default while making the
# direct tick and live laboratory agree with that loop.
DEFAULT_EMBODIED_CONFIG = EmbodiedAgentConfig()

# The old direct AIFAgent3D.tick()/live-UI default.  This exists only as an
# experimental control for a configuration comparison, never as an implicit
# alternate behaviour path.
LEGACY_DIRECT_TICK_CONFIG = DEFAULT_EMBODIED_CONFIG.with_overrides(
    tonic_amp=0.0,
    tonic_gate=0.30,
)

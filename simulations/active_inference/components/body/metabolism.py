"""V3 metabolic-body transducer.

This is intentionally a contract, not a hidden controller.  The future body
implementation maintains physical/physiological state and exposes only
sanctioned sensory currents to PAULA: gut load, usable energy, and a bounded
metabolic-rate signal.  It must not select FORAGE, EXPLORE, or SLEEP.
"""

from __future__ import annotations

COMPONENT = "body.metabolic_organs"

GUT_LOAD_AFFERENT = "gut_load"
ENERGY_AFFERENT = "energy_store"
METABOLIC_RATE_AFFERENT = "metabolic_state"


def normalized_afferents(world) -> tuple[float, float, float, float]:
    """Return ``(gut, energy, low_energy, digestion_rate)`` from a body.

    The function is a sensor readout only.  It contains no mode selection or
    motor policy; the values are injected into PAULA afferent populations by
    the agent loop.
    """
    gut = float(getattr(world, "gut_load", 0.0))
    energy = float(getattr(world, "energy_store", 1.0))
    digest = float(getattr(world, "digestion_rate", 0.0))
    return gut, energy, max(0.0, 1.0 - energy), digest

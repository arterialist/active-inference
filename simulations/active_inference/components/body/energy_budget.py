"""Explicit energetic accounting for research bodies, not a behavior policy.

Joules and seconds are declared model units. Parameters are illustrative,
not measurements from a particular organism. Positive mechanical work and
activation have separate costs, including activity without net displacement.
Unmet energy is retained as physiological failure debt. It never silently
clips demand or switches the motor off, which would confound neural regulation
with a host-side exhaustion brake. Motion after debt starts is not viable
behavior and must not count as survival or successful rest.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class EnergyBudgetParameters:
    capacity_j: float = 12.
    gut_capacity_j: float = 48.
    digestion_w: float = 3.
    basal_w: float = .2
    efficiency: float = .25
    activation_w: float = .5

    def __post_init__(self):
        if (not all(math.isfinite(v) and v > 0 for v in vars(self).values())
                or self.efficiency > 1):
            raise ValueError('Expected positive finite energetic parameters')


class EnergyBudget:
    fields = ('energy_j', 'gut_j', 'digested_j', 'demand_j', 'unmet_j', 'spill_j', 'ingested_j')

    def __init__(self, *, energy_j=9., gut_j=48., params=None):
        self.params = params or EnergyBudgetParameters()
        if (not math.isfinite(energy_j) or not math.isfinite(gut_j)
                or not 0 <= energy_j <= self.params.capacity_j
                or not 0 <= gut_j <= self.params.gut_capacity_j):
            raise ValueError('Invalid initial organ contents')
        self.energy_j, self.gut_j = float(energy_j), float(gut_j)
        self.digested_j = self.demand_j = self.unmet_j = self.spill_j = self.ingested_j = 0.

    @property
    def energy_store(self):
        return self.energy_j/self.params.capacity_j

    @property
    def gut_load(self):
        return self.gut_j/self.params.gut_capacity_j

    @property
    def digestion_rate(self):
        return self.params.digestion_w/self.params.gut_capacity_j if self.gut_j > 0 else 0.

    def state(self):
        return np.array([getattr(self, name) for name in self.fields])

    def ingest(self, amount_j):
        if not math.isfinite(amount_j) or amount_j < 0:
            raise ValueError('Invalid nutrient amount')
        accepted = min(amount_j, self.params.gut_capacity_j-self.gut_j)
        self.gut_j += accepted; self.ingested_j += accepted
        return accepted

    def advance(self, dt, positive_work_j, activation_squared_dt):
        if (not all(math.isfinite(v) for v in (dt, positive_work_j, activation_squared_dt))
                or dt <= 0 or positive_work_j < 0 or activation_squared_dt < 0):
            raise ValueError('Invalid physical energetic input')
        p = self.params
        digested = min(self.gut_j, p.digestion_w*dt)
        demand = p.basal_w*dt + positive_work_j/p.efficiency + p.activation_w*activation_squared_dt
        available = self.energy_j+digested
        unmet = max(0., demand-available)
        remaining = max(0., available-demand)
        spill = max(0., remaining-p.capacity_j)
        self.energy_j = min(p.capacity_j, remaining)
        self.gut_j -= digested
        self.digested_j += digested; self.demand_j += demand
        self.unmet_j += unmet; self.spill_j += spill
        return digested, demand, unmet, spill

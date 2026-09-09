"""Illustrative movement-to-resource body, with no neural or task policy.

A hinge opens a bounded chamber. Only increasing chamber volume admits fresh
gas; oxygen demand continues during inactivity. This is an ideal exchange
model, not a reconstructed respiratory system: no dead space, pressure load,
perfusion, CO2, or extraction limitation. Millilitres, joules and seconds are
declared illustrative units. The linkage is assumed to add negligible load.
Resource debt records failure without a host-side motor brake.
"""
from dataclasses import dataclass
import math

import numpy as np

from .energy_budget import EnergyBudget, EnergyBudgetParameters
from .loaded_hinge import LoadedHinge, DT, MOTOR_GEAR


@dataclass(frozen=True)
class VentilationParameters:
    chamber_mid_ml: float = 6.
    chamber_swing_ml: float = 5.
    linkage_per_rad: float = 20.
    oxygen_fraction: float = .21
    oxygen_capacity_ml: float = .3
    oxygen_demand_ml_s: float = .3

    def __post_init__(self):
        if (not all(math.isfinite(v) and v > 0 for v in vars(self).values())
                or self.oxygen_fraction > 1
                or self.chamber_swing_ml >= self.chamber_mid_ml):
            raise ValueError('Invalid ventilation parameters')

    def volume(self, angle):
        if not math.isfinite(angle):
            raise ValueError('Nonfinite chamber angle')
        return self.chamber_mid_ml + self.chamber_swing_ml*math.tanh(self.linkage_per_rad*angle)


def research_energy_parameters():
    return EnergyBudgetParameters(capacity_j=.3, gut_capacity_j=2., digestion_w=.12,
                                  basal_w=.02, efficiency=.25, activation_w=.1)


class VentilationOrgans:
    fields = ('oxygen_ml', 'inspired_ml', 'uptake_ml', 'oxygen_demand_ml',
              'oxygen_unmet_ml', 'oxygen_spill_ml') + EnergyBudget.fields
    exchange_fields = ('volume_before_ml', 'volume_after_ml', 'inspired_ml',
                       'uptake_ml', 'oxygen_demand_ml', 'oxygen_unmet_ml',
                       'oxygen_spill_ml', 'positive_work_j', 'activation_squared_dt')

    def __init__(self, *, params=None, oxygen_ml=.15, energy=None):
        self.params = params or VentilationParameters()
        if not math.isfinite(oxygen_ml) or not 0 <= oxygen_ml <= self.params.oxygen_capacity_ml:
            raise ValueError('Invalid oxygen reserve')
        self.oxygen_ml = float(oxygen_ml)
        self.inspired_ml = self.uptake_ml = self.oxygen_demand_ml = 0.
        self.oxygen_unmet_ml = self.oxygen_spill_ml = 0.
        self.energy = energy if energy is not None else EnergyBudget(
            energy_j=.225, gut_j=1.2, params=research_energy_parameters())

    def state(self):
        return np.r_[[getattr(self, n) for n in self.fields[:6]], self.energy.state()]

    def restore(self, state):
        a = np.asarray(state, dtype=float)
        if a.shape != (len(self.fields),) or not np.isfinite(a).all() or np.any(a < 0):
            raise ValueError('Invalid organ state')
        if (a[0] > self.params.oxygen_capacity_ml or a[6] > self.energy.params.capacity_j
                or a[7] > self.energy.params.gut_capacity_j):
            raise ValueError('Organ reserve exceeds capacity')
        for name, value in zip(self.fields[:6], a[:6]):
            setattr(self, name, float(value))
        for name, value in zip(EnergyBudget.fields, a[6:]):
            setattr(self.energy, name, float(value))

    def advance(self, angle_before, angle_after, muscles, *, dt=DT, transmission=1., exchange=1.):
        m = np.asarray(muscles, dtype=float)
        if (m.shape != (2,) or not np.isfinite(m).all() or np.any(m < 0)
                or not math.isfinite(dt) or dt <= 0
                or transmission not in (0., 1.) or exchange not in (0., 1.)):
            raise ValueError('Invalid muscle activity or physical intervention')
        p = self.params
        v0, v1 = p.volume(angle_before), p.volume(angle_after)
        inspired = max(0., v1-v0)
        uptake, demand = exchange*p.oxygen_fraction*inspired, p.oxygen_demand_ml_s*dt
        available = self.oxygen_ml+uptake
        unmet = max(0., demand-available)
        remaining = max(0., available-demand)
        spill = max(0., remaining-p.oxygen_capacity_ml)
        self.oxygen_ml = min(p.oxygen_capacity_ml, remaining)
        self.inspired_ml += inspired; self.uptake_ml += uptake
        self.oxygen_demand_ml += demand; self.oxygen_unmet_ml += unmet
        self.oxygen_spill_ml += spill
        dq = angle_after-angle_before
        # Antagonists are charged separately. Net motor torque would hide
        # positive work and isometric/co-contraction activation expenditure.
        work = transmission*MOTOR_GEAR*(max(0., m[0]*dq)+max(0., -m[1]*dq))
        activation = float(np.dot(m, m))*dt
        self.energy.advance(dt, work, activation)
        return np.array([v0,v1,inspired,uptake,demand,unmet,spill,work,activation])


class VentilatedHinge(LoadedHinge):
    """Existing MuJoCo mechanics plus an explicit, passive resource transducer."""
    def __init__(self, drag=.8, spring=.15, gate=.008, *, organs=None):
        super().__init__(drag, spring, gate)
        self.organs = organs if organs is not None else VentilationOrgans()
        self.last_exchange = np.zeros(len(self.organs.exchange_fields))

    def step(self, command):
        raise TypeError('Use step_muscles: net command alone hides activation cost')

    def step_muscles(self, muscles, *, transmission=1., exchange=1.):
        m = np.asarray(muscles, dtype=float)
        if (m.shape != (2,) or not np.isfinite(m).all() or np.any(m < 0)
                or transmission not in (0., 1.) or exchange not in (0., 1.)):
            raise ValueError('Invalid muscle activity or physical intervention')
        before = float(self.data.qpos[0])
        result = super().step(float(transmission*(m[0]-m[1])))
        self.last_exchange = self.organs.advance(before, float(self.data.qpos[0]), m,
                                                 transmission=transmission, exchange=exchange)
        return result

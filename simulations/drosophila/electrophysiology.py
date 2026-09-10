"""Declared experimental current injection, separate from neural computation.

This instrument adds a predetermined current at PAULA's membrane-current
boundary. It is not a plastic synapse, a controller or a voltage-clamp model.
It does not freeze any cellular parameter or suppress native synaptic events.
The temporary method patch is process-scoped. Run concurrent experiments in
separate processes; do not attach this instrument to a running live server.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np


class CurrentElectrode:
    def __init__(self, cell, current, *, first_tick=0):
        self.cell = cell
        self.command = np.array(current, dtype=float, copy=True)
        if self.command.ndim != 1 or not len(self.command) or not np.isfinite(self.command).all():
            raise ValueError("Electrode command must be a finite nonempty current course")
        if type(first_tick) is not int or first_tick < 0:
            raise ValueError("first_tick must be a nonnegative integer")
        self.first_tick = first_tick
        self.native_current = np.zeros_like(self.command)
        self.total_current = np.zeros_like(self.command)
        self.count = 0
        self._patch = None

    def __enter__(self):
        if self._patch is not None:
            raise RuntimeError("Electrode cannot be reused")
        original = type(self.cell)._hillock_current

        def inject(cell, tick, dt):
            if cell is not self.cell:
                return original(cell, tick, dt)
            row = tick-self.first_tick
            if row != self.count or not 0 <= row < len(self.command):
                raise ValueError("Electrode requires consecutive commanded ticks")
            native = original(cell, tick, dt)
            # A zero command preserves the ordinary numeric type as well as
            # the value; a measurement-only electrode must not change rounding.
            total = native if self.command[row] == 0 else native+self.command[row]
            self.native_current[row] = native
            self.total_current[row] = total
            self.count += 1
            return total

        self._patch = patch.object(type(self.cell), "_hillock_current", inject)
        self._patch.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._patch.__exit__(exc_type, exc_value, traceback)
        if exc_type is None and self.count != len(self.command):
            raise ValueError("Incomplete electrode course")
        return False

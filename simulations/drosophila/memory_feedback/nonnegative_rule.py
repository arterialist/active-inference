"""Apply PAULA's existing w_min=0 boundary only to selected memory inputs.

The first gamma4 composition crossed zero on three weak KC inputs at tick 216,
then the selected-rule sign guard stopped the run at tick 219. Global per-cell
w_min=0 would also silence inhibitory reciprocal inputs. This subclass uses the
same hard-bound semantics after the existing selected-port update, leaving all
other inputs and native returning adaptation unchanged. Zeroed ports can recover
under the additive rule; there is no positive weight floor or reset schedule.
"""
from .input_rule import MemoryInputRuleNeuron


class NonnegativeMemoryInputNeuron(MemoryInputRuleNeuron):
    def tick(self, external_inputs, current_tick, dt=1.):
        events = super().tick(external_inputs, current_tick, dt)
        hits = 0
        for sid in self.metadata.get("memory_rule_ports", ()):
            point = self.postsynaptic_points[sid]
            if point.u_i.info < 0:
                point.u_i.info = 0.
                hits += 1
        if hits:
            self.metadata["memory_floor_hits"] = self.metadata.get("memory_floor_hits", 0)+hits
            self.metadata.setdefault("memory_floor_first_tick", current_tick)
        return events

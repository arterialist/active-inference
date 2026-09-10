"""Mechanism constraints used to choose the next memory-transfer comparison."""
import numpy as np

from simulations.drosophila.memory_feedback.nonnegative_rule import NonnegativeMemoryInputNeuron
from neuron.neuron import NeuronParameters, PostsynapticPoint, PostsynapticInputVector


def test_existing_selected_rule_has_no_delayed_credit_without_new_arrival():
    cell = NonnegativeMemoryInputNeuron(1, NeuronParameters(num_inputs=1, eta_post=1e-5, eta_retro=1e-6),
        metadata={"memory_rule_ports": [0]}, log_level="CRITICAL")
    cell.params.nm_plasticity_kappa = -10.
    cell.postsynaptic_points[0] = PostsynapticPoint(PostsynapticInputVector(info=.1, plast=0., adapt=np.zeros(2)))
    cell.distances[0] = 2
    cell.input_buffer[0, 0] = 1.
    cell.tick({}, 0)
    after_arrival = cell.postsynaptic_points[0].u_i.info
    assert after_arrival != .1
    for tick in range(1, 21):
        cell.tick({}, tick)
    cell.M_vector[1] = 1.  # Isolate a late local receptor state in this unit probe.
    for tick in range(21, 41):
        cell.tick({}, tick)
    assert cell.M_vector[1] > 0.
    assert cell.postsynaptic_points[0].u_i.info == after_arrival
    assert cell.params.eta_post > 0 and cell.params.eta_retro > 0

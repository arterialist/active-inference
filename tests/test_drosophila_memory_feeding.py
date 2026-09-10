"""Physical benefit must depend on contact, never on a cue or target action."""
import numpy as np

from simulations.drosophila.memory_feedback.feeding import FeedingBody, WELL_ANGLE, DOSE_J


def test_food_well_requires_existing_physical_contact_and_available_food():
    organism = FeedingBody()
    assert organism.offer(False, True) == (0., 0.)
    organism.body.data.qpos[0] = WELL_ANGLE
    assert organism.offer(False, False) == (0., 0.)
    before = organism.organs.gut_j
    assert organism.offer(False, True) == (0., DOSE_J)
    assert organism.organs.gut_j == before+DOSE_J
    organism.body.data.qpos[0] = 0.
    assert organism.offer(True, False) == (DOSE_J, 0.)


def test_physical_checkpoint_preserves_continuing_contact_and_motor_response(tmp_path):
    a = FeedingBody()
    for tick in range(100):
        doses = a.offer(False, True)
        a.step(float(tick % 10 == 0), *doses)
    path = tmp_path/'body.npz'; a.save(path)
    b = FeedingBody(); b.restore(path)
    for tick in range(50):
        ad = a.offer(False, True); bd = b.offer(False, True)
        assert ad == bd
        np.testing.assert_array_equal(a.step(float(tick % 7 == 0), *ad),
                                      b.step(float(tick % 7 == 0), *bd))


def test_memory_zero_bound_preserves_inhibitory_learning_and_allows_recovery():
    from neuron.neuron import NeuronParameters, PostsynapticPoint, PostsynapticInputVector
    from simulations.drosophila.memory_feedback.nonnegative_rule import NonnegativeMemoryInputNeuron
    cell = NonnegativeMemoryInputNeuron(1,NeuronParameters(num_inputs=2,eta_post=.01,eta_retro=1e-6),
        metadata={"memory_rule_ports":[0]},log_level="CRITICAL")
    for sid,weight in enumerate((.001,-.5)):
        cell.postsynaptic_points[sid]=PostsynapticPoint(PostsynapticInputVector(info=weight,plast=0.,adapt=np.zeros(2)))
        cell.distances[sid]=0
        cell.input_buffer[sid,0]=1.
    cell.tick({},0)
    assert cell.postsynaptic_points[0].u_i.info == 0.
    assert cell.postsynaptic_points[1].u_i.info < 0.
    assert cell.postsynaptic_points[1].u_i.info != -.5
    cell.t_last_fire=1
    cell.input_buffer[:,0]=1.
    cell.tick({},1)
    assert cell.postsynaptic_points[0].u_i.info > 0.

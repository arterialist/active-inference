import numpy as np
import pytest

from simulations.drosophila import orn_onset_analysis as analysis
from simulations.drosophila.connectome import Subgraph


def test_first_spike_attribution_respects_synaptic_and_membrane_delays(monkeypatch):
    monkeypatch.setattr(analysis, "WATCH", ("2",))
    nodes = {r: {"global_index": i, "annotation": {"root_id": r,
        "hemibrain_type": "test", "cell_class": "ALLN"}} for i, r in enumerate(("1", "2"))}
    g = Subgraph(("1", "2"), nodes, np.array([[1, 2, 0, 1, 400, 1, 400, 0, 0]], dtype=np.int64), {})
    # One receiving event at 3 reaches the hillock at 5; soma crosses 1.
    a = {"inputs": np.zeros((10, 2, 4)), "weights": np.tile([30., 1.], (10, 1)),
        "current": np.zeros((10, 1)), "soma": np.zeros((10, 2, 3)),
        "intrinsic": np.tile([30., 1., 1.2, 20.], (10, 1, 1))}
    a["inputs"][3, 0, 0] = 1
    a["current"][5, 0] = 30.*.95**2
    a["soma"][5, 1, 1] = 1
    s = {"roots": np.array(["1", "2"]), "watch_offsets": np.array([0, 2])}
    out = analysis.input_witnesses(g, {"intervention": None}, s, a)[0]
    assert out["first_spike"] == 5
    assert out["ports"][0]["first_receiving_tick"] == 3
    assert out["ports"][0]["voltage_contribution_at_first_spike"] == pytest.approx(30*.95**2/20)
    a["current"][5, 0] += .01
    with pytest.raises(ValueError, match="float32 bound"):
        analysis.input_witnesses(g, {"intervention": None}, s, a)

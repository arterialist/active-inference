"""Tests of the actual LH cells, not a surrogate logical AND function."""
import unittest

from simulations.active_inference import aif_agent3d as ag
from simulations.active_inference.live.versions import get_version


def probe(left, right, aversion, weight=0.04):
    agent = ag.AIFAgent3D(seed=11, components=get_version("v2").components,
                        config=ag.DEFAULT_EMBODIED_CONFIG.with_overrides(w_lh_sensor=weight))
    counts = [0, 0]
    for t in range(256):
        for group, ids in enumerate((ag.LH_LEFT, ag.LH_RIGHT)):
            for nid in ids:
                n = agent.nb[nid]
                n.input_buffer.fill(0)
                for sid, (source, terminal) in n.synapse_sources.items():
                    amplitude = left if source in ag.FLp else right if source in ag.FRp else aversion
                    # Deliver at dendrites, with inherited dendritic propagation
                    # and membrane dynamics. No recorded output is prescribed.
                    n.input_buffer[sid, 0] = amplitude if t % 2 == 0 else 0
                n.tick({}, t)
                counts[group] += int(n.O > 0)
    return counts


class ValenceConvergence(unittest.TestCase):
    def test_original_weight_allows_odor_only_avoidance(self):
        self.assertGreater(probe(1, 0, 0, weight=1.2)[0], 0)

    def test_subthreshold_sensory_route_requires_aversion(self):
        self.assertEqual(probe(1, 0, 0), [0, 0])
        self.assertEqual(probe(0, 1, 0), [0, 0])
        left = probe(1, 0, 1)
        right = probe(0, 1, 1)
        self.assertGreater(left[0], left[1])
        self.assertGreater(right[1], right[0])
        self.assertEqual(left, right[::-1])


if __name__ == "__main__":
    unittest.main()

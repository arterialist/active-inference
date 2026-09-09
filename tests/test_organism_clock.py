"""Caller scheduling must not choose the organism's sensorimotor dynamics."""
import unittest

from simulations.active_inference import aif_agent3d as ag
from simulations.active_inference.live.versions import VERSIONS


def trajectory(version, sub, repeated=False):
    agent = ag.AIFAgent3D(seed=11, components=VERSIONS[version].components)
    agent.world.foods = [[-0.6, 0]] + [[100, 100] for _ in range(6)]
    agent.world.toxins = [[100, 100] for _ in range(6)]
    agent.world.respawn_food = False
    agent.world._sync_mocap()
    agent.birth()
    rows = []

    def observe(current):
        rows.append((tuple(current.world.data.qpos), tuple(current.world.data.qvel),
                     tuple((nid, float(n.S), float(n.O), tuple(n.M_vector))
                           for nid, n in sorted(current.nb.items())),
                     tuple(sorted(current.last_sensor_drives.items()))))

    for _ in range(64 // sub if repeated else 1):
        ag.run_episode(agent, steps=1 if repeated else 64 // sub, sub=sub,
                       vision=False, log_every=10**9, tick_hook=observe)
    return rows


class OrganismClock(unittest.TestCase):
    def test_batching_and_call_boundaries_preserve_every_tick(self):
        for version in VERSIONS:
            with self.subTest(version=version):
                expected = trajectory(version, 1)
                self.assertEqual(expected, trajectory(version, 16))
                self.assertEqual(expected, trajectory(version, 16, repeated=True))

    def test_versions_do_not_claim_acceptance_without_evidence(self):
        self.assertTrue(all(not v.accepted for v in VERSIONS.values()))


if __name__ == "__main__":
    unittest.main()

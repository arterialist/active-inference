"""Independent worms and reset trials must not share motor dynamics state."""

from types import SimpleNamespace

import pytest

from simulations.c_elegans.neuron_mapping import CElegansNervousSystem
from simulations.connectome_loader import ConnectomeData


class Network:
    def __init__(self):
        self.resets = 0
        self.network = SimpleNamespace(neurons={0: SimpleNamespace(S=0.0)})

    def reset_simulation(self):
        self.resets += 1


@pytest.fixture
def build_stub(monkeypatch):
    builds = []

    def build(nervous_system):
        builds.append(nervous_system)
        nervous_system._network = Network()
        nervous_system._name_to_id = {"AVBL": 0}

    monkeypatch.setattr(CElegansNervousSystem, "_build", build)
    return builds


def nervous_system():
    return CElegansNervousSystem(ConnectomeData([], [], []))


def test_command_noise_and_proprioception_are_owned_by_each_worm(build_stub):
    first, second = nervous_system(), nervous_system()
    assert first._cmd_noise_state is not second._cmd_noise_state
    assert first._proprio_ema is not second._proprio_ema
    first._inject_command_noise()
    first._proprio_ema["DB1"] = 0.4
    assert "AVBL" in first._cmd_noise_state
    assert second._cmd_noise_state == second._proprio_ema == {}
    noise_before_reset = dict(first._cmd_noise_state)
    second.reset()
    assert first._cmd_noise_state == noise_before_reset
    assert first._proprio_ema == {"DB1": 0.4}


@pytest.mark.parametrize("rebuild_network", [False, True])
def test_reset_clears_noise_proprioception_and_reversal_latch(build_stub, rebuild_network):
    nervous = nervous_system()
    network_before = nervous._network
    nervous._cmd_noise_state["AVBL"] = 0.7
    nervous._proprio_ema["DB1"] = 0.3
    nervous._rev_state = "active"
    nervous._rev_state_until_tick = 900
    nervous._rev_current_tick = 700
    nervous._muscle_activations["seg0_DL"] = 0.4
    nervous.reset(rebuild_network=rebuild_network)
    assert nervous._cmd_noise_state == nervous._proprio_ema == {}
    assert nervous._rev_state == "idle"
    assert nervous._rev_state_until_tick == nervous._rev_current_tick == 0
    assert all(value == 0.0 for value in nervous._muscle_activations.values())
    if rebuild_network:
        assert nervous._network is not network_before
        assert len(build_stub) == 2
    else:
        assert nervous._network is network_before
        assert network_before.resets == 1
        assert len(build_stub) == 1

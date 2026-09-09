from copy import deepcopy
import numpy as np
import pytest

from simulations.active_inference.experiments.evidence_embedded import (
    embedded_config, cell_factory, full_snapshot, record, compare_lower,
)
from simulations.active_inference.experiments.association_evidence_population import (
    evidence_config, record as record_bank,
)
from simulations.active_inference.experiments.eligibility_association_probe import protocol
from simulations.active_inference.experiments.evidence_consumer import consumer_config
from simulations.active_inference.experiments.evidence_consumer_audit import ConsumerAudit
from simulations.active_inference.experiments.evidence_embedded_audit import SourceTerminalAudit
from simulations.active_inference.experiments.multimodal_pairing_probe import fresh
from simulations.active_inference.experiments.composition_probe import encode
from neuron.extensions.experimental.port_modulation import PortModulationNeuron


def test_embedding_changes_only_the_declared_projection():
    old, _ = evidence_config(11)
    new, groups = embedded_config(11)
    assert len(new['neurons']) == 433
    assert new['neurons'][:368] == old['neurons']
    assert new['synaptic_points'][:len(old['synaptic_points'])] == old['synaptic_points']
    assert new['connections'][:len(old['connections'])] == old['connections']
    assert new['external_inputs'] == old['external_inputs']
    assert len(new['connections'])-len(old['connections']) == 288
    assert groups['contrast'] == list(range(402, 434))


def test_embedded_observer_is_passive_and_consumer_audit_uses_actual_releases(tmp_path):
    cfg, groups = embedded_config(11)
    path = tmp_path/'connected.json'; path.write_text(encode(cfg))
    net, *_ = fresh(path, 11, cell_factory)
    plain = deepcopy(net)
    oldcfg, oldgroups = evidence_config(11)
    oldpath = tmp_path/'source.json'; oldpath.write_text(encode(oldcfg))
    old, *_ = fresh(oldpath, 11, PortModulationNeuron)
    masks, trials = protocol(groups, 11, 'paired', 1)
    auditor = ConsumerAudit(consumer_config('evidence_slow'))
    source_auditor = SourceTerminalAudit()
    for trial in trials:
        lower, upper = record(net, groups, masks, trial)
        reference = record_bank(old, oldgroups, masks, trial)
        compare_lower(lower, reference)
        before_source_audit = deepcopy(source_auditor)
        source_auditor.check(lower, upper)
        auditor.check(upper)
        for t in range(trial['ticks']):
            if t < 32 and t % 8 == 0:
                for role, key in (('vision', 'cue'), ('audio', 'sound')):
                    for nid in masks[role][trial[key]]: plain.set_external_input(nid, 0, 1.)
            plain.run_tick()
        assert full_snapshot(net) == full_snapshot(plain)
    assert lower['bank_terminal_info'].min() < 1
    assert np.array_equal(upper['incoming'][1:, :192], upper['source'][:-1].astype(np.float32))
    bad = deepcopy(lower); bad['bank_terminal_info'][0,0] += .01
    with pytest.raises(ValueError, match='return update'):
        before_source_audit.check(bad, upper)

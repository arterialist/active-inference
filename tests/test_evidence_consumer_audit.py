from copy import deepcopy
import numpy as np
import pytest
from simulations.active_inference.experiments.evidence_consumer import consumer_config, record
from simulations.active_inference.experiments.evidence_consumer_audit import ConsumerAudit
from simulations.active_inference.experiments.multimodal_pairing_probe import fresh
from simulations.active_inference.experiments.composition_probe import encode
from neuron.extensions.graded import GradedNeuron


@pytest.fixture
def recording(tmp_path):
    cfg = consumer_config('evidence_slow')
    path = tmp_path/'config.json'; path.write_text(encode(cfg))
    net, *_ = fresh(path, 11, GradedNeuron)
    tape = np.zeros((64, 192), bool); tape[[0, 8, 16, 24], :96] = True
    return cfg, record(net, tape)


def test_independent_consumer_audit_matches_native_trace(recording):
    cfg, raw = recording
    audit = ConsumerAudit(cfg)
    assert audit.check(raw).shape == (64, 65, 2)
    assert audit.max_error < 2e-12


@pytest.mark.parametrize('field,index', [
    ('states', (3, 33, 0)), ('states', (3, 33, 5)), ('incoming', (1, 0)), ('before', (1, 0)),
    ('after', (1, 0)), ('terminal_info', (4, 0)), ('source', (0, 0)),
])
def test_corrupted_consumer_ledger_is_rejected(recording, field, index):
    cfg, original = recording; raw = deepcopy(original)
    raw[field][index] = 0 if field == 'source' else raw[field][index]+.1
    with pytest.raises(ValueError): ConsumerAudit(cfg).check(raw)

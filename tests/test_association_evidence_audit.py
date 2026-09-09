from copy import deepcopy
import numpy as np
import pytest
from simulations.active_inference.experiments.association_evidence_audit import EvidenceAudit
from simulations.active_inference.experiments.association_evidence_population import evidence_config,record
from simulations.active_inference.experiments.eligibility_association_probe import protocol
from simulations.active_inference.experiments.composition_probe import encode
from simulations.active_inference.experiments.multimodal_pairing_probe import fresh
from neuron.extensions.experimental.port_modulation import PortModulationNeuron


@pytest.fixture
def acquired(tmp_path):
    cfg,g=evidence_config(11);p=tmp_path/'config.json';p.write_text(encode(cfg))
    net,*_=fresh(p,11,PortModulationNeuron);m,trials=protocol(g,11,'paired',1)
    return cfg,g,record(net,g,m,trials[0])


def test_independent_soma_and_both_plasticity_rules(acquired):
    cfg,g,raw=acquired;a=EvidenceAudit(cfg,g);states=a.check(raw)
    assert states.shape==(96,192,3) and a.max_error<2e-12
    assert np.any(raw['evidence_before'][:,:,:32]!=raw['evidence_after'][:,:,:32])
    assert np.any(raw['evidence_before'][:,:,35:]!=raw['evidence_after'][:,:,35:])


def test_audit_rejects_somatic_and_learning_corruption(acquired):
    cfg,g,raw=acquired
    for field,index in [('states',(4,176,0)),('evidence_after',(4,0,0)),('evidence_after',(4,0,35))]:
        corrupt=deepcopy(raw);corrupt[field][index]+=.001
        with pytest.raises(ValueError):EvidenceAudit(cfg,g).check(corrupt)

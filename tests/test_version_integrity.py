from simulations.active_inference.experiments.version_integrity_audit import audit
from simulations.active_inference.experiments.version_evidence_validator import validate


def test_version_integrity_audit_is_fail_closed():
    result = audit()
    assert result["passed"], result["failures"]
    assert result["fail_closed_contract"]["strict_topology"] is True
    assert result["fail_closed_contract"]["selected_version_forwarded"] is True
    assert result["fail_closed_contract"]["independent_raw_trace_recomputation"] is True
    assert result["fail_closed_contract"]["unreachable_control_is_inconclusive"] is True
    assert result["versions"]["v3"]["direct_build_parameters"]["w_unc_mode"] == 0.8
    assert result["versions"]["v3"]["entrypoint_build_parameters"]["w_unc_mode"] == 0.8


def test_strict_harnesses_cover_every_version_and_have_environment_suites():
    result = audit()
    harnesses = result["versions"]["harnesses"]
    assert {version for spec in harnesses.values() if spec["version_policy"] == "strict_agent" for version in spec["supported_versions"]} >= {"v1", "v2", "v3", "v4"}
    assert len(harnesses["food_collection"]["worlds"]) == 11
    assert len(harnesses["toxin_escape"]["worlds"]) == 11
    assert harnesses["food_collection"]["worlds"][-1] == "all"
    assert harnesses["toxin_escape"]["worlds"][-1] == "all"


def test_validator_rejects_an_endpoint_only_or_incomplete_directory(tmp_path):
    (tmp_path / "manifest.json").write_text('{"experiment":"x"}\n')
    result = validate(tmp_path)
    assert result["valid"] is False
    assert result["failures"]

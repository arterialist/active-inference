"""Require the complete causal chain before reporting the bounded positive result."""
import argparse
import json
from pathlib import Path

import numpy as np

from .coverage_analysis import run as audit
from ..connectome import sha256


def run(base,output):
    base,output=Path(base),Path(output)
    evidence=audit(base,output)
    direct=evidence["expression"]
    assert direct["sham"]["exact_sham"]
    assert direct["sham"]["well_j"]>direct["erased"]["well_j"]
    assert np.isclose(direct["transferred"]["well_j"],direct["sham"]["well_j"],rtol=0,atol=1e-12)
    assert direct["sham"]["stored_energy_gain_J"]>direct["erased"]["stored_energy_gain_J"]
    courses=evidence["continuation"]
    names=("eight_intact","eight_cut","eight_teacher_removed","eight_unpaired","eight_displaced","eight_alpha_removed")
    assert all(n in courses for n in names)
    learned=courses["eight_intact"]["probes"]["B"]["spikes"]["SMP108"]
    control=courses["eight_cut"]["probes"]["B"]["spikes"]["SMP108"]
    assert learned>control
    for name in names:
        r=courses[name]
        assert r["nutrient_j"]==0 and r["minimum_eta_post"]>0 and r["minimum_eta_retro"]>0
        assert r["phases"][-1]["name"]=="retention" and r["phases"][-1]["end"]-r["phases"][-1]["begin"]==1000
        assert r["probes"]["B"]["changed_weights"]>0
        w=np.load(base/r["record"]/"weights.npy",mmap_mode="r")
        assert np.any(w[0]!=w[-1])
    for name in ("eight_teacher_removed","eight_unpaired","eight_displaced"):
        assert courses[name]["probes"]["B"]["spikes"]["SMP108"]==control
    assert courses["eight_intact"]["probes"]["A"]["well_j"]>direct["erased"]["well_j"]
    assert np.isclose(courses["eight_cut"]["probes"]["A"]["well_j"],courses["eight_intact"]["probes"]["A"]["well_j"],rtol=0,atol=1e-12)
    assert evidence["teacher_intervention"]["all_other_runtime_state_exact"]
    assert evidence["teacher_intervention"]["cue"]=="A"
    assert set(evidence["teacher_intervention"]["targets"])=={"MBON07","MBON04"}
    probes=evidence["diagnostics"]["eight"]["probes"]
    assert probes["removed"]["spikes"]["SMP108"]==control
    assert probes["transferred"]["spikes"]["SMP108"]==learned
    assert probes["sham"]["original_replay_exact"] and probes["blocked_sham"]["original_replay_exact"]
    chain=evidence["teacher_chain"]["probes"]
    for name in ("teacher-into-intact-dry","unpaired-into-intact-dry","displaced-into-intact-dry"):
        assert chain[name]["spikes"]["SMP108"]==control
    assert chain["intact-into-teacher-dry"]["spikes"]["SMP108"]==learned
    food=evidence["B_feeding"]
    assert food["sham"]["well_j"]>food["removed"]["well_j"]
    assert food["transferred"]["well_j"]>food["blocked-sham"]["well_j"]
    assert food["sham"]["stored_energy_gain_J"]>food["removed"]["stored_energy_gain_J"]
    assert chain["intact-into-teacher-food"]["well_j"]>chain["teacher-sham-food"]["well_j"]
    assert chain["intact-into-teacher-food"]["stored_energy_gain_J"]>chain["teacher-sham-food"]["stored_energy_gain_J"]
    assert np.isclose(chain["teacher-into-intact-food"]["well_j"],food["removed"]["well_j"],rtol=0,atol=1e-12)
    specificity=evidence["cue_specificity"]
    assert specificity["C_body_exact"] and specificity["C_soma_exact"]
    anatomy={}
    for f in ("edges.npz","nodes.json"):
        digest=sha256(base/"memory-balanced-codes-20260910"/f)
        assert digest==sha256(base/"memory-terminal-cut-20260910"/f)
        anatomy[f]=digest
    for birth in evidence["birth"].values():
        assert birth["all_other_runtime_state_exact"] and birth["body_exact"]
        assert birth["anatomy"]["internal_connections_exact"] and birth["anatomy"]["source_registration_exact"]
    manifest=json.loads((base/"memory-coverage-paired-20260910/manifest.json").read_text())
    assumptions=manifest["assumptions"]
    alpha_spikes=courses["eight_alpha_removed"]["probes"]["B"]["spikes"]["SMP108"]
    result=dict(source_sha256=sha256(Path(__file__)),full_audit_sha256=sha256(output/"coverage-acquisition.json"),
        conclusion="In this declared engineered preparation, acquired A terminal memory causes a retained, useful B response through measured neural feedback while adaptation continues.",
        first_order=dict(paired_A_food_J=direct["sham"]["well_j"],A_memory_removed_food_J=direct["erased"]["well_j"],
                         A_memory_transferred_food_J=direct["transferred"]["well_j"],A_food_after_second_order_J=courses["eight_intact"]["probes"]["A"]["well_j"]),
        second_order=dict(pairings=8,nutrient_J=0,retention_ticks=1000,physical_seconds_per_tick=manifest["physical_seconds_per_tick"],
            B_dry_spikes={n:courses[n]["probes"]["B"]["spikes"]["SMP108"] for n in names},
            B_food_J={n:r["well_j"] for n,r in food.items()},
            same_state_food_benefit_J=food["sham"]["well_j"]-food["removed"]["well_j"],
            same_state_stored_energy_benefit_J=food["sham"]["stored_energy_gain_J"]-food["removed"]["stored_energy_gain_J"],
            first_contact_ticks={n:r["first_contact_tick"] for n,r in food.items()},
            teacher_memory_and_B_memory_chain_verified=True,held_out_C_unchanged=True),
        alpha1_interpretation=("Removing alpha1 A-terminal release memory alone leaves the extra B response. This does not establish exclusively alpha1-to-gamma4 teaching; primary training also trained gamma4 A terminals."
            if alpha_spikes==learned else "Removing alpha1 A-terminal release memory changes the B response in this preparation; gamma4 A memory was also acquired during primary training."),
        anatomy=dict(unchanged_source_files=anatomy,executed_internal_pairs=4218,selected_cells=276,
                     incoming_ports_including_boundary_and_experimental=50940,outgoing_terminals_including_boundary=41028),
        boundaries=dict(manifest_sha256=sha256(base/"memory-coverage-paired-20260910/manifest.json"),
            native_parameters=assumptions["parameters"],boundary_condition=assumptions["boundary_condition"],
            experimental_input=assumptions["experimental_input"],student_interface=assumptions["student_interface"],
            terminal_credit={k:assumptions["terminal_credit"][k] for k in ("enabled","tau_kc","tau_dopamine","eta_credit","native","dopamine","limit")},
            primary_roles=assumptions["primary_boundary"]["roles"],primary_current_per_dose=assumptions["primary_boundary"]["current_per_dose"],
            feedback_efficacy=assumptions["coverage_efficacy"]["total_birth_gain"],output_sensitivity_factor=manifest["factor"],
            student_output_threshold_factor=assumptions["student_output"]["factor"],well_angle=manifest["well_angle"],
            motor_gain=manifest["motor_gain"],dose_J=manifest["dose_j"]),
        scope_limits=["One deterministic seed and one balanced controlled cue code assignment; no claim of generalization across odors, animals or parameter choices.",
            "One extra dry output spike and one food quantum, with one-tick earlier contact. This is a small model effect, not a large behavioral improvement.",
            "Equal cue currents, nutrient-to-PAM current, output sensitivity, source efficacy 256, terminal-credit kinetics and the SMP108-to-hinge transducer are engineering choices, not measured fly physiology.",
            "Second-order conditioning already exists in biology and other models; this establishes a composition in this PAULA preparation, not a new biological discovery."])
    (output/"causal-transfer-verdict.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("base",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();run(a.base,a.output)

"""List parameter fields flagged as written but not read (static audit)."""
from __future__ import annotations

# All specs mapped from the parameter files
specs_by_file = {
    'simulation_params.py': {
        # Simulation control
        'engine.neural_ticks_per_physics_step': 'read_in_engine',  # Used in engine.py
        # Neuromodulation
        'ns._K_STRESS_SYN': 'read_in_neuron_mapping',
        'ns._K_REWARD_SYN': 'read_in_neuron_mapping',
        'ns._K_VOL_STRESS': 'read_in_neuron_mapping',
        'ns._K_VOL_REWARD': 'read_in_neuron_mapping',
        'ns._STRESS_DEADZONE': 'read_in_neuron_mapping',
        'ns._CHEM_EMA_ALPHA_FAST': 'read_in_neuron_mapping',
        'ns._CHEM_EMA_ALPHA_SLOW': 'read_in_neuron_mapping',
        'ns._TONIC_FWD_CMD': 'read_in_neuron_mapping',
        'ns._TONIC_FWD_MOTOR': 'read_in_neuron_mapping',
        'ns._K_OFF_SUPPRESS': 'read_in_neuron_mapping',
        'ns._TONIC_OFF_CELL': 'read_in_neuron_mapping',
        'ns._PROPRIO_MOTOR_GAIN': 'read_in_neuron_mapping',
        'ns._PROPRIO_TAIL_DECAY': 'read_in_neuron_mapping',
        'ns._HEAD_CPG_FREQ_HZ': 'read_in_neuron_mapping',
        'ns._HEAD_CPG_AMP': 'read_in_neuron_mapping',
        'ns._enable_m0': 'read_in_neuron_mapping',
        'ns._enable_m1': 'read_in_neuron_mapping',
        # Environment / rebuild
        'aec.ENV_PLATE_RADIUS_M': 'read_in_body_build',
        'aec.WALL_HEIGHT_M': 'read_in_body_build',
        'aec.WALL_FRICTION_TANGENT': 'read_in_body_build',
        'aec.WALL_SEGMENTS_N': 'read_in_body_build',
        'aec.BOUNDARY_TELEPORT_FACTOR': 'read_at_runtime',
        'aec.FAKE_WALL_OBSERVATION': 'read_at_runtime',
        'aec.JOINT_ANGLE_MAX_RAD': 'read_at_reset',
        # Sensorimotor
        'ns._muscle_filter_alpha': 'read_in_neuron_mapping',
        'ns._nmj_scale': 'DEAD_NOT_READ',
        'ns._nmj_threshold': 'DEAD_NOT_READ',
        # Thermodynamics
        'ns._gap_diffusion_rate': 'DEAD_NOT_READ',
        'ns._tonic_metabolic_heat': 'DEAD_NOT_READ',
    }
}

dead_specs = []
for fname, specs_dict in specs_by_file.items():
    for target, status in specs_dict.items():
        if status == "DEAD_NOT_READ":
            dead_specs.append((fname, target))


def main() -> None:
    print("DEAD CODE SPECS (setter writes but nothing reads):\n")
    for fname, target in dead_specs:
        # Extract the attribute name for cleaner output
        if "." in target:
            module, attr = target.rsplit(".", 1)
            print(f"  {attr:<25} in {target}")
        else:
            print(f"  {target}")

    print(f"\nTotal dead specs found: {len(dead_specs)}")


if __name__ == "__main__":
    main()

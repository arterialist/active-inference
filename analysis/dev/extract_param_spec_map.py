"""Print the mapping from lab ``sim.*`` paths to engine / namespace attributes."""
from __future__ import annotations

# simulation_params.py specs
sim_specs = {
    "sim.neural_ticks_per_physics": "engine.neural_ticks_per_physics_step",
    # Neuromod specs (from the NEUROMOD_SPECS list)
    "sim.neuromod.K_STRESS_SYN": "ns._K_STRESS_SYN",
    "sim.neuromod.K_REWARD_SYN": "ns._K_REWARD_SYN",
    "sim.neuromod.K_VOL_STRESS": "ns._K_VOL_STRESS",
    "sim.neuromod.K_VOL_REWARD": "ns._K_VOL_REWARD",
    "sim.neuromod.STRESS_DEADZONE": "ns._STRESS_DEADZONE",
    "sim.neuromod.CHEM_EMA_ALPHA_FAST": "ns._CHEM_EMA_ALPHA_FAST",
    "sim.neuromod.CHEM_EMA_ALPHA_SLOW": "ns._CHEM_EMA_ALPHA_SLOW",
    "sim.neuromod.TONIC_FWD_CMD": "ns._TONIC_FWD_CMD",
    "sim.neuromod.TONIC_FWD_MOTOR": "ns._TONIC_FWD_MOTOR",
    "sim.neuromod.K_OFF_SUPPRESS": "ns._K_OFF_SUPPRESS",
    "sim.neuromod.TONIC_OFF_CELL": "ns._TONIC_OFF_CELL",
    "sim.neuromod.PROPRIO_MOTOR_GAIN": "ns._PROPRIO_MOTOR_GAIN",
    "sim.neuromod.PROPRIO_TAIL_DECAY": "ns._PROPRIO_TAIL_DECAY",
    "sim.neuromod.HEAD_CPG_FREQ_HZ": "ns._HEAD_CPG_FREQ_HZ",
    "sim.neuromod.HEAD_CPG_AMP": "ns._HEAD_CPG_AMP",
    "sim.neuromod.enable_m0": "ns._enable_m0",
    "sim.neuromod.enable_m1": "ns._enable_m1",
    # Environment specs
    "sim.env.plate_radius": "aec.ENV_PLATE_RADIUS_M",
    "sim.env.wall_height_m": "aec.WALL_HEIGHT_M",
    "sim.env.wall_friction": "aec.WALL_FRICTION_TANGENT",
    "sim.env.wall_segments_n": "aec.WALL_SEGMENTS_N",
    "sim.env.boundary_teleport_factor": "aec.BOUNDARY_TELEPORT_FACTOR",
    "sim.env.fake_wall_obs": "aec.FAKE_WALL_OBSERVATION",
    # Sensorimotor specs
    "sim.joints.angle_max": "aec.JOINT_ANGLE_MAX_RAD",
    "sim.muscles.filter_alpha": "ns._muscle_filter_alpha",
    "sim.muscles.nmj_scale": "ns._nmj_scale",
    "sim.muscles.nmj_threshold": "ns._nmj_threshold",
    # Thermodynamics specs
    "sim.thermo.gap_diffusion_rate": "ns._gap_diffusion_rate",
    "sim.thermo.tonic_metabolic_heat": "ns._tonic_metabolic_heat",
}

# mujoco_engine_params specs (all read from mjOption)
mj_specs = {
    "sim.mujoco.opt.timestep": "opt.timestep",
    "sim.mujoco.opt.integrator": "opt.integrator",
    "sim.mujoco.opt.impratio": "opt.impratio",
    "sim.mujoco.opt.cone": "opt.cone",
    "sim.mujoco.opt.jacobian": "opt.jacobian",
    "sim.mujoco.opt.solver": "opt.solver",
    "sim.mujoco.opt.iterations": "opt.iterations",
    "sim.mujoco.opt.noslip_iterations": "opt.noslip_iterations",
    "sim.mujoco.opt.sdf_iterations": "opt.sdf_iterations",
    "sim.mujoco.opt.ccd_iterations": "opt.ccd_iterations",
    "sim.mujoco.opt.ls_iterations": "opt.ls_iterations",
    "sim.mujoco.opt.sdf_initpoints": "opt.sdf_initpoints",
    "sim.mujoco.opt.tolerance": "opt.tolerance",
    "sim.mujoco.opt.noslip_tolerance": "opt.noslip_tolerance",
    "sim.mujoco.opt.ls_tolerance": "opt.ls_tolerance",
    "sim.mujoco.opt.ccd_tolerance": "opt.ccd_tolerance",
    "sim.mujoco.opt.sleep_tolerance": "opt.sleep_tolerance",
    "sim.mujoco.opt.density": "opt.density",
    "sim.mujoco.opt.viscosity": "opt.viscosity",
    "sim.mujoco.opt.gravity[0]": "opt.gravity[0]",
    "sim.mujoco.opt.gravity[1]": "opt.gravity[1]",
    "sim.mujoco.opt.gravity[2]": "opt.gravity[2]",
    "sim.mujoco.opt.wind[0]": "opt.wind[0]",
    "sim.mujoco.opt.wind[1]": "opt.wind[1]",
    "sim.mujoco.opt.wind[2]": "opt.wind[2]",
    "sim.mujoco.opt.magnetic[0]": "opt.magnetic[0]",
    "sim.mujoco.opt.magnetic[1]": "opt.magnetic[1]",
    "sim.mujoco.opt.magnetic[2]": "opt.magnetic[2]",
    "sim.mujoco.opt.o_margin": "opt.o_margin",
    "sim.mujoco.opt.o_solref[0]": "opt.o_solref[0]",
    "sim.mujoco.opt.o_solref[1]": "opt.o_solref[1]",
    "sim.mujoco.opt.o_solimp[0]": "opt.o_solimp[0]",
    "sim.mujoco.opt.o_solimp[1]": "opt.o_solimp[1]",
    "sim.mujoco.opt.o_solimp[2]": "opt.o_solimp[2]",
    "sim.mujoco.opt.o_solimp[3]": "opt.o_solimp[3]",
    "sim.mujoco.opt.o_solimp[4]": "opt.o_solimp[4]",
    "sim.mujoco.opt.o_friction[0]": "opt.o_friction[0]",
    "sim.mujoco.opt.o_friction[1]": "opt.o_friction[1]",
    "sim.mujoco.opt.o_friction[2]": "opt.o_friction[2]",
    "sim.mujoco.opt.o_friction[3]": "opt.o_friction[3]",
    "sim.mujoco.opt.o_friction[4]": "opt.o_friction[4]",
    "sim.mujoco.opt.disableflags": "opt.disableflags",
    "sim.mujoco.opt.enableflags": "opt.enableflags",
    "sim.mujoco.opt.disableactuator": "opt.disableactuator",
    "sim.mujoco.opt.enableactuator": "opt.enableactuator",
}


def main() -> None:
    print("SIMULATION PARAMS:")
    for path, target in sorted(sim_specs.items()):
        print(f"  {path:50s} -> {target}")

    print("\nMUJOCO ENGINE PARAMS:")
    for path, target in sorted(mj_specs.items()):
        print(f"  {path:50s} -> {target}")


if __name__ == "__main__":
    main()

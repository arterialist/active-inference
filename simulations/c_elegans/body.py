"""
C. elegans MuJoCo body wrapper.

Wraps the body_model.xml MuJoCo model and exposes a BaseBody interface.

Scale note: the MJCF model uses 1000x biological scale (1mm → 1m).
The BodyState returned converts positions back to biological metres by
multiplying by 1e-3.
"""

from __future__ import annotations

from contextlib import contextmanager
from math import cos, pi, sin
from pathlib import Path
from typing import Generator

import mujoco
import numpy as np
from loguru import logger

from simulations.base_body import BaseBody, BodyState
from simulations.c_elegans import config as aec
from simulations.c_elegans.config import (
    N_BODY_SEGMENTS,
    MUSCLE_QUADRANTS,
)

_MODEL_XML = Path(__file__).parent / "body_model.xml"

# 1000x scale factor: model units → biological metres
_SCALE_MODEL_TO_BIO = 1e-3
# Bio → model units (inverse). Used when generating wall geometry that
# must match aec.ENV_PLATE_RADIUS_M (biological metres) inside MuJoCo.
_SCALE_BIO_TO_MODEL = 1.0 / _SCALE_MODEL_TO_BIO

# Floor plane sits at this z in model units (see body_model.xml worldbody).
_FLOOR_Z_MODEL = -0.04


def _build_wall_ring_xml(
    plate_radius_bio_m: float,
    wall_height_bio_m: float,
    wall_thickness_bio_m: float,
    n_segments: int,
    friction_tangent: float,
) -> str:
    """
    Generate a ring of static box geoms approximating a hollow cylinder
    around the agar disk.

    Each box has its inner face flush with the configured plate radius,
    its tangential extent slightly overlapping its neighbour to avoid
    gaps, and its base resting on the floor plane. All values are
    converted from biological metres to MuJoCo model units (1000×).
    """
    n = max(8, int(n_segments))
    radius_m = float(plate_radius_bio_m) * _SCALE_BIO_TO_MODEL
    height_m = float(wall_height_bio_m) * _SCALE_BIO_TO_MODEL
    thick_m = float(wall_thickness_bio_m) * _SCALE_BIO_TO_MODEL

    half_thickness = thick_m * 0.5
    half_height = height_m * 0.5
    centre_radius = radius_m + half_thickness
    # 10 % chord overlap between neighbours — kills any gap a worm
    # capsule could squeeze through under an oblique impact.
    half_chord = (centre_radius * sin(pi / n)) * 1.10
    z_centre = _FLOOR_Z_MODEL + half_height

    # Stiffer contact than the default floor: tighter solref time
    # constant + steeper solimp so the worm cannot tunnel under
    # strong muscle drive at the boundary.
    solref = "0.002 1"
    solimp = "0.99 0.999 0.0001"
    rgba = "0.55 0.45 0.35 0.55"

    geoms: list[str] = []
    for i in range(n):
        theta = 2.0 * pi * i / n
        cx = centre_radius * cos(theta)
        cy = centre_radius * sin(theta)
        size = f"{half_thickness:.6f} {half_chord:.6f} {half_height:.6f}"
        pos = f"{cx:.6f} {cy:.6f} {z_centre:.6f}"
        euler = f"0 0 {theta:.6f}"
        friction = f"{friction_tangent:.4f} 0.8 0.001"
        geoms.append(
            f'    <geom name="plate_wall_{i}" type="box" size="{size}"'
            f' pos="{pos}" euler="{euler}" friction="{friction}"'
            f' solref="{solref}" solimp="{solimp}" rgba="{rgba}"'
            f' condim="3"/>'
        )

    return "<!-- plate boundary wall (auto-generated) -->\n" + "\n".join(geoms)


def _load_model_with_wall() -> mujoco.MjModel:
    """
    Read body_model.xml, inject the boundary-wall ring just before the
    closing </worldbody> tag, and load via from_xml_string.

    The wall geometry is regenerated on every call from the current
    aec.* config values, so the lab "rebuild" parameters (plate radius,
    wall height, etc.) take effect on body re-creation without editing
    the XML on disk.
    """
    xml_text = _MODEL_XML.read_text(encoding="utf-8")
    wall_xml = _build_wall_ring_xml(
        plate_radius_bio_m=float(aec.ENV_PLATE_RADIUS_M),
        wall_height_bio_m=float(aec.WALL_HEIGHT_M),
        wall_thickness_bio_m=float(aec.WALL_THICKNESS_M),
        n_segments=int(aec.WALL_SEGMENTS_N),
        friction_tangent=float(aec.WALL_FRICTION_TANGENT),
    )
    marker = "</worldbody>"
    if marker not in xml_text:
        raise RuntimeError(
            f"body_model.xml is missing {marker}; cannot inject wall ring."
        )
    patched = xml_text.replace(marker, wall_xml + "\n  " + marker, 1)
    # Set assetdir so any relative includes still resolve.
    return mujoco.MjModel.from_xml_string(patched, assets={})


class CElegansBody(BaseBody):
    """
    MuJoCo-backed C. elegans worm body.

    Args:
        timestep:       Physics timestep (seconds).  Overrides XML value if set.
        render_width:   Width of rendered frames in pixels.
        render_height:  Height of rendered frames in pixels.
        settle_steps:   MuJoCo steps for gravity settle on reset (default 2000).
    """

    _SETTLE_STEPS_DEFAULT = 2000

    def __init__(
        self,
        timestep: float | None = None,
        render_width: int = 640,
        render_height: int = 480,
        settle_steps: int | None = None,
    ):
        logger.info(f"Loading MuJoCo model from {_MODEL_XML} (with plate wall)")
        self._model = _load_model_with_wall()
        self._data = mujoco.MjData(self._model)

        if timestep is not None:
            self._model.opt.timestep = timestep

        # Apply runtime joint-angle limit from config.JOINT_ANGLE_MAX_RAD.
        self._apply_joint_limits()

        self._render_width = render_width
        self._render_height = render_height
        self._renderer: mujoco.Renderer | None = None

        # Cache actuator name -> index map
        self._actuator_index: dict[str, int] = {}
        for i in range(self._model.nu):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self._actuator_index[name] = i

        # Cache sensor name -> index map
        self._sensor_index: dict[str, int] = {}
        for i in range(self._model.nsensor):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name:
                self._sensor_index[name] = i

        # Cache body name -> id
        self._body_id: dict[str, int] = {}
        for i in range(self._model.nbody):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                self._body_id[name] = i

        # Cache joint names
        self._joint_names_list: list[str] = []
        for i in range(self._model.njnt):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and "root" not in name:
                self._joint_names_list.append(name)

        self._settle_steps = (
            int(settle_steps)
            if settle_steps is not None
            else self._SETTLE_STEPS_DEFAULT
        )

        logger.info(
            f"CElegansBody ready: "
            f"{self._model.nu} actuators, "
            f"{len(self._joint_names_list)} joints, "
            f"dt={self._model.opt.timestep:.4f}s"
            f", settle_steps={self._settle_steps}"
        )

        self._root_joint_id: int = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, "root"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mass_weighted_com_model(self) -> np.ndarray:
        """Mass-weighted COM in MuJoCo model units (same scale as xpos)."""
        seg0_id = self._body_id.get("seg0", 1)
        seg_ids = [
            self._body_id[name]
            for name in self._body_id
            if name.startswith("seg") and name[3:].isdigit()
        ]
        if seg_ids:
            seg_ids = sorted(seg_ids)
            xpos = self._data.xpos[seg_ids]
            mass = self._model.body_mass[seg_ids]
            return (xpos * mass[:, None]).sum(axis=0) / max(float(mass.sum()), 1e-12)
        return self._data.xpos[seg0_id].copy()

    def _apply_joint_limits(self) -> None:
        """Override every hinge joint range with ±JOINT_ANGLE_MAX_RAD on yaws,
        and a tight ±PITCH_LIMIT_RAD on pitches.

        body_model.xml hardcodes joint ranges (historically ±1.2 rad), but
        ``config.JOINT_ANGLE_MAX_RAD`` is the runtime source of truth so the
        tuning UI can actually restrict body curvature without rewriting XML.
        Also stiffens the joint-limit constraint (solref/solimp) so the body
        cannot overshoot by 2-3× the nominal limit under strong muscle drive.

        Pitch (vertical-plane bend) gets a much smaller range so the body
        stays effectively planar — real C. elegans on agar crawls in 2D
        because surface tension binds it to the gel surface, which we don't
        simulate; the tight pitch limit substitutes geometrically.
        """
        try:
            max_rad = float(aec.JOINT_ANGLE_MAX_RAD)
            pitch_lim = float(getattr(aec, "PITCH_ANGLE_MAX_RAD", 0.02))
        except Exception:  # noqa: BLE001
            return
        # Stiff, well-damped contact constraint so limits behave as hard stops.
        stiff_solref = np.array([0.005, 1.0])  # 5 ms time constant
        stiff_solimp = np.array([0.995, 0.9999, 0.001, 0.5, 2.0])
        n_yaw = 0; n_pitch = 0
        for i in range(self._model.njnt):
            jtype = int(self._model.jnt_type[i])
            if jtype == mujoco.mjtJoint.mjJNT_HINGE:
                jname = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
                if "pitch" in jname:
                    self._model.jnt_range[i, 0] = -pitch_lim
                    self._model.jnt_range[i, 1] = +pitch_lim
                    n_pitch += 1
                else:
                    self._model.jnt_range[i, 0] = -max_rad
                    self._model.jnt_range[i, 1] = +max_rad
                    n_yaw += 1
                self._model.jnt_solref[i] = stiff_solref
                self._model.jnt_solimp[i] = stiff_solimp
        logger.debug(
            f"CElegansBody: yaw ±{max_rad:.3f} rad on {n_yaw}, "
            f"pitch ±{pitch_lim:.3f} rad on {n_pitch} (hard stops)"
        )

    # ------------------------------------------------------------------
    # BaseBody interface
    # ------------------------------------------------------------------

    def reset(self) -> BodyState:
        """Reset to default pose, then let gravity settle onto the floor.

        The body starts slightly above the floor to avoid
        interpenetration-kick artifacts, then runs zero-ctrl physics
        steps until the body rests on the substrate with zero velocity.
        """
        from simulations import evol_trace

        # Re-apply runtime joint-angle limit in case config changed since load
        self._apply_joint_limits()

        with evol_trace.span("reset_body_prep"):
            mujoco.mj_resetData(self._model, self._data)
            # Start above the floor so initial contact is clean
            self._data.qpos[2] = 0.045
            self._data.qvel[:] = 0.0
            mujoco.mj_forward(self._model, self._data)
        with evol_trace.span("reset_body_settle"):
            for _ in range(self._settle_steps):
                self._data.ctrl[:] = 0.0
                mujoco.mj_step(self._model, self._data)
        with evol_trace.span("reset_body_finalize"):
            self._data.qvel[:] = 0.0
            mujoco.mj_forward(self._model, self._data)

        return self.get_state()

    def step(self, muscle_activations: dict[str, float]) -> BodyState:
        """
        Apply muscle activations and step the physics.

        Args:
            muscle_activations: {muscle_name: activation in [0, 1]}
        """
        # Write activations into ctrl array
        self._data.ctrl[:] = 0.0
        for name, activation in muscle_activations.items():
            idx = self._actuator_index.get(name)
            if idx is not None:
                # Scale [0,1] activation to forcerange max
                force_max = self._model.actuator_forcerange[idx, 1]
                self._data.ctrl[idx] = float(np.clip(activation, 0.0, 1.0)) * force_max

        mujoco.mj_step(self._model, self._data)

        return self.get_state()

    def enforce_plate_bounds_radius_m(self, plate_radius_bio_m: float) -> bool:
        """
        Numerical-stability safety net for the plate boundary.

        With a real wall ring around the agar disk (see
        :func:`_build_wall_ring_xml`) the worm is normally obstructed by
        contact forces, so this method should never fire under healthy
        physics. It exists only to recover from blow-ups: if the COM or
        worst-case segment leaves the disk by more than
        ``aec.BOUNDARY_TELEPORT_FACTOR`` × R (default 1.2 → 20 % beyond),
        we recentre the root free joint and zero its xy velocity.

        Args:
            plate_radius_bio_m: Disk radius in biological metres (same as
                ``AgarPlateEnvironment.plate_radius_m`` / ``ENV_PLATE_RADIUS_M``).

        Returns:
            True if a corrective translation was applied.
        """
        if self._root_joint_id < 0:
            return False

        mujoco.mj_forward(self._model, self._data)

        seg_bio = self.get_body_shape()
        xy_rad = np.linalg.norm(seg_bio[:, :2], axis=1)
        r_max_seg = float(np.max(xy_rad))
        com_model = self._mass_weighted_com_model()
        com_bio_xy = com_model[:2] * _SCALE_MODEL_TO_BIO
        r_com = float(np.linalg.norm(com_bio_xy))

        # Safety threshold: only step in when the worm is well past the
        # wall (numerical instability), not on every nudge.
        factor = float(getattr(aec, "BOUNDARY_TELEPORT_FACTOR", 1.2))
        threshold = plate_radius_bio_m * factor

        need_fix = r_max_seg > threshold or r_com > threshold
        if not need_fix:
            return False

        qadr = int(self._model.jnt_qposadr[self._root_joint_id])
        # Free joint: qpos = (tx, ty, tz, qw, qx, qy, qz)
        self._data.qpos[qadr] -= float(com_model[0])
        self._data.qpos[qadr + 1] -= float(com_model[1])

        dadr = int(self._model.jnt_dofadr[self._root_joint_id])
        # Free joint vel: (vx, vy, vz, wx, wy, wz)
        self._data.qvel[dadr] = 0.0
        self._data.qvel[dadr + 1] = 0.0

        mujoco.mj_forward(self._model, self._data)
        logger.warning(
            "Plate bounds safety net: recentered worm (r_com={:.4g} m, "
            "r_seg_max={:.4g} m, R={:.4g} m, factor={:.2f})",
            r_com,
            r_max_seg,
            plate_radius_bio_m,
            factor,
        )
        return True

    def get_state(self) -> BodyState:
        """Read current simulation state into a BodyState."""
        mujoco.mj_forward(self._model, self._data)

        # --- True centre of mass of the full worm (mass-weighted) ---
        # Previously this was seg0 (head) position, which swings side-to-side
        # with every head sweep and is not representative of the body's motion
        # through the environment. Here we average all 13 body segments
        # weighted by body_mass so CoM reflects actual translation.
        com_model = self._mass_weighted_com_model()
        position = com_model * _SCALE_MODEL_TO_BIO

        seg0_id = self._body_id.get("seg0", 1)
        # --- Orientation (quaternion of head segment) ---
        xquat = self._data.xquat[seg0_id].copy()

        # --- Joint angles and velocities ---
        joint_angles: dict[str, float] = {}
        joint_vels: dict[str, float] = {}
        for jname in self._joint_names_list:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                qadr = self._model.jnt_qposadr[jid]
                dadr = self._model.jnt_dofadr[jid]
                joint_angles[jname] = float(self._data.qpos[qadr])
                joint_vels[jname] = float(self._data.qvel[dadr])

        # --- Touch sensors ---
        # Every <touch> sensor in the MJCF is exported by name into
        # BodyState.contact_forces. SensorEncoder picks per-neuron
        # receptive fields from this dict — no neuron is hard-wired
        # to a specific site here.
        contact_forces: dict[str, np.ndarray] = {}
        touch_names = (
            "touch_nose_sensor",
            "touch_nose_dorsal",
            "touch_nose_ventral",
            "touch_nose_left",
            "touch_nose_right",
            "touch_ant_sensor",
            "touch_post_sensor",
            "touch_seg0_sensor",
            "touch_seg1_sensor",
            "touch_seg2_sensor",
            "touch_seg3_sensor",
            "touch_seg4_sensor",
            "touch_seg5_sensor",
            "touch_seg6_sensor",
            "touch_seg7_sensor",
            "touch_seg8_sensor",
            "touch_seg9_sensor",
            "touch_seg10_sensor",
            "touch_seg11_sensor",
            "touch_seg12_sensor",
        )
        for sname in touch_names:
            sidx = self._sensor_index.get(sname)
            if sidx is not None:
                sadr = self._model.sensor_adr[sidx]
                val = float(self._data.sensordata[sadr])
                contact_forces[sname] = np.array([val, 0.0, 0.0])

        # --- Head position from sensor (biological scale) ---
        head_sidx = self._sensor_index.get("head_pos")
        if head_sidx is not None:
            sadr = self._model.sensor_adr[head_sidx]
            head_pos = (
                self._data.sensordata[sadr : sadr + 3].copy() * _SCALE_MODEL_TO_BIO
            )
        else:
            head_pos = position.copy()

        return BodyState(
            position=position,
            orientation=xquat,
            joint_angles=joint_angles,
            joint_velocities=joint_vels,
            contact_forces=contact_forces,
            head_position=head_pos,
        )

    def render(self, camera: str = "top") -> np.ndarray | None:
        """Render the scene. Returns (H, W, 3) uint8 RGB or None if headless."""
        try:
            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    self._model, self._render_height, self._render_width
                )
            self._renderer.update_scene(self._data)
            return self._renderer.render()
        except (RuntimeError, OSError) as e:
            logger.debug("Render failed (headless mode?): %s", e)
            return None

    @property
    def model(self) -> mujoco.MjModel:
        """MuJoCo model (for viewer, etc.)."""
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        """MuJoCo data (for viewer, etc.)."""
        return self._data

    @property
    def dt(self) -> float:
        return float(self._model.opt.timestep)

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names_list)

    @property
    def muscle_names(self) -> list[str]:
        return list(self._actuator_index.keys())

    # ------------------------------------------------------------------
    # Extra helpers
    # ------------------------------------------------------------------

    def get_body_shape(self) -> np.ndarray:
        """
        Return the positions of all 13 segment CoMs as (13, 3) array
        in biological metres, suitable for visualising the worm shape.
        """
        positions = np.zeros((N_BODY_SEGMENTS, 3))
        for i in range(N_BODY_SEGMENTS):
            bid = self._body_id.get(f"seg{i}")
            if bid is not None:
                positions[i] = self._data.xpos[bid] * _SCALE_MODEL_TO_BIO
        return positions

    def export_mujoco_state(self) -> tuple[list[float], list[float]]:
        """Serialise MuJoCo qpos/qvel for checkpointing (model units)."""
        mujoco.mj_forward(self._model, self._data)
        return self._data.qpos.copy().tolist(), self._data.qvel.copy().tolist()

    def import_mujoco_state(self, qpos: list[float], qvel: list[float]) -> None:
        """Restore MuJoCo qpos/qvel from lists (model units)."""
        self._data.qpos[:] = np.asarray(qpos, dtype=self._data.qpos.dtype)
        self._data.qvel[:] = np.asarray(qvel, dtype=self._data.qvel.dtype)
        mujoco.mj_forward(self._model, self._data)

    def close(self) -> None:
        """Release renderer resources."""
        if self._renderer is not None:
            del self._renderer
            self._renderer = None

    @contextmanager
    def render_context(self) -> Generator[CElegansBody, None, None]:
        """Context manager that ensures renderer is closed on exit."""
        try:
            yield self
        finally:
            self.close()

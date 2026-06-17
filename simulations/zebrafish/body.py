"""
MuJoCo-backed larval zebrafish body.

The model is a 5-7 dpf anatomical scaffold: a free-swimming head/trunk with a
16-link axial tail, yaw and pitch hinges per segment, paired left/right and
dorsal/ventral motor pools, eyes, yolk, swim bladder, and water-column
boundaries. Positions are kept in biological SI metres, unlike the C. elegans
body which uses a 1000x MuJoCo scale internally.
"""

from __future__ import annotations

from contextlib import contextmanager
from math import cos, pi, sin
from typing import Generator

import mujoco
import numpy as np
from loguru import logger

from simulations.base_body import BaseBody, BodyState
from simulations.zebrafish import config as zfc


def _angle_wrap(x: float) -> float:
    return float((x + np.pi) % (2.0 * np.pi) - np.pi)


def _fmt3(v: tuple[float, float, float] | np.ndarray) -> str:
    return " ".join(f"{float(x):.9g}" for x in v)


def _quat_from_yaw(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)


def _wall_ring_xml() -> str:
    """Return static box geoms forming the circular arena wall."""

    n = 80
    radius = zfc.ARENA_RADIUS_M
    height = zfc.ARENA_DEPTH_M
    thickness = 1.2e-3
    centre_radius = radius + thickness * 0.5
    half_chord = centre_radius * sin(pi / n) * 1.12
    z_centre = -height * 0.5
    rows: list[str] = []
    for i in range(n):
        theta = 2.0 * pi * i / n
        cx = centre_radius * cos(theta)
        cy = centre_radius * sin(theta)
        rows.append(
            f'<geom name="arena_wall_{i:02d}" type="box" '
            f'pos="{cx:.9g} {cy:.9g} {z_centre:.9g}" '
            f'euler="0 0 {theta:.9g}" '
            f'size="{thickness * 0.5:.9g} {half_chord:.9g} {height * 0.5:.9g}" '
            f'material="wall" friction="0.02 0.001 0.001" '
            f'solref="0.004 1" solimp="0.95 0.99 0.001" condim="3"/>'
        )
    return "\n      ".join(rows)


def _segment_body_xml(index: int) -> str:
    """Recursive tail segment body with two hinges and one ellipsoid geom."""

    seg_len = zfc.SEGMENT_LENGTH_M
    frac = index / max(1, zfc.N_BODY_SEGMENTS - 1)
    radius_y = zfc.BODY_RADIUS_M * (1.05 - 0.62 * frac)
    radius_z = zfc.BODY_RADIUS_M * (0.82 - 0.46 * frac)
    radius_y = max(radius_y, zfc.BODY_RADIUS_M * 0.22)
    radius_z = max(radius_z, zfc.BODY_RADIUS_M * 0.18)
    geom_pos = (-seg_len * 0.5, 0.0, 0.0)
    site_pos = (-seg_len, 0.0, 0.0)
    child = _segment_body_xml(index + 1) if index + 1 < zfc.N_BODY_SEGMENTS else ""
    return f"""
      <body name="tail_segment_{index:02d}" pos="{-seg_len:.9g} 0 0">
        <joint name="tail_yaw_{index:02d}" type="hinge" axis="0 0 1"
               range="{-zfc.TAIL_BEAT_MAX_AMPLITUDE_RAD:.6g} {zfc.TAIL_BEAT_MAX_AMPLITUDE_RAD:.6g}"
               damping="{zfc.TAIL_YAW_JOINT_DAMPING:.9g}"
               stiffness="{zfc.TAIL_YAW_JOINT_STIFFNESS:.9g}"
               springref="0" armature="2.0e-11"/>
        <joint name="tail_pitch_{index:02d}" type="hinge" axis="0 1 0"
               range="{-zfc.PITCH_LIMIT_RAD:.6g} {zfc.PITCH_LIMIT_RAD:.6g}"
               damping="{zfc.TAIL_PITCH_JOINT_DAMPING:.9g}"
               stiffness="{zfc.TAIL_PITCH_JOINT_STIFFNESS:.9g}"
               springref="0" armature="2.0e-11"/>
        <geom name="tail_geom_{index:02d}" type="ellipsoid"
              pos="{_fmt3(geom_pos)}"
              size="{seg_len * 0.50:.9g} {radius_y:.9g} {radius_z:.9g}"
              material="body" fluidshape="ellipsoid"/>
        <site name="tail_site_{index:02d}" pos="{_fmt3(site_pos)}" size="{max(radius_y, radius_z):.9g}"/>
        <site name="touch_tail_{index:02d}" pos="{_fmt3(geom_pos)}" size="{max(radius_y, radius_z) * 1.15:.9g}"/>
        {child}
      </body>"""


def _actuator_xml() -> str:
    gain_yaw = zfc.TAIL_YAW_ACTUATOR_GAIN
    gain_pitch = zfc.TAIL_PITCH_ACTUATOR_GAIN
    rows: list[str] = []
    for i in range(zfc.N_BODY_SEGMENTS):
        rows.extend(
            [
                f'<motor name="tail_{i:02d}_left" joint="tail_yaw_{i:02d}" '
                f'gear="{-gain_yaw:.9g}" ctrlrange="0 1" forcerange="0 1"/>',
                f'<motor name="tail_{i:02d}_right" joint="tail_yaw_{i:02d}" '
                f'gear="{gain_yaw:.9g}" ctrlrange="0 1" forcerange="0 1"/>',
                f'<motor name="tail_{i:02d}_dorsal" joint="tail_pitch_{i:02d}" '
                f'gear="{gain_pitch:.9g}" ctrlrange="0 1" forcerange="0 1"/>',
                f'<motor name="tail_{i:02d}_ventral" joint="tail_pitch_{i:02d}" '
                f'gear="{-gain_pitch:.9g}" ctrlrange="0 1" forcerange="0 1"/>',
            ]
        )
    return "\n    ".join(rows)


def _sensor_xml() -> str:
    rows = [
        '<framepos name="head_pos" objtype="site" objname="head_site"/>',
        '<framepos name="tail_tip_pos" objtype="site" objname="tail_site_15"/>',
        '<framelinvel name="head_linvel" objtype="site" objname="head_site"/>',
        '<frameangvel name="head_angvel" objtype="site" objname="head_site"/>',
        '<touch name="touch_head" site="touch_head"/>',
    ]
    for i in range(zfc.N_BODY_SEGMENTS):
        rows.append(f'<touch name="touch_tail_{i:02d}_sensor" site="touch_tail_{i:02d}"/>')
    return "\n    ".join(rows)


def _body_model_xml() -> str:
    """Generate MJCF for the larval fish body and water column."""

    head_len = zfc.HEAD_LENGTH_FRACTION * zfc.BODY_LENGTH_M
    trunk_len = zfc.SEGMENT_LENGTH_M * 1.2
    body_xml = f"""<mujoco model="larval_zebrafish">
  <compiler angle="radian" autolimits="true" inertiafromgeom="true"
            boundmass="1e-10" boundinertia="1e-13"/>
  <size njmax="256" nconmax="256"/>
  <option timestep="{zfc.PHYSICS_TIMESTEP_S:.9g}" gravity="0 0 0"
          density="1000" viscosity="0.0009" integrator="implicitfast"
          cone="elliptic" iterations="20"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <material name="water" rgba="0.10 0.42 0.50 0.25"/>
    <material name="body" rgba="0.84 0.91 0.90 1"/>
    <material name="stripe" rgba="0.18 0.28 0.34 1"/>
    <material name="eye" rgba="0.02 0.025 0.03 1"/>
    <material name="yolk" rgba="0.92 0.72 0.36 0.72"/>
    <material name="organ" rgba="0.72 0.86 0.90 0.45"/>
    <material name="wall" rgba="0.76 0.90 0.88 0.24"/>
  </asset>
  <default>
    <geom contype="1" conaffinity="1" condim="3" density="1050"
          friction="0.01 0.001 0.001" solref="0.006 1" solimp="0.93 0.99 0.001"
          fluidshape="ellipsoid" fluidcoef="0.9 0.25 1.4 1.0 1.0"/>
    <joint limited="true"/>
    <motor ctrllimited="true"/>
    <site rgba="0.2 0.7 0.9 0.25"/>
  </default>
  <worldbody>
    <light name="top_light" pos="0 -0.03 0.05" diffuse="0.8 0.9 1.0"/>
    <camera name="top" pos="0 -0.075 0.055" xyaxes="1 0 0 0 0.62 0.78"/>
    <camera name="side" pos="0 -0.09 -0.002" xyaxes="1 0 0 0 0 1"/>
    <geom name="water_volume" type="cylinder" pos="0 0 {-zfc.ARENA_DEPTH_M * 0.5:.9g}"
          size="{zfc.ARENA_RADIUS_M:.9g} {zfc.ARENA_DEPTH_M * 0.5:.9g}"
          material="water" contype="0" conaffinity="0" group="2"/>
    <geom name="water_floor" type="plane" pos="0 0 {zfc.MIN_SWIM_DEPTH_M:.9g}"
          size="{zfc.ARENA_RADIUS_M:.9g} {zfc.ARENA_RADIUS_M:.9g} 0.001"
          material="wall" solref="0.006 1" solimp="0.95 0.99 0.001"/>
    <geom name="water_surface" type="plane" pos="0 0 {zfc.MAX_SWIM_DEPTH_M:.9g}"
          zaxis="0 0 -1" size="{zfc.ARENA_RADIUS_M:.9g} {zfc.ARENA_RADIUS_M:.9g} 0.001"
          material="wall" solref="0.006 1" solimp="0.95 0.99 0.001"/>
    {_wall_ring_xml()}
    <body name="head" pos="0 0 {zfc.SWIM_DEPTH_M:.9g}">
      <freejoint name="root"/>
      <geom name="head_geom" type="ellipsoid" pos="{head_len * 0.10:.9g} 0 0"
            size="{head_len * 0.50:.9g} {zfc.BODY_RADIUS_M * 1.85:.9g} {zfc.BODY_RADIUS_M * 1.35:.9g}"
            material="body" fluidshape="ellipsoid"/>
      <geom name="trunk_geom" type="ellipsoid" pos="{-trunk_len * 0.35:.9g} 0 0"
            size="{trunk_len * 0.55:.9g} {zfc.BODY_RADIUS_M * 1.45:.9g} {zfc.BODY_RADIUS_M * 1.05:.9g}"
            material="body" fluidshape="ellipsoid"/>
      <geom name="yolk_geom" type="ellipsoid" pos="{-head_len * 0.18:.9g} 0 {-zfc.BODY_RADIUS_M * 0.82:.9g}"
            size="{head_len * 0.32:.9g} {zfc.BODY_RADIUS_M * 1.05:.9g} {zfc.BODY_RADIUS_M * 0.72:.9g}"
            material="yolk" contype="0" conaffinity="0"/>
      <geom name="swim_bladder_geom" type="ellipsoid" pos="{-head_len * 0.42:.9g} 0 {zfc.BODY_RADIUS_M * 0.50:.9g}"
            size="{head_len * 0.20:.9g} {zfc.BODY_RADIUS_M * 0.55:.9g} {zfc.BODY_RADIUS_M * 0.35:.9g}"
            material="organ" contype="0" conaffinity="0"/>
      <geom name="eye_left_geom" type="sphere" pos="{head_len * 0.42:.9g} {zfc.BODY_RADIUS_M * 1.20:.9g} {zfc.BODY_RADIUS_M * 0.18:.9g}"
            size="{zfc.BODY_RADIUS_M * 0.42:.9g}" material="eye" contype="0" conaffinity="0"/>
      <geom name="eye_right_geom" type="sphere" pos="{head_len * 0.42:.9g} {-zfc.BODY_RADIUS_M * 1.20:.9g} {zfc.BODY_RADIUS_M * 0.18:.9g}"
            size="{zfc.BODY_RADIUS_M * 0.42:.9g}" material="eye" contype="0" conaffinity="0"/>
      <site name="head_site" pos="{head_len * 0.62:.9g} 0 0" size="{zfc.BODY_RADIUS_M:.9g}"/>
      <site name="touch_head" pos="{head_len * 0.42:.9g} 0 0" size="{zfc.BODY_RADIUS_M * 1.75:.9g}"/>
      {_segment_body_xml(0)}
    </body>
  </worldbody>
  <actuator>
    {_actuator_xml()}
  </actuator>
  <sensor>
    {_sensor_xml()}
  </sensor>
</mujoco>"""
    return body_xml


class ZebrafishBody(BaseBody):
    """MuJoCo-backed segmented larval zebrafish."""

    def __init__(
        self,
        timestep: float | None = None,
        arena_radius_m: float = zfc.ARENA_RADIUS_M,
        render_width: int = 960,
        render_height: int = 640,
    ):
        self._arena_radius_m = float(arena_radius_m)
        self._model = mujoco.MjModel.from_xml_string(_body_model_xml(), assets={})
        self._data = mujoco.MjData(self._model)
        if timestep is not None:
            self._model.opt.timestep = float(timestep)
        self._render_width = int(render_width)
        self._render_height = int(render_height)
        self._renderer: mujoco.Renderer | None = None

        self._actuator_index: dict[str, int] = {}
        self._joint_names_list: list[str] = []
        self._body_id: dict[str, int] = {}
        self._site_id: dict[str, int] = {}
        self._sensor_index: dict[str, int] = {}
        for i in range(self._model.nu):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self._actuator_index[name] = i
        for i in range(self._model.njnt):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and name != "root":
                self._joint_names_list.append(name)
        for i in range(self._model.nbody):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                self._body_id[name] = i
        for i in range(self._model.nsite):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_SITE, i)
            if name:
                self._site_id[name] = i
        for i in range(self._model.nsensor):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name:
                self._sensor_index[name] = i

        self._root_joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self._head_body_id = self._body_id.get("head", 1)
        self._last_motor_outputs = {name: 0.0 for name in self.muscle_names}
        self._last_drive = 0.0
        self._last_turn_bias = 0.0
        self._last_pitch_bias = 0.0
        self.actuator_gains = {name: 1.0 for name in self.muscle_names}
        # Compatibility names used by the lab schema. They are mapped onto
        # MuJoCo option/actuator values rather than an external body integrator.
        self.linear_drag_per_s = float(zfc.LINEAR_DRAG_PER_S)
        self.angular_drag_per_s = float(zfc.ANGULAR_DRAG_PER_S)
        self.vertical_drag_per_s = float(zfc.VERTICAL_DRAG_PER_S)
        self.turn_accel_rad_s2 = float(zfc.TURN_ACCEL_RAD_S2)
        self.max_vertical_accel_m_s2 = float(zfc.MAX_VERTICAL_ACCEL_M_S2)
        self.pitch_drag_per_s = float(zfc.PITCH_DRAG_PER_S)
        self.pitch_restoring_per_s2 = float(zfc.PITCH_RESTORING_PER_S2)
        self.pitch_nonlinear_restoring_per_s2 = float(zfc.PITCH_NONLINEAR_RESTORING_PER_S2)
        self.pitch_nonlinear_threshold_rad = float(zfc.PITCH_NONLINEAR_THRESHOLD_RAD)
        self.depth_home_gain = float(zfc.DEPTH_HOME_GAIN)
        logger.info(
            "ZebrafishBody MuJoCo ready: {} bodies, {} joints, {} actuators, dt={:.4g}s",
            self._model.nbody,
            len(self._joint_names_list),
            self._model.nu,
            self.dt,
        )

    def reset(self) -> BodyState:
        mujoco.mj_resetData(self._model, self._data)
        qadr = int(self._model.jnt_qposadr[self._root_joint_id])
        self._data.qpos[qadr : qadr + 3] = np.array([0.0, 0.0, zfc.SWIM_DEPTH_M])
        self._data.qpos[qadr + 3 : qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self._data.qvel[:] = 0.0
        self._data.ctrl[:] = 0.0
        self._data.xfrc_applied[:] = 0.0
        self._last_motor_outputs = {name: 0.0 for name in self.muscle_names}
        self._last_drive = 0.0
        self._last_turn_bias = 0.0
        self._last_pitch_bias = 0.0
        mujoco.mj_forward(self._model, self._data)
        return self.get_state()

    def step(self, muscle_activations: dict[str, float]) -> BodyState:
        self._data.ctrl[:] = 0.0
        self._data.xfrc_applied[:] = 0.0
        cleaned: dict[str, float] = {}
        for name, raw in muscle_activations.items():
            idx = self._actuator_index.get(name)
            if idx is None:
                continue
            value = float(np.clip(raw * self.actuator_gains.get(name, 1.0), 0.0, 1.0))
            self._data.ctrl[idx] = value
            cleaned[name] = value
        self._last_motor_outputs = {name: cleaned.get(name, 0.0) for name in self.muscle_names}
        self._apply_swimming_forces()
        mujoco.mj_step(self._model, self._data)
        self._apply_runtime_damping()
        self._enforce_water_column()
        return self.get_state()

    def get_state(self) -> BodyState:
        mujoco.mj_forward(self._model, self._data)
        shape = self.get_body_shape()
        if shape.size:
            position = shape.mean(axis=0)
            head_position = shape[0].copy()
        else:
            position = self._data.xpos[self._head_body_id].copy()
            head_position = position.copy()

        orientation = self._data.xquat[self._head_body_id].copy()
        joint_angles: dict[str, float] = {}
        joint_vels: dict[str, float] = {}
        for jname in self._joint_names_list:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                continue
            qadr = int(self._model.jnt_qposadr[jid])
            dadr = int(self._model.jnt_dofadr[jid])
            joint_angles[jname] = float(self._data.qpos[qadr])
            joint_vels[jname] = float(self._data.qvel[dadr])

        contact_forces: dict[str, np.ndarray] = {}
        for sname, sidx in self._sensor_index.items():
            if not sname.startswith("touch_"):
                continue
            adr = int(self._model.sensor_adr[sidx])
            value = float(self._data.sensordata[adr])
            contact_forces[sname] = np.array([value, 0.0, 0.0], dtype=float)

        root_vel = self._root_velocity()
        yaw_rate, pitch_rate = self._angular_rates()
        heading, pitch = self._heading_pitch(shape)
        tail_yaw = np.array(
            [joint_angles.get(f"tail_yaw_{i:02d}", 0.0) for i in range(zfc.N_BODY_SEGMENTS)],
            dtype=float,
        )
        tail_pitch = np.array(
            [joint_angles.get(f"tail_pitch_{i:02d}", 0.0) for i in range(zfc.N_BODY_SEGMENTS)],
            dtype=float,
        )
        return BodyState(
            position=position,
            orientation=orientation,
            joint_angles=joint_angles,
            joint_velocities=joint_vels,
            contact_forces=contact_forces,
            head_position=head_position,
            extra={
                "heading": heading,
                "pitch_rad": pitch,
                "velocity": root_vel.copy(),
                "speed_m_s": float(np.linalg.norm(root_vel[:2])),
                "vertical_velocity_m_s": float(root_vel[2]),
                "angular_velocity_rad_s": yaw_rate,
                "pitch_velocity_rad_s": pitch_rate,
                "tail_points": shape.copy(),
                "tail_angles": tail_yaw,
                "tail_pitch_angles": tail_pitch,
                "swim_drive": float(self._last_drive),
                "turn_bias": float(self._last_turn_bias),
                "pitch_bias": float(self._last_pitch_bias),
            },
        )

    def render(self, camera: str = "top") -> np.ndarray | None:
        try:
            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    self._model,
                    self._render_height,
                    self._render_width,
                )
            self._renderer.update_scene(self._data, camera=camera)
            return self._renderer.render()
        except (RuntimeError, OSError, ValueError) as exc:
            logger.debug("Zebrafish render failed: {}", exc)
            return None

    def get_body_shape(self) -> np.ndarray:
        mujoco.mj_forward(self._model, self._data)
        points: list[np.ndarray] = []
        head_id = self._site_id.get("head_site")
        if head_id is not None:
            points.append(self._data.site_xpos[head_id].copy())
        for i in range(zfc.N_BODY_SEGMENTS):
            sid = self._site_id.get(f"tail_site_{i:02d}")
            if sid is not None:
                points.append(self._data.site_xpos[sid].copy())
        return np.asarray(points, dtype=float)

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
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

    def export_mujoco_state(self) -> tuple[list[float], list[float]]:
        mujoco.mj_forward(self._model, self._data)
        return self._data.qpos.copy().tolist(), self._data.qvel.copy().tolist()

    def import_mujoco_state(self, qpos: list[float], qvel: list[float]) -> None:
        self._data.qpos[:] = np.asarray(qpos, dtype=self._data.qpos.dtype)
        self._data.qvel[:] = np.asarray(qvel, dtype=self._data.qvel.dtype)
        mujoco.mj_forward(self._model, self._data)

    def close(self) -> None:
        if self._renderer is not None:
            del self._renderer
            self._renderer = None

    @contextmanager
    def render_context(self) -> Generator[ZebrafishBody, None, None]:
        try:
            yield self
        finally:
            self.close()

    def _root_velocity(self) -> np.ndarray:
        if self._root_joint_id < 0:
            return np.zeros(3)
        dadr = int(self._model.jnt_dofadr[self._root_joint_id])
        return self._data.qvel[dadr : dadr + 3].copy()

    def _angular_rates(self) -> tuple[float, float]:
        if self._root_joint_id < 0:
            return 0.0, 0.0
        dadr = int(self._model.jnt_dofadr[self._root_joint_id])
        ang = self._data.qvel[dadr + 3 : dadr + 6]
        return float(ang[2]), float(ang[1])

    def _heading_pitch(self, shape: np.ndarray) -> tuple[float, float]:
        heading, pitch, _, _ = self._body_frame(shape)
        return heading, pitch

    def _body_frame(self, shape: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
        if shape.shape[0] < 2:
            forward_h = np.array([1.0, 0.0, 0.0])
            lateral = np.array([0.0, 1.0, 0.0])
            return 0.0, 0.0, forward_h, lateral
        axis = shape[0] - shape[-1]
        horizontal = float(np.linalg.norm(axis[:2]))
        if horizontal > 1e-12:
            forward_h = np.array([axis[0] / horizontal, axis[1] / horizontal, 0.0])
            heading = float(np.arctan2(axis[1], axis[0]))
        else:
            forward_h = np.array([1.0, 0.0, 0.0])
            heading = 0.0
        lateral = np.array([-forward_h[1], forward_h[0], 0.0])
        pitch = float(np.arctan2(axis[2], max(horizontal, 1e-12)))
        return _angle_wrap(heading), pitch, forward_h, lateral

    def _apply_swimming_forces(self) -> None:
        left = np.array([self._last_motor_outputs.get(f"tail_{i:02d}_left", 0.0) for i in range(zfc.N_BODY_SEGMENTS)])
        right = np.array([self._last_motor_outputs.get(f"tail_{i:02d}_right", 0.0) for i in range(zfc.N_BODY_SEGMENTS)])
        dorsal = np.array([self._last_motor_outputs.get(f"tail_{i:02d}_dorsal", 0.0) for i in range(zfc.N_BODY_SEGMENTS)])
        ventral = np.array([self._last_motor_outputs.get(f"tail_{i:02d}_ventral", 0.0) for i in range(zfc.N_BODY_SEGMENTS)])
        axial = left + right
        signed = right - left
        vertical_signed = dorsal - ventral
        drive = float(np.clip(np.mean(axial), 0.0, 1.0))
        turn_bias = float(np.clip(np.mean(signed[: max(3, zfc.N_BODY_SEGMENTS // 3)]), -1.0, 1.0))
        pitch_bias = float(np.clip(np.mean(vertical_signed[: max(3, zfc.N_BODY_SEGMENTS // 3)]), -1.0, 1.0))
        self._last_drive = drive
        self._last_turn_bias = turn_bias
        self._last_pitch_bias = pitch_bias
        if drive <= 1e-5 and abs(pitch_bias) <= 1e-5:
            return

        shape = self.get_body_shape()
        if shape.shape[0] < 2:
            return
        _, _, forward_h, lateral = self._body_frame(shape)
        mass = max(float(np.sum(self._model.body_mass)), 1e-10)
        tail_work = float(np.mean(np.abs(np.diff(signed, prepend=signed[0] if signed.size else 0.0))))
        thrust_accel = zfc.MAX_FORWARD_ACCEL_M_S2 * drive * float(np.clip(0.35 + tail_work, 0.0, 1.35))
        vertical_accel = self.max_vertical_accel_m_s2 * pitch_bias
        force = mass * (thrust_accel * forward_h + vertical_accel * np.array([0.0, 0.0, 1.0]))
        self._data.xfrc_applied[self._head_body_id, :3] += force
        torque = mass * zfc.BODY_LENGTH_M * (
            0.18 * self.turn_accel_rad_s2 * turn_bias * np.array([0.0, 0.0, 1.0])
            - 0.10 * zfc.PITCH_ACCEL_RAD_S2 * pitch_bias * lateral
        )
        self._data.xfrc_applied[self._head_body_id, 3:] += torque

    def _enforce_water_column(self) -> None:
        if self._root_joint_id < 0:
            return
        mujoco.mj_forward(self._model, self._data)
        qadr = int(self._model.jnt_qposadr[self._root_joint_id])
        dadr = int(self._model.jnt_dofadr[self._root_joint_id])
        pos = self._data.qpos[qadr : qadr + 3]
        vel = self._data.qvel[dadr : dadr + 3]
        changed = False
        if pos[2] > zfc.MAX_SWIM_DEPTH_M:
            pos[2] = zfc.MAX_SWIM_DEPTH_M
            if vel[2] > 0.0:
                vel[2] *= -zfc.BOUNDARY_RESTITUTION
            changed = True
        elif pos[2] < zfc.MIN_SWIM_DEPTH_M:
            pos[2] = zfc.MIN_SWIM_DEPTH_M
            if vel[2] < 0.0:
                vel[2] *= -zfc.BOUNDARY_RESTITUTION
            changed = True
        radial = float(np.linalg.norm(pos[:2]))
        safety = self._arena_radius_m * 1.2
        if radial > safety:
            pos[:2] *= (self._arena_radius_m * 0.95) / max(radial, 1e-12)
            vel[:2] = 0.0
            changed = True
            logger.warning("Zebrafish arena safety reset fired at radius {:.4g} m", radial)
        if changed:
            mujoco.mj_forward(self._model, self._data)

    def _apply_runtime_damping(self) -> None:
        if self._root_joint_id < 0:
            return
        dadr = int(self._model.jnt_dofadr[self._root_joint_id])
        dt = self.dt
        self._data.qvel[dadr : dadr + 2] *= float(np.exp(-self.linear_drag_per_s * dt))
        self._data.qvel[dadr + 2] *= float(np.exp(-self.vertical_drag_per_s * dt))
        angular = self._data.qvel[dadr + 3 : dadr + 6]
        angular *= float(np.exp(-self.angular_drag_per_s * dt))
        qadr = int(self._model.jnt_qposadr[self._root_joint_id])
        depth_error = zfc.DEPTH_HOME_M - float(self._data.qpos[qadr + 2])
        self._data.qvel[dadr + 2] += float(self.depth_home_gain * depth_error * dt)
        shape = self.get_body_shape()
        _, pitch, _, lateral = self._body_frame(shape)
        pitch_rate = float(np.dot(angular, lateral))
        pitch_decay = float(np.exp(-self.pitch_drag_per_s * dt))
        angular += lateral * (pitch_rate * (pitch_decay - 1.0))
        nonlinear_pitch = max(0.0, abs(pitch) - max(0.0, self.pitch_nonlinear_threshold_rad))
        nonlinear_restore = (
            self.pitch_nonlinear_restoring_per_s2
            * nonlinear_pitch
            * np.sign(pitch)
        )
        angular += lateral * float((self.pitch_restoring_per_s2 * pitch + nonlinear_restore) * dt)

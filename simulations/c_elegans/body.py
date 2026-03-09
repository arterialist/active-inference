"""
C. elegans MuJoCo body wrapper.

Wraps the body_model.xml MuJoCo model and exposes a BaseBody interface.

Scale note: the MJCF model uses 1000x biological scale (1mm → 1m).
The BodyState returned converts positions back to biological metres by
multiplying by 1e-3.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import mujoco
import numpy as np
from loguru import logger

from simulations.base_body import BaseBody, BodyState
from simulations.c_elegans.config import N_BODY_SEGMENTS, MUSCLE_QUADRANTS

_MODEL_XML = Path(__file__).parent / "body_model.xml"

# 1000x scale factor: model units → biological metres
_SCALE_MODEL_TO_BIO = 1e-3


class CElegansBody(BaseBody):
    """
    MuJoCo-backed C. elegans worm body.

    Args:
        timestep:       Physics timestep (seconds).  Overrides XML value if set.
        render_width:   Width of rendered frames in pixels.
        render_height:  Height of rendered frames in pixels.
    """

    def __init__(
        self,
        timestep: float | None = None,
        render_width: int = 640,
        render_height: int = 480,
    ):
        logger.info(f"Loading MuJoCo model from {_MODEL_XML}")
        self._model = mujoco.MjModel.from_xml_path(str(_MODEL_XML))
        self._data = mujoco.MjData(self._model)

        if timestep is not None:
            self._model.opt.timestep = timestep

        self._render_width = render_width
        self._render_height = render_height
        self._renderer: mujoco.Renderer | None = None

        # Cache actuator name -> index map
        self._actuator_index: dict[str, int] = {}
        for i in range(self._model.nu):
            name = mujoco.mj_id2name(
                self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i
            )
            if name:
                self._actuator_index[name] = i

        # Cache sensor name -> index map
        self._sensor_index: dict[str, int] = {}
        for i in range(self._model.nsensor):
            name = mujoco.mj_id2name(
                self._model, mujoco.mjtObj.mjOBJ_SENSOR, i
            )
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
            name = mujoco.mj_id2name(
                self._model, mujoco.mjtObj.mjOBJ_JOINT, i
            )
            if name and "root" not in name:
                self._joint_names_list.append(name)

        logger.info(
            f"CElegansBody ready: "
            f"{self._model.nu} actuators, "
            f"{len(self._joint_names_list)} joints, "
            f"dt={self._model.opt.timestep:.4f}s"
        )

    # ------------------------------------------------------------------
    # BaseBody interface
    # ------------------------------------------------------------------

    def reset(self) -> BodyState:
        """Reset to default pose (straight worm along x-axis)."""
        mujoco.mj_resetData(self._model, self._data)
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

    def get_state(self) -> BodyState:
        """Read current simulation state into a BodyState."""
        mujoco.mj_forward(self._model, self._data)

        # --- Centre of mass (head segment, biological scale) ---
        seg0_id = self._body_id.get("seg0", 1)
        com_model = self._data.xpos[seg0_id].copy()
        position = com_model * _SCALE_MODEL_TO_BIO

        # --- Orientation (quaternion of head segment) ---
        xquat = self._data.xquat[seg0_id].copy()

        # --- Joint angles and velocities ---
        joint_angles: dict[str, float] = {}
        joint_vels: dict[str, float] = {}
        for jname in self._joint_names_list:
            jid = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_JOINT, jname
            )
            if jid >= 0:
                qadr = self._model.jnt_qposadr[jid]
                dadr = self._model.jnt_dofadr[jid]
                joint_angles[jname] = float(self._data.qpos[qadr])
                joint_vels[jname] = float(self._data.qvel[dadr])

        # --- Touch sensors ---
        contact_forces: dict[str, np.ndarray] = {}
        for sname in ("touch_nose_sensor", "touch_ant_sensor", "touch_post_sensor"):
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

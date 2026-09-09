"""Physical-only bridge for the retained reference MuJoCo rower geometry."""
import mujoco
import numpy as np

from ...nmrower2 import XML, GGAIN


class ResearchRower:
    state_signature = mujoco.mjtState.mjSTATE_INTEGRATION

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.joint_ids = [self.model.joint(n).id for n in ('pl','pr')]
        self.positions = [int(self.model.jnt_qposadr[n]) for n in self.joint_ids]
        self.actuators = [self.model.actuator(n).id for n in ('plp','plr','prp','prr')]

    def sense(self):
        # A bounded opponent joint-position transducer, not a prediction.
        # The scale comes from the declared joint range in the physical model.
        values = self.data.qpos[self.positions]
        scales = np.max(np.abs(self.model.jnt_range[self.joint_ids]),axis=1)
        normalized = np.clip(values/scales,-1.,1.)
        return np.maximum(0.,np.array([normalized[0],-normalized[0],normalized[1],-normalized[1]]))

    def step(self, muscle_state, *, gain=GGAIN):
        values = np.asarray(muscle_state,dtype=float)
        if values.shape != (4,) or not np.isfinite(values).all() or not np.isfinite(gain) or gain < 0:
            raise ValueError('Invalid muscle-to-actuator input')
        self.data.ctrl[self.actuators] = gain*np.maximum(0.,values)
        mujoco.mj_step(self.model,self.data)
        if not np.isfinite(self.state()).all():
            raise FloatingPointError('Nonfinite physical state')

    def state(self):
        state = np.empty(mujoco.mj_stateSize(self.model,self.state_signature))
        mujoco.mj_getState(self.model,self.data,state,self.state_signature)
        return state

    def restore(self, state):
        state = np.asarray(state,dtype=float)
        if state.shape != self.state().shape or not np.isfinite(state).all():
            raise ValueError('Invalid physical integration state')
        mujoco.mj_setState(self.model,self.data,state,self.state_signature)

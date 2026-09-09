"""A physical sweep task with passive, movement-dependent resistance.

The organism drives one real MuJoCo hinge. Spring and additional drag are
sampled at the start of a physics step and applied as explicit environmental
torque. Existing joint damping is integrated by MuJoCo. Force receptors measure
the applied environmental torque, not a target, score or commanded correction.
Alternating position crossings count completed sweeps; they do not drive neurons.
"""
import math

import mujoco
import numpy as np


DT = .004
MOTOR_GEAR = .2
XML = '''<mujoco model="loaded-sweep"><compiler angle="radian"/>
<option timestep="0.004" gravity="0 0 0" integrator="implicitfast"/>
<worldbody><body name="arm"><joint name="hinge" type="hinge" axis="0 0 1"
 damping="0.12" armature="0.04"/><geom type="capsule" fromto="0 0 0 .25 0 0"
 size=".02" mass=".3"/></body></worldbody>
<actuator><motor name="muscle" joint="hinge" gear="0.2"/></actuator></mujoco>'''


class LoadedHinge:
    def __init__(self, drag=0., spring=.15, gate=.008):
        if (not all(math.isfinite(x) for x in (drag, spring, gate))
                or not 0 <= drag <= 1. or spring <= 0 or gate <= 0):
            raise ValueError('Invalid passive sweep environment')
        self.drag, self.spring, self.gate = drag, spring, gate
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.size = mujoco.mj_stateSize(self.model, self.spec)
        self.next_gate = 1
        self.crossings = 0

    def state(self):
        value = np.empty(self.size)
        mujoco.mj_getState(self.model, self.data, value, self.spec)
        return value

    def restore(self, state, *, next_gate=1, crossings=0):
        if next_gate not in (-1, 1) or type(crossings) is not int or crossings < 0:
            raise ValueError('Invalid gate history')
        mujoco.mj_setState(self.model, self.data, state, self.spec)
        mujoco.mj_forward(self.model, self.data)
        self.next_gate, self.crossings = next_gate, crossings

    def environmental_torque(self):
        return -self.spring*float(self.data.qpos[0])-self.drag*float(self.data.qvel[0])

    def step(self, command):
        if not math.isfinite(command):
            raise ValueError('Nonfinite muscle command')
        force = self.environmental_torque()
        self.data.ctrl[0] = command
        self.data.qfrc_applied[0] = force
        mujoco.mj_step(self.model, self.data)
        crossed = self.next_gate*float(self.data.qpos[0]) >= self.gate
        if crossed:
            self.crossings += 1
            self.next_gate *= -1
        return force, int(crossed)


def afferents(body):
    """Six nonnegative transducer channels in fixed physical units; no clipping."""
    force = body.environmental_torque()/MOTOR_GEAR
    position = float(body.data.qpos[0])/.05
    velocity = float(body.data.qvel[0])/.2
    return np.array([max(0., x) for x in (force, -force, position, -position, velocity, -velocity)])

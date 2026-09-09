"""Explicit angle-unit intervention, leaving the reference XML unchanged.

Only the XML compiler's angle convention changes. The declared hinge range
becomes +/-1.8 radians instead of +/-1.8 degrees. This is a different body,
not a silent correction or behavioral-equivalence claim about old results.
"""
import mujoco

from .research_rower import ResearchRower
from ...nmrower2 import XML

RADIAN_XML = XML.replace('<mujoco>', '<mujoco><compiler angle="radian"/>', 1)


class RadianResearchRower(ResearchRower):
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(RADIAN_XML)
        self.data = mujoco.MjData(self.model)
        self.joint_ids = [self.model.joint(n).id for n in ('pl','pr')]
        self.positions = [int(self.model.jnt_qposadr[n]) for n in self.joint_ids]
        self.actuators = [self.model.actuator(n).id for n in ('plp','plr','prp','prr')]

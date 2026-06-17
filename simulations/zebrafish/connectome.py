"""
Reduced larval zebrafish circuit scaffold.

This is not a Fish1-scale connectome. It is a biologically annotated seed graph
for a first embodied simulation, with named circuit motifs from zebrafish
visuomotor, lateral-line, olfactory/approach, startle, and spinal swimming
literature. The source corpus in ``research/`` records the public datasets that
should replace or refine this graph over time.
"""

from __future__ import annotations

from simulations.connectome_loader import ConnectomeData, NeuronInfo, SynapticEdge
from simulations.zebrafish import config as zfc


def load_connectome() -> ConnectomeData:
    """Return a reduced PAULA-ready larval zebrafish circuit."""
    neurons: list[NeuronInfo] = []
    for idx, name in enumerate(zfc.ALL_NEURONS):
        if name in zfc.SENSORY_NEURONS:
            ntype = "sensory"
        elif name in zfc.MOTOR_NEURONS:
            ntype = "motor"
        else:
            ntype = "interneuron"
        neurons.append(NeuronInfo(name=name, neuron_type=ntype, paula_id=idx))

    chem: list[SynapticEdge] = []

    def add(pre: str, post: str, weight: float = 1.0) -> None:
        chem.append(
            SynapticEdge(
                pre_name=pre,
                post_name=post,
                synapse_type="chemical",
                weight=float(weight),
            )
        )

    # Visual target / prey approach: retina -> tectum -> hindbrain/approach.
    add("RETINA_L", "TECTUM_L", 3.0)
    add("RETINA_R", "TECTUM_R", 3.0)
    add("TECTUM_L", "ARTR_L", 1.7)
    add("TECTUM_R", "ARTR_R", 1.7)
    add("TECTUM_L", "APPROACH_GATE", 1.1)
    add("TECTUM_R", "APPROACH_GATE", 1.1)

    # Optomotor/pretectal stabilization.
    add("OPTIC_FLOW_L", "PRETECTUM_L", 2.4)
    add("OPTIC_FLOW_R", "PRETECTUM_R", 2.4)
    add("PRETECTUM_L", "HINDBRAIN_R", 1.2)
    add("PRETECTUM_R", "HINDBRAIN_L", 1.2)
    add("PRETECTUM_L", "SWIM_GATE", 0.8)
    add("PRETECTUM_R", "SWIM_GATE", 0.8)

    # Lateral line and wall avoidance.
    add("LATERAL_LINE_L", "HINDBRAIN_R", 1.0)
    add("LATERAL_LINE_R", "HINDBRAIN_L", 1.0)
    add("WALL_L", "HINDBRAIN_R", 2.2)
    add("WALL_R", "HINDBRAIN_L", 2.2)
    add("WALL_L", "AVOID_GATE", 1.4)
    add("WALL_R", "AVOID_GATE", 1.4)

    # Olfactory/thermal approach and avoidance gates.
    add("OLFACTORY_L", "APPROACH_GATE", 1.6)
    add("OLFACTORY_R", "APPROACH_GATE", 1.6)
    add("THERMO_HOT", "AVOID_GATE", 1.2)
    add("THERMO_COLD", "AVOID_GATE", 1.2)
    add("APPROACH_GATE", "SWIM_GATE", 1.5)
    add("AVOID_GATE", "SWIM_GATE", 1.1)

    # Vestibular correction.
    add("VESTIBULAR_L", "HINDBRAIN_R", 0.8)
    add("VESTIBULAR_R", "HINDBRAIN_L", 0.8)
    add("VESTIBULAR_UP", "DEPTH_HOMEOSTAT", 0.8)
    add("VESTIBULAR_DOWN", "DEPTH_HOMEOSTAT", 0.8)
    add("DEPTH_SHALLOW", "DEPTH_HOMEOSTAT", 1.7)
    add("DEPTH_DEEP", "DEPTH_HOMEOSTAT", 1.7)
    add("DEPTH_HOMEOSTAT", "ASCEND_GATE", 1.4)
    add("DEPTH_HOMEOSTAT", "DIVE_GATE", 1.4)

    # Startle/Mauthner C-start path.
    add("STARTLE", "MAUTHNER_L", 2.8)
    add("STARTLE", "MAUTHNER_R", 2.8)
    add("MAUTHNER_L", "RETICULOSPINAL_L", 2.3)
    add("MAUTHNER_R", "RETICULOSPINAL_R", 2.3)

    # Hindbrain / ARTR turns and swim gate.
    add("ARTR_L", "HINDBRAIN_L", 1.3)
    add("ARTR_R", "HINDBRAIN_R", 1.3)
    add("HINDBRAIN_L", "RETICULOSPINAL_L", 2.0)
    add("HINDBRAIN_R", "RETICULOSPINAL_R", 2.0)
    add("SWIM_GATE", "BOUT_CLOCK", 1.2)
    add("BOUT_CLOCK", "SPINAL_CPG_L", 1.4)
    add("BOUT_CLOCK", "SPINAL_CPG_R", 1.4)
    add("RETICULOSPINAL_L", "SPINAL_CPG_L", 1.6)
    add("RETICULOSPINAL_R", "SPINAL_CPG_R", 1.6)
    add("ASCEND_GATE", "SPINAL_CPG_L", 0.5)
    add("ASCEND_GATE", "SPINAL_CPG_R", 0.5)
    add("DIVE_GATE", "SPINAL_CPG_L", 0.5)
    add("DIVE_GATE", "SPINAL_CPG_R", 0.5)

    # Segmental spinal motor pools.
    for i in range(zfc.N_BODY_SEGMENTS):
        taper = 1.0 - 0.35 * (i / max(1, zfc.N_BODY_SEGMENTS - 1))
        add("SPINAL_CPG_L", f"MOTOR_L_{i:02d}", 1.8 * taper)
        add("SPINAL_CPG_R", f"MOTOR_R_{i:02d}", 1.8 * taper)
        add("RETICULOSPINAL_L", f"MOTOR_L_{i:02d}", 0.9 * taper)
        add("RETICULOSPINAL_R", f"MOTOR_R_{i:02d}", 0.9 * taper)
        add("ASCEND_GATE", f"MOTOR_D_{i:02d}", 1.2 * taper)
        add("DIVE_GATE", f"MOTOR_V_{i:02d}", 1.2 * taper)

    return ConnectomeData(neurons=neurons, chemical_edges=chem, gap_junction_edges=[])


def print_connectome_summary(connectome: ConnectomeData) -> None:
    """Print a concise source-aware summary."""
    print(
        "Larval zebrafish reduced circuit: "
        f"{connectome.n_neurons} neurons, "
        f"{len(connectome.chemical_edges)} directed chemical edges, "
        f"{len(connectome.gap_junction_edges)} gap junction edges"
    )

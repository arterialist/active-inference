"""AIF AGENT 3D — the whole mini-brain in the MuJoCo world: see, believe, doubt, decide, act.

Everything below is one PAULA network. The only non-neural steps remain the sanctioned transducers
(luminance/odour -> current, graded-muscle membrane -> force) plus one-time birth seeds.

  VISION        retina -> ON/OFF -> V1 -> V2 -> salience            (visual_cortex, generic)
  COMPASS       heading ring + P-EN velocity shift                  (central_complex)
  ANCHORING     salience -> VR -> GABAergic ring anchoring          (seeing_agent mechanism)
  PATH INTEG.   CPU4 ladders -> CPU1 comparator -> muscles          (cx_navigator)
  SELF-MODEL    PE[j] = AND(vision says heading j, bump is NOT at j).  A prediction-error population:
                the agent's own heading BELIEF checked against the world. Sum(PE) is literally surprise.
  CONFIDENCE    UNC, a bidirectional ladder that FILLS while moving (path integration accrues error) and
                is DRAINED by seeing a landmark (a fix restores certainty). This is the EPISTEMIC state:
                "how much do I still trust where I think I am".
  ARBITER       spiking WTA over FORAGE / HOME / EXPLORE, driven by the brain's own beliefs:
                   FORAGE  <- hunger ladder            (pragmatic, interoceptive)
                   HOME    <- |home vector| (MUS_F)    (pragmatic)
                   EXPLORE <- UNC + surprise           (EPISTEMIC -- act to reduce uncertainty)
                which is expected-free-energy action selection in spikes.

Run standalone:  python aif_agent3d.py    (stagewise checks, then a closed-loop episode)
"""
import sys, importlib.util, numpy as np
from pathlib import Path

# Keep the custom organism importable both as a package and through its legacy
# standalone entrypoints.  The PAULA substrate is a sibling repository, not a
# machine-specific absolute path.
_MODULE_DIR = Path(__file__).resolve().parent
_NEURON_MODEL_DIR = _MODULE_DIR.parents[2] / "neuron-model"
if str(_NEURON_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_NEURON_MODEL_DIR))
from paula_agent import ckit as k
def _L(n,p):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s)
    # Register before execution: dataclasses and other annotation-aware code
    # resolve their module through sys.modules while their class body runs.
    sys.modules[n]=m; s.loader.exec_module(m); return m
cc=_L("cc",_MODULE_DIR / "central_complex.py"); vc=_L("vc",_MODULE_DIR / "visual_cortex.py"); w3=_L("w3",_MODULE_DIR / "components" / "body" / "world.py")
nv=_L("nv",_MODULE_DIR / "cx_navigator.py"); ar=_L("ar",_MODULE_DIR / "components" / "arbitration" / "paula.py")
pacc=_L("pacc",_MODULE_DIR / "pi_accum.py")
pstone=_L("pstone",_MODULE_DIR / "pi_stone.py")   # redesigned PI accumulator (leaky, circular, global inhibition)
met=_L("metabolic_parts",_MODULE_DIR / "components" / "arbitration" / "metabolic_parts.py")
obs=_L("obstacle_sensory",_MODULE_DIR / "components" / "sensory" / "obstacle.py")
obm=_L("obstacle_reflex",_MODULE_DIR / "components" / "motor" / "obstacle_reflex.py")
composer=_L("brain_composer",_MODULE_DIR / "core" / "brain_composer.py")
ec=_L("embodied_config",_MODULE_DIR / "embodied_config.py")
EmbodiedAgentConfig=ec.EmbodiedAgentConfig
DEFAULT_EMBODIED_CONFIG=ec.DEFAULT_EMBODIED_CONFIG
LEGACY_DIRECT_TICK_CONFIG=ec.LEGACY_DIRECT_TICK_CONFIG

VR =[80000+j for j in range(cc.NR)]        # visual ring neurons (anchoring)
PE =[81000+j for j in range(cc.NR)]        # prediction error: vision vs heading belief
def set_ring(n):
    """Rebuild every ring-sized list at a new angular resolution (see cc.set_NR)."""
    global VR,PE
    cc.set_NR(n); nv.STRIDE=cc.NR//nv.NC
    VR=[80000+j for j in range(cc.NR)]
    PE=[81000+j for j in range(cc.NR)]
    return cc.NR
NU=12; UNC=[82000+i for i in range(NU)]    # uncertainty ladder (epistemic state)
UNC_SEED_SYN=None                          # computed in parts(): drains shift the seed port
MODES=ar.MODES

# V4 tactile-detour contract.  These aliases keep isolated harnesses and the
# exporter independent of import mechanics while the actual populations live
# in modular component files.
OBL, OBR, OBDL, OBDR = obs.OBL, obs.OBR, obs.OBDL, obs.OBDR
OBS_LEFT, OBS_RIGHT, OBS_BRAKE, OBS_WALL = obm.OBS_LEFT, obm.OBS_RIGHT, obm.OBS_BRAKE, obm.OBS_WALL

# Opponent angular-velocity neurons.  They are opt-in while the recurrent
# compass is calibrated.  The graded route low-passes a signed raw gyro; the
# pulse route instead makes sparse, mutually exclusive PAULA vestibular
# events, which gate the existing local P-EN shift only where the bump is.
# IDs follow the vAC/TRISE blocks without overlap.
VEST_CCW, VEST_CW = 87600, 87601
# The comparator route retains the signed low-pass before rectification:
# POS/NEG are non-spiking physical-rate integrators, while CCW/CW are their
# sparse opponent event outputs.  Keep these ids distinct from the graded
# route above so an experiment can ablate one representation cleanly.
VEST_POS, VEST_NEG = 87602, 87603
# Signed delay-line accumulators for the full-stroke notch representation.
VEST_NET_CCW, VEST_NET_CW = 87604, 87605
# Graded opponent readouts of the two notch afferents.  These are kept
# separate from the spiking VEST_CCW/VEST_CW comparators above.
VEST_OPP_CCW, VEST_OPP_CW = 87606, 87607
# Heterogeneous opponent afferents for the population-code experiment.  The
# gap from the other vestibular ids avoids colliding with the lateral-horn
# populations that follow 87620.
VEST_BANK_CCW = [87800 + i for i in range(8)]
VEST_BANK_CW = [87820 + i for i in range(8)]
# Efference populations pool the existing final spiking paddle relays.  They
# are intentionally distinct from the sensory bank: these cells model the
# circuit's predicted self-rotation, while the raw gyro remains a sensor.
EFF_BANK_CCW = [87850 + i for i in range(4)]
EFF_BANK_CW = [87860 + i for i in range(4)]
# Separate, opt-in sensorimotor-estimator populations.  These ids do not
# appear in the demo/default agent.  The path keeps motor prediction,
# body-derived yaw residual, and their PAULA prediction error distinct until
# the final local P-EN update.
SM_SENS_CCW = [88000 + i for i in range(4)]
SM_SENS_CW = [88010 + i for i in range(4)]
SM_PRED_CCW = [88020 + i for i in range(4)]
SM_PRED_CW = [88030 + i for i in range(4)]
SM_PE_CCW, SM_PE_CW = 88040, 88041
SM_UPDATE_CCW, SM_UPDATE_CW = 88042, 88043
# Version two preserves the v1 residual experiment above unchanged.  It uses
# explicit non-negative opponent sensor currents because PAULA's sensory
# input-buffer transport treats negative ``info`` as inactive; inhibition is
# represented by a negative *synaptic weight*, not a negative sensory value.
SM2_SENS_CCW = [88100 + i for i in range(4)]
SM2_SENS_CW = [88110 + i for i in range(4)]
SM2_PRED_CCW = [88120 + i for i in range(4)]
SM2_PRED_CW = [88130 + i for i in range(4)]
SM2_PE_CCW, SM2_PE_CW = 88140, 88141
SM2_UPDATE_CCW, SM2_UPDATE_CW = 88142, 88143
# V3's leading update tract is a phase-distributed PAULA population.  Four
# real CPG phases × six cells provide repeated, bounded update opportunities
# rather than a single static high-gain edge.  These cells are not a claim of
# a literal P-EN subtype count; they are an inspectable circuit hypothesis.
SM3_PHASE_COPIES = 6
SM3_LEAD_CCW = [88200 + i for i in range(4 * SM3_PHASE_COPIES)]
SM3_LEAD_CW = [88240 + i for i in range(4 * SM3_PHASE_COPIES)]
# Diagnostic-only phenomenological sampler (see neuron/extensions/experimental/phase_locked.py).
STROKE_CLOCK, STROKE_CCW, STROKE_CW = 87900, 87901, 87902
STROKE_CLOCK_SYN = None
# Ordinary PAULA candidate: a recurrent CPG clock drives inhibitory reset
# synapses on two signed graded integrators.  Unlike the IDs above, this path
# has no sample/hold subclass or hidden counter.
STROKE_RESET_CLOCK, STROKE_RESET_CCW, STROKE_RESET_CW = 87910, 87911, 87912
STROKE_RESET_CLOCK_SYN = None
# Default-off fast vestibular pair used only by the explicit PB phase-gated
# update experiment. These remain ordinary graded PAULA neurons; the 44-cell
# phase population is declared in central_complex alongside the PB gates.
PHASE_VEST_CCW, PHASE_VEST_CW = 87920, 87921

# Lateral-horn-like learned-valence gates.  MB output encodes whether the
# currently recognised odour is dangerous; this paired population combines
# that state with the *physical* left/right food-odour comparison before the
# existing turn cells.  Its combined output interrupts the innate approach
# route and produces an evasive turn.  It is a normal structural component;
# an explicit configuration switch remains available for causal ablations.
NLH = 6
LH_LEFT = [87620 + i for i in range(NLH)]
LH_RIGHT = [87630 + i for i in range(NLH)]
# Optional experimental ports for a prescribed *neural* turn course.  They
# are absent from the default organism and exist only so a compass component
# test can drive the same TL/TR -> relay -> muscle -> MuJoCo chain without a
# host-side kinematic turn.  Populated by ``navcore_parts`` when requested.
TURN_PROBE_SYN: dict[int, int] = {}

def _nsyn(sy,nid): return sum(1 for s in sy if s.get("neuron_id")==nid and s.get("type")=="postsynaptic")


def _set_conjunctive_velocity_port(ne, target, positive_synapse, negative_synapse=None):
    """Retarget an opt-in P-EN coincidence gate to its actual PAULA input.

    ``central_complex.parts`` reserves synapse 1 for a direct isolated-drive
    harness.  Vestibular builders append real sensory dendrites later, so a
    conjunctive P-EN must be pointed at that appended port.  When a signed
    opponent pair is used, retain the inhibitory partner as well: the
    inherited neuron sums the two local dendritic drives before rectifying.
    """
    synapses = (
        [int(synapse) for synapse in positive_synapse]
        if isinstance(positive_synapse, (list, tuple)) else [int(positive_synapse)]
    )
    for neuron in ne:
        metadata = neuron.get("metadata", {})
        if neuron["id"] == target and metadata.get("conjunctive_graded_gain", 0.0) > 0.0:
            metadata["conjunctive_velocity_synapse"] = synapses[0]
            if len(synapses) > 1:
                metadata["conjunctive_velocity_synapses"] = synapses
            if negative_synapse is not None:
                metadata["conjunctive_velocity_negative_synapse"] = int(negative_synapse)
            return


def vestibular_opponent_parts(ne, sy, conns, ex, tau=60.0, gain=1.0):
    """Attach a PAULA opponent vestibular low-pass to the P-EN velocity ports.

    Raw body yaw contains large alternating rowing transients.  Two graded
    non-spiking cells integrate the positive and negative half-waves; wiring
    their *difference* to each P-EN direction removes the common stroke
    component inside the neural circuit.  The only host-side operation is the
    sanctioned physical yaw-rate sign split into two sensory currents.
    """
    for nid in (VEST_CCW, VEST_CW):
        ne.append(k.neuron(nid, r=1e9, c=2, lam=float(tau),
                           meta={"graded_gain": 1.0, "graded_S0": 0.0}))
        sy.append(k.syn(nid, 0, 1.0, 1)); ex.append(k.ext(nid, 0))
        sy.append(k.term(nid, 20000+nid))
    for i in range(cc.NR):
        for p in range(cc.NP):
            # Positive yaw shifts CCW; negative yaw shifts CW.  Each target
            # sees both analog cells with opposing signs, so common-mode
            # stroke power cannot activate both shift directions together.
            for target, positive, negative in (
                (cc.CL[i][p], VEST_CCW, VEST_CW),
                (cc.CR[i][p], VEST_CW, VEST_CCW),
            ):
                j = _nsyn(sy, target)
                sy.append(k.syn(target, j, abs(gain), 1))
                conns.append(k.conn(positive, target, j, 20000+positive)); j += 1
                sy.append(k.syn(target, j, -abs(gain), 1))
                conns.append(k.conn(negative, target, j, 20000+negative))
                _set_conjunctive_velocity_port(ne, target, j - 1, j)
    return ne, sy, conns, ex


def vestibular_opponent_bank_parts(ne, sy, conns, ex,
                                   taus=(8.0, 12.0, 18.0, 28.0, 44.0, 68.0, 104.0, 160.0),
                                   gain=1.0):
    """Population-code raw gyro through heterogeneous PAULA opponent pairs.

    Each pair receives only one signed sensory current, then smooths it with
    its own membrane constant.  Every P-EN receives the *normalized signed
    sum* of all pairs at local dendrites.  This increases temporal precision
    through a heterogeneous population rather than moving a filter to the
    host or supplying a hidden heading value.
    """
    taus = tuple(float(tau) for tau in taus)
    if not taus or len(taus) > len(VEST_BANK_CCW) or any(tau < 1.0 for tau in taus):
        raise ValueError("vop_bank_taus must contain 1..8 PAULA time constants >= 1")
    for ccw, cw, tau in zip(VEST_BANK_CCW[:len(taus)], VEST_BANK_CW[:len(taus)], taus, strict=True):
        for nid in (ccw, cw):
            ne.append(k.neuron(nid, r=1e9, c=2, lam=tau,
                               meta={"graded_gain": 1.0, "graded_S0": 0.0}))
            sy.append(k.syn(nid, 0, 1.0, 1)); ex.append(k.ext(nid, 0))
            sy.append(k.term(nid, 20000 + nid))
    pair_gain = abs(float(gain)) / len(taus)
    for i in range(cc.NR):
        for p in range(cc.NP):
            for target, own, other in (
                (cc.CL[i][p], VEST_BANK_CCW, VEST_BANK_CW),
                (cc.CR[i][p], VEST_BANK_CW, VEST_BANK_CCW),
            ):
                synapses = []
                for positive, negative in zip(own[:len(taus)], other[:len(taus)], strict=True):
                    j = _nsyn(sy, target)
                    sy.append(k.syn(target, j, pair_gain, 1))
                    conns.append(k.conn(positive, target, j, 20000 + positive)); synapses.append(j)
                    j += 1
                    sy.append(k.syn(target, j, -pair_gain, 1))
                    conns.append(k.conn(negative, target, j, 20000 + negative)); synapses.append(j)
                _set_conjunctive_velocity_port(ne, target, synapses)
    return ne, sy, conns, ex


def relay_efference_population_parts(ne, sy, conns,
                                     taus=(6.0, 12.0, 24.0, 48.0),
                                     input_gain=6.0, pen_gain=1.0):
    """Pool motor-relay spikes into a signed P-EN efference population.

    The body yaw is generated by the relay populations' asymmetric paddle
    activity.  Rather than pass a host-side command to the compass, this
    creates heterogeneous graded PAULA cells from those spikes and sends
    their normalized signed difference to the existing local P-EN gate.
    """
    taus = tuple(float(tau) for tau in taus)
    if not taus or len(taus) > len(EFF_BANK_CCW) or any(tau < 1.0 for tau in taus):
        raise ValueError("relay_efference_taus must contain 1..4 time constants >= 1")
    # The world model establishes this polarity: ML paddles make CCW yaw and
    # MR paddles make CW yaw.  Each pool spans both stroke phases on its side.
    sources = (
        (EFF_BANK_CCW[:len(taus)], (MLp, MLr)),
        (EFF_BANK_CW[:len(taus)], (MRp, MRr)),
    )
    relay_count = 2 * NRLY
    for targets, muscles in sources:
        relay_sources = [relay for muscle in muscles for relay in RLY[muscle]]
        for nid, tau in zip(targets, taus, strict=True):
            ne.append(k.neuron(nid, r=1e9, c=2, lam=tau,
                               meta={"graded_gain": 1.0, "graded_S0": 0.0}))
            for synapse_id, source in enumerate(relay_sources):
                sy.append(k.syn(nid, synapse_id, float(input_gain) / relay_count, 1))
                conns.append(k.conn(source, nid, synapse_id, 20000 + source))
            sy.append(k.term(nid, 20000 + nid))
    pair_gain = abs(float(pen_gain)) / len(taus)
    for i in range(cc.NR):
        for p in range(cc.NP):
            for target, own, other in (
                (cc.CL[i][p], EFF_BANK_CCW[:len(taus)], EFF_BANK_CW[:len(taus)]),
                (cc.CR[i][p], EFF_BANK_CW[:len(taus)], EFF_BANK_CCW[:len(taus)]),
            ):
                synapses = []
                for positive, negative in zip(own, other, strict=True):
                    j = _nsyn(sy, target)
                    sy.append(k.syn(target, j, pair_gain, 1))
                    conns.append(k.conn(positive, target, j, 20000 + positive)); synapses.append(j)
                    j += 1
                    sy.append(k.syn(target, j, -pair_gain, 1))
                    conns.append(k.conn(negative, target, j, 20000 + negative)); synapses.append(j)
                _set_conjunctive_velocity_port(ne, target, synapses)
    return ne, sy, conns


def sensorimotor_estimator_parts(
    ne, sy, conns, ex,
    sensory_windows=(36, 40, 44, 48), prediction_taus=(6.0, 12.0, 24.0, 48.0),
    sensory_input_gain=1000.0, sensory_weight=1.0,
    prediction_input_gain=6.0, prediction_weight=1.0,
    error_weight=1.0, update_gain=1.0, update_lam=4.0,
    pen_gain=1.0,
):
    """Build an all-PAULA complementary heading-update pathway.

    The existing relay-efference experiment replaced raw yaw with motor
    activity.  This candidate instead preserves both causal streams:

    * a heterogeneous, full-stroke sensory-residual bank receives only raw
      signed yaw current;
    * a heterogeneous motor-prediction bank receives only the final spiking
      paddle relays; and
    * paired ordinary graded prediction-error cells express the signed
      sensory-minus-prediction residual before local P-EN update cells combine
      it with the predicted turn.

    A PAULA prediction error is therefore a neural signal, not a Python
    subtraction.  The final update remains conjunctive with the local E-PG
    bump when ``conjunctive_graded_shift`` is enabled.  It is an experimental
    sensorimotor architecture inspired by corollary-discharge and sensory
    correction principles; it is not asserted to be a literal fly connectome.
    """
    sensory_windows = tuple(int(window) for window in sensory_windows)
    prediction_taus = tuple(float(tau) for tau in prediction_taus)
    if (not sensory_windows or len(sensory_windows) > len(SM_SENS_CCW)
            or any(window < 2 for window in sensory_windows)):
        raise ValueError("sensorimotor sensory_windows must contain 1..4 windows >= 2")
    if (not prediction_taus or len(prediction_taus) > len(SM_PRED_CCW)
            or any(tau < 1.0 for tau in prediction_taus)):
        raise ValueError("sensorimotor prediction_taus must contain 1..4 time constants >= 1")
    if min(float(sensory_input_gain), float(sensory_weight),
           float(prediction_input_gain), float(prediction_weight),
           float(error_weight), float(update_gain), float(pen_gain)) < 0.0 or float(update_lam) <= 0.0:
        raise ValueError("sensorimotor estimator gains must be nonnegative and update_lam positive")
    if float(sensory_weight) == 0.0 and float(prediction_weight) == 0.0:
        raise ValueError("sensorimotor estimator requires sensory or prediction drive")

    # A delayed negative copy on the same raw-yaw dendrite is a finite
    # full-stroke residual.  Different windows are a population representation
    # of uncertain gait phase, rather than a host-side chosen smoothing time.
    for targets, sign in ((SM_SENS_CCW[:len(sensory_windows)], 1.0),
                          (SM_SENS_CW[:len(sensory_windows)], -1.0)):
        for nid, window in zip(targets, sensory_windows, strict=True):
            ne.append(k.neuron(nid, r=1e9, c=2, lam=100000.0,
                               meta={"graded_gain": 1.0, "graded_S0": 0.0,
                                     "sensorimotor_role": "sensory_residual",
                                     "sensorimotor_sign": "CCW" if sign > 0 else "CW",
                                     "sensorimotor_window": window}))
            sy.append(k.syn(nid, 0, sign * float(sensory_input_gain), 1)); ex.append(k.ext(nid, 0))
            sy.append(k.syn(nid, 1, -sign * float(sensory_input_gain), window)); ex.append(k.ext(nid, 1))
            sy.append(k.term(nid, 20000 + nid))

    # The relays are the last spiking population upstream of the graded
    # muscles.  They are the available PAULA corollary-discharge source and
    # predict the sign of motor-generated yaw before the raw sensory residual
    # has accumulated.
    relay_count = 2 * NRLY
    for targets, muscles, label in (
        (SM_PRED_CCW[:len(prediction_taus)], (MLp, MLr), "CCW"),
        (SM_PRED_CW[:len(prediction_taus)], (MRp, MRr), "CW"),
    ):
        relay_sources = [relay for muscle in muscles for relay in RLY[muscle]]
        for nid, tau in zip(targets, prediction_taus, strict=True):
            ne.append(k.neuron(nid, r=1e9, c=2, lam=tau,
                               meta={"graded_gain": 1.0, "graded_S0": 0.0,
                                     "sensorimotor_role": "motor_prediction",
                                     "sensorimotor_sign": label,
                                     "sensorimotor_tau": tau}))
            for sid, source in enumerate(relay_sources):
                sy.append(k.syn(nid, sid, float(prediction_input_gain) / relay_count, 1))
                conns.append(k.conn(source, nid, sid, 20000 + source))
            sy.append(k.term(nid, 20000 + nid))

    # Each error population computes a signed local residual entirely through
    # dendritic weights: sensed own direction minus sensed opponent direction,
    # then minus predicted own direction plus predicted opponent direction.
    # Its rectified graded release is a correction in the direction whose
    # physical rotation exceeded the neural motor prediction.
    for nid, own_sens, opp_sens, own_pred, opp_pred, label in (
        (SM_PE_CCW, SM_SENS_CCW[:len(sensory_windows)], SM_SENS_CW[:len(sensory_windows)],
         SM_PRED_CCW[:len(prediction_taus)], SM_PRED_CW[:len(prediction_taus)], "CCW"),
        (SM_PE_CW, SM_SENS_CW[:len(sensory_windows)], SM_SENS_CCW[:len(sensory_windows)],
         SM_PRED_CW[:len(prediction_taus)], SM_PRED_CCW[:len(prediction_taus)], "CW"),
    ):
        ne.append(k.neuron(nid, r=1e9, c=2, lam=float(update_lam),
                           meta={"graded_gain": 1.0, "graded_S0": 0.0,
                                 "sensorimotor_role": "sensory_prediction_error",
                                 "sensorimotor_sign": label}))
        sid = 0
        for source, weight in (
            *((source, float(sensory_weight) / len(own_sens)) for source in own_sens),
            *((source, -float(sensory_weight) / len(opp_sens)) for source in opp_sens),
            *((source, -float(prediction_weight) / len(own_pred)) for source in own_pred),
            *((source, float(prediction_weight) / len(opp_pred)) for source in opp_pred),
        ):
            sy.append(k.syn(nid, sid, weight, 1))
            conns.append(k.conn(source, nid, sid, 20000 + source)); sid += 1
        sy.append(k.term(nid, 20000 + nid))

    # Prediction plus its same-signed sensory residual.  Opponent prediction
    # is inhibitory, so a symmetric paddle rhythm does not masquerade as a
    # turn. The P-EN sees only this explicit neural update population.
    for nid, own_pred, opp_pred, own_error, label in (
        (SM_UPDATE_CCW, SM_PRED_CCW[:len(prediction_taus)], SM_PRED_CW[:len(prediction_taus)], SM_PE_CCW, "CCW"),
        (SM_UPDATE_CW, SM_PRED_CW[:len(prediction_taus)], SM_PRED_CCW[:len(prediction_taus)], SM_PE_CW, "CW"),
    ):
        ne.append(k.neuron(nid, r=1e9, c=2, lam=float(update_lam),
                           meta={"graded_gain": float(update_gain), "graded_S0": 0.0,
                                 "sensorimotor_role": "fused_heading_update",
                                 "sensorimotor_sign": label}))
        sid = 0
        for source in own_pred:
            sy.append(k.syn(nid, sid, float(prediction_weight) / len(own_pred), 1))
            conns.append(k.conn(source, nid, sid, 20000 + source)); sid += 1
        for source in opp_pred:
            sy.append(k.syn(nid, sid, -float(prediction_weight) / len(opp_pred), 1))
            conns.append(k.conn(source, nid, sid, 20000 + source)); sid += 1
        sy.append(k.syn(nid, sid, float(error_weight), 1))
        conns.append(k.conn(own_error, nid, sid, 20000 + own_error))
        sy.append(k.term(nid, 20000 + nid))

    for i in range(cc.NR):
        for p in range(cc.NP):
            for target, source in ((cc.CL[i][p], SM_UPDATE_CCW),
                                   (cc.CR[i][p], SM_UPDATE_CW)):
                sid = _nsyn(sy, target)
                sy.append(k.syn(target, sid, float(pen_gain), 1))
                conns.append(k.conn(source, target, sid, 20000 + source))
                _set_conjunctive_velocity_port(ne, target, sid)
    return ne, sy, conns, ex


def sensorimotor_estimator_v2_parts(
    ne, sy, conns, ex,
    sensory_taus=(6.0, 12.0, 24.0, 48.0), prediction_taus=(6.0, 12.0, 24.0, 48.0),
    sensory_input_gain=0.12, sensory_weight=1.0,
    prediction_input_gain=6.0, prediction_weight=1.0,
    error_weight=1.0, update_gain=1.0, update_lam=4.0,
    pen_gain=1.0, phase_gated_lead=False, phase_copies=SM3_PHASE_COPIES,
    phase_gate_gain=1.0, phase_gate_tau=1.0,
    p_enb_trailing=False, p_enb_update_gain=1.0,
    p_ena_bank=False, p_ena_update_gain=1.0,
):
    """Build the corrected PAULA sensorimotor heading-update candidate.

    Version one deliberately established a causal test in which motor
    prediction and body yaw were distinct.  Its signed raw-yaw port was not a
    valid PAULA sensory representation, though: transport only schedules
    positive input-buffer ``info`` values.  A negative signed current is
    therefore absent rather than an inhibitory sensory event.  V2 explicitly
    gives each direction two *non-negative* physical-current ports and makes
    the opponent relation with ordinary negative dendritic weights.

    Each sensory cell is a leaky, signed self-motion estimate: own-direction
    yaw excites it, opposite-direction yaw inhibits it.  Heterogeneous time
    constants retain both rapid and stroke-scale evidence without a host-side
    smoother.  Separate motor prediction, bidirectional prediction-error,
    and fused-update populations remain all PAULA signals.  This is still an
    experimental architecture inspired by biological corollary discharge and
    sensory correction, not an assertion of a literal insect cell type.
    """
    sensory_taus = tuple(float(tau) for tau in sensory_taus)
    prediction_taus = tuple(float(tau) for tau in prediction_taus)
    if (not sensory_taus or len(sensory_taus) > len(SM2_SENS_CCW)
            or any(tau < 1.0 for tau in sensory_taus)):
        raise ValueError("sensorimotor_v2 sensory_taus must contain 1..4 values >= 1")
    if (not prediction_taus or len(prediction_taus) > len(SM2_PRED_CCW)
            or any(tau < 1.0 for tau in prediction_taus)):
        raise ValueError("sensorimotor_v2 prediction_taus must contain 1..4 values >= 1")
    parameters = (sensory_input_gain, sensory_weight, prediction_input_gain,
                  prediction_weight, error_weight, update_gain, pen_gain,
                  p_enb_update_gain, p_ena_update_gain)
    if min(float(value) for value in parameters) < 0.0 or float(update_lam) <= 0.0:
        raise ValueError("sensorimotor_v2 gains must be nonnegative and update_lam positive")
    if float(sensory_weight) == 0.0 and float(prediction_weight) == 0.0:
        raise ValueError("sensorimotor_v2 requires sensory or prediction drive")
    if bool(phase_gated_lead) and (not 1 <= int(phase_copies) <= SM3_PHASE_COPIES):
        raise ValueError(f"sensorimotor_v2 phase_copies must be 1..{SM3_PHASE_COPIES}")
    if float(phase_gate_gain) < 0.0:
        raise ValueError("sensorimotor_v2 phase_gate_gain must be nonnegative")
    if float(phase_gate_tau) < 1.0:
        raise ValueError("sensorimotor_v2 phase_gate_tau must be at least one")

    # Input port 0 is own-direction physical yaw and port 1 is the opposing
    # physical yaw.  Both external values are always >=0; the signed local
    # evidence arises at the cell's ordinary excitatory/inhibitory dendrites.
    for targets, label in ((SM2_SENS_CCW[:len(sensory_taus)], "CCW"),
                           (SM2_SENS_CW[:len(sensory_taus)], "CW")):
        for nid, tau in zip(targets, sensory_taus, strict=True):
            ne.append(k.neuron(nid, r=1e9, c=2, lam=tau,
                               meta={"graded_gain": 1.0, "graded_S0": 0.0,
                                     "sensorimotor_role": "signed_sensory_self_motion",
                                     "sensorimotor_sign": label,
                                     "sensorimotor_tau": tau}))
            sy.append(k.syn(nid, 0, float(sensory_input_gain), 1)); ex.append(k.ext(nid, 0))
            sy.append(k.syn(nid, 1, -float(sensory_input_gain), 1)); ex.append(k.ext(nid, 1))
            sy.append(k.term(nid, 20000 + nid))

    relay_count = 2 * NRLY
    for targets, muscles, label in (
        (SM2_PRED_CCW[:len(prediction_taus)], (MLp, MLr), "CCW"),
        (SM2_PRED_CW[:len(prediction_taus)], (MRp, MRr), "CW"),
    ):
        relay_sources = [relay for muscle in muscles for relay in RLY[muscle]]
        for nid, tau in zip(targets, prediction_taus, strict=True):
            ne.append(k.neuron(nid, r=1e9, c=2, lam=tau,
                               meta={"graded_gain": 1.0, "graded_S0": 0.0,
                                     "sensorimotor_role": "motor_prediction",
                                     "sensorimotor_sign": label,
                                     "sensorimotor_tau": tau}))
            for sid, source in enumerate(relay_sources):
                sy.append(k.syn(nid, sid, float(prediction_input_gain) / relay_count, 1))
                conns.append(k.conn(source, nid, sid, 20000 + source))
            sy.append(k.term(nid, 20000 + nid))

    # A signed residual is realised as two rectified PAULA populations:
    #   e_ccw = sens_ccw - sens_cw - pred_ccw + pred_cw
    #   e_cw  = -e_ccw.
    # This avoids treating a negative ``info`` current as a synaptic signal.
    for nid, own_sens, opp_sens, own_pred, opp_pred, label in (
        (SM2_PE_CCW, SM2_SENS_CCW[:len(sensory_taus)], SM2_SENS_CW[:len(sensory_taus)],
         SM2_PRED_CCW[:len(prediction_taus)], SM2_PRED_CW[:len(prediction_taus)], "CCW"),
        (SM2_PE_CW, SM2_SENS_CW[:len(sensory_taus)], SM2_SENS_CCW[:len(sensory_taus)],
         SM2_PRED_CW[:len(prediction_taus)], SM2_PRED_CCW[:len(prediction_taus)], "CW"),
    ):
        ne.append(k.neuron(nid, r=1e9, c=2, lam=float(update_lam),
                           meta={"graded_gain": 1.0, "graded_S0": 0.0,
                                 "sensorimotor_role": "sensory_prediction_error",
                                 "sensorimotor_sign": label}))
        sid = 0
        for source, weight in (
            *((source, float(sensory_weight) / len(own_sens)) for source in own_sens),
            *((source, -float(sensory_weight) / len(opp_sens)) for source in opp_sens),
            *((source, -float(prediction_weight) / len(own_pred)) for source in own_pred),
            *((source, float(prediction_weight) / len(opp_pred)) for source in opp_pred),
        ):
            sy.append(k.syn(nid, sid, weight, 1))
            conns.append(k.conn(source, nid, sid, 20000 + source)); sid += 1
        sy.append(k.term(nid, 20000 + nid))

    # The fused opponent pair receives the signed motor prediction and *both*
    # residual halves.  Hence a CW residual can reduce the CCW channel as
    # well as strengthen CW; V1 only added its same-signed residual and could
    # leave an opposing prediction alive.
    for nid, own_pred, opp_pred, own_error, opp_error, label in (
        (SM2_UPDATE_CCW, SM2_PRED_CCW[:len(prediction_taus)], SM2_PRED_CW[:len(prediction_taus)],
         SM2_PE_CCW, SM2_PE_CW, "CCW"),
        (SM2_UPDATE_CW, SM2_PRED_CW[:len(prediction_taus)], SM2_PRED_CCW[:len(prediction_taus)],
         SM2_PE_CW, SM2_PE_CCW, "CW"),
    ):
        ne.append(k.neuron(nid, r=1e9, c=2, lam=float(update_lam),
                           meta={"graded_gain": float(update_gain), "graded_S0": 0.0,
                                 "sensorimotor_role": "fused_heading_update",
                                 "sensorimotor_sign": label}))
        sid = 0
        for source, weight in (
            *((source, float(prediction_weight) / len(own_pred)) for source in own_pred),
            *((source, -float(prediction_weight) / len(opp_pred)) for source in opp_pred),
            (own_error, float(error_weight)),
            (opp_error, -float(error_weight)),
        ):
            sy.append(k.syn(nid, sid, weight, 1))
            conns.append(k.conn(source, nid, sid, 20000 + source)); sid += 1
        sy.append(k.term(nid, 20000 + nid))

    if phase_gated_lead:
        # Experimental P-ENa-like leading tract.  An ordinary two-dendrite
        # PAULA coincidence cell receives the already fused signed update and
        # one *real* CPG phase.  Four phase groups make an update opportunity
        # every motor quarter-cycle; parallel cells scale the weak analog
        # population by neuron count rather than a host-side gain controller.
        # There is no sample/hold, counter, or Python phase branch here.
        sources_by_direction = {}
        for targets, update, label in (
            (SM3_LEAD_CCW, SM2_UPDATE_CCW, "CCW"),
            (SM3_LEAD_CW, SM2_UPDATE_CW, "CW"),
        ):
            selected = []
            for phase, cpg_source in enumerate(CPGP):
                for copy in range(int(phase_copies)):
                    nid = targets[phase * SM3_PHASE_COPIES + copy]
                    selected.append(nid)
                    ne.append(k.neuron(
                        nid, r=1e9, c=2, lam=1.0,
                        meta={
                            "graded_gain": float(phase_gate_gain),
                            "graded_S0": 0.0,
                            "conjunctive_graded_gain": float(phase_gate_gain),
                            "conjunctive_ring_synapse": 0,
                            "conjunctive_velocity_synapse": 1,
                            # This is a normal first-order PAULA dendritic
                            # trace, not the diagnostic phase-locked
                            # sample/hold subclass. It makes adjacent CPG
                            # phase opportunities overlap smoothly.
                            "conjunctive_velocity_tau": float(phase_gate_tau),
                            "sensorimotor_role": "phase_gated_leading_update",
                            "sensorimotor_sign": label,
                            "motor_phase": phase,
                            "parallel_copy": copy,
                        },
                    ))
                    sy.append(k.syn(nid, 0, 1.0, 1))
                    conns.append(k.conn(update, nid, 0, 20000 + update))
                    sy.append(k.syn(nid, 1, 1.0, 1))
                    conns.append(k.conn(cpg_source, nid, 1, 20000 + cpg_source))
                    sy.append(k.term(nid, 20000 + nid))
            sources_by_direction[label] = selected
        pen_sources = ((cc.CL, sources_by_direction["CCW"]),
                       (cc.CR, sources_by_direction["CW"]))
    else:
        pen_sources = ((cc.CL, [SM2_UPDATE_CCW]), (cc.CR, [SM2_UPDATE_CW]))

    for columns, sources in pen_sources:
        for column in columns:
            for target in column:
                synapses = []
                for source in sources:
                    sid = _nsyn(sy, target)
                    sy.append(k.syn(target, sid, float(pen_gain), 1))
                    conns.append(k.conn(source, target, sid, 20000 + source))
                    synapses.append(sid)
                _set_conjunctive_velocity_port(ne, target, synapses)

    if p_enb_trailing:
        # The optional trailing route is deliberately fed by the *same*
        # PAULA fused self-motion estimate as the leading P-ENa-like route.
        # It does not receive decoded heading, raw pose, or a Python turn
        # command.  Its other dendrite was built locally from P-EG in the
        # PB/EB topology, so both sources and their delays are explicit.
        for targets, source in ((cc.PENB_CL, SM2_UPDATE_CCW),
                                (cc.PENB_CR, SM2_UPDATE_CW)):
            for target in targets:
                sid = _nsyn(sy, target)
                sy.append(k.syn(target, sid, float(p_enb_update_gain), 1))
                conns.append(k.conn(source, target, sid, 20000 + source))
                _set_conjunctive_velocity_port(ne, target, sid)

    if p_ena_bank:
        # The P-ENa micro-bank is a real scaled PAULA population, not a
        # scalar gain on the old P-EN terminals.  All copies share the
        # circuit's signed fused update and retain their local E-PG dendrite;
        # their heterogeneous local time constants were declared by
        # ``central_complex.parts``.
        for columns, source in ((cc.PENA_CL, SM2_UPDATE_CCW),
                                (cc.PENA_CR, SM2_UPDATE_CW)):
            for column in columns:
                for target in column:
                    sid = _nsyn(sy, target)
                    sy.append(k.syn(target, sid, float(p_ena_update_gain), 1))
                    conns.append(k.conn(source, target, sid, 20000 + source))
                    _set_conjunctive_velocity_port(ne, target, sid)
    return ne, sy, conns, ex


def vestibular_phase_locked_parts(ne, sy, conns, ex, period=44,
                                  input_gain=1.0, output_gain=0.02,
                                  pen_gain=1.0, clock_threshold=0.0, hold_ticks=0):
    """Build an *experimental phenomenological* PAULA stroke estimator.

    A self-recurrent clock, seeded only at birth, marks a complete mechanical
    stroke.  Two inherited phase-locked graded neurons sum opposite signed
    gyro dendrites between clock events and hold their positive result until
    the next event.  P-ENs receive their local signed difference.

    The clock and inputs are network-internal, but the subclass's discrete
    sample/reset/hold state is an engineered diagnostic primitive, not a
    biologically faithful P-EN or central-complex model.  Do not enable this
    in an accepted organism configuration.  The ordinary candidate is the
    explicit CPG/interneuron/inhibitory-reset construction.
    """
    global STROKE_CLOCK_SYN
    if int(period) < 2:
        raise ValueError("experimental phase-locked stroke period must be at least two ticks")
    # The one-time external birth seed starts a completely neural 44-tick
    # oscillator. Its delayed self-excitation is the clock thereafter.
    ne.append(k.neuron(STROKE_CLOCK, r=0.6, c=2, lam=3))
    # A PAULA dendritic distance ``d`` is observed one neural tick after the
    # source event, so d=period-1 yields the requested event-to-event period.
    sy.append(k.syn(STROKE_CLOCK, 0, 4.0, int(period) - 1))
    conns.append(k.conn(STROKE_CLOCK, STROKE_CLOCK, 0, 20000 + STROKE_CLOCK))
    STROKE_CLOCK_SYN = 1
    sy.append(k.syn(STROKE_CLOCK, STROKE_CLOCK_SYN, 1.0, 1)); ex.append(k.ext(STROKE_CLOCK, STROKE_CLOCK_SYN))
    sy.append(k.term(STROKE_CLOCK, 20000 + STROKE_CLOCK))
    # Both cells use the same local signed-difference morphology.  Their
    # direction is determined by which raw half-wave is routed to synapse 0
    # versus synapse 1 by the sensory transducer below.
    for nid in (STROKE_CCW, STROKE_CW):
        ne.append(k.neuron(
            nid, r=1e9, c=2, lam=100000.0,
            meta={
                "graded_gain": 1.0,
                "phase_locked_gain": float(output_gain),
                "phase_locked_positive_synapse": 0,
                "phase_locked_negative_synapse": 1,
                "phase_locked_clock_synapse": 2,
                "phase_locked_clock_threshold": float(clock_threshold),
                "phase_locked_hold_ticks": int(hold_ticks),
            },
        ))
        sy.append(k.syn(nid, 0, abs(float(input_gain)), 1)); ex.append(k.ext(nid, 0))
        sy.append(k.syn(nid, 1, -abs(float(input_gain)), 1)); ex.append(k.ext(nid, 1))
        sy.append(k.syn(nid, 2, 1.0, 1))
        conns.append(k.conn(STROKE_CLOCK, nid, 2, 20000 + STROKE_CLOCK))
        sy.append(k.term(nid, 20000 + nid))
    for i in range(cc.NR):
        for p in range(cc.NP):
            for target, positive, negative in (
                (cc.CL[i][p], STROKE_CCW, STROKE_CW),
                (cc.CR[i][p], STROKE_CW, STROKE_CCW),
            ):
                j = _nsyn(sy, target)
                sy.append(k.syn(target, j, abs(float(pen_gain)), 1))
                conns.append(k.conn(positive, target, j, 20000 + positive)); positive_synapse = j
                j += 1
                sy.append(k.syn(target, j, -abs(float(pen_gain)), 1))
                conns.append(k.conn(negative, target, j, 20000 + negative)); negative_synapse = j
                _set_conjunctive_velocity_port(ne, target, positive_synapse, negative_synapse)
    return ne, sy, conns, ex


def vestibular_stroke_reset_parts(ne, sy, conns, ex, period=44,
                                  input_gain=1.0, reset_gain=40.0,
                                  integrator_lam=44.0, output_gain=1.0,
                                  output_S0=0.0,
                                  pen_gain=1.0, cross_inhib=0.0,
                                  pb_stroke_reset=False, pb_reset_gain=0.0,
                                  pb_phase_update=False, pb_phase_start=11,
                                  pb_phase_sensor_gain=1.0, pb_phase_sensor_max=4.0):
    """Build a stroke-segmented gyro path from ordinary PAULA elements.

    A recurrent spiking clock is an explicit CPG source.  Each ordinary
    graded integrator receives positive and negative gyro half-waves on two
    dendrites, then a delayed *inhibitory* clock synapse.  The clock does not
    snapshot or hold an internal value: it merely resets accumulated membrane
    potential through the same synaptic/membrane dynamics used everywhere
    else.  P-ENs get the signed difference through their existing local
    coincidence dendrites.

    This is a circuit hypothesis to test, not a claim that the particular
    parameters reproduce a named insect cell type.
    """
    global STROKE_RESET_CLOCK_SYN
    if int(period) < 2:
        raise ValueError("stroke-reset period must be at least two ticks")
    if float(integrator_lam) <= 0.0:
        raise ValueError("stroke-reset integrator time constant must be positive")
    # One initial external seed, followed exclusively by delayed PAULA
    # self-excitation.  ``period - 1`` gives a measured event spacing of
    # ``period`` under PAULA's arrival scheduling.
    ne.append(k.neuron(STROKE_RESET_CLOCK, r=0.6, c=2, lam=3))
    sy.append(k.syn(STROKE_RESET_CLOCK, 0, 4.0, int(period) - 1))
    conns.append(k.conn(STROKE_RESET_CLOCK, STROKE_RESET_CLOCK, 0,
                        20000 + STROKE_RESET_CLOCK))
    STROKE_RESET_CLOCK_SYN = 1
    sy.append(k.syn(STROKE_RESET_CLOCK, STROKE_RESET_CLOCK_SYN, 1.0, 1))
    ex.append(k.ext(STROKE_RESET_CLOCK, STROKE_RESET_CLOCK_SYN))
    sy.append(k.term(STROKE_RESET_CLOCK, 20000 + STROKE_RESET_CLOCK))

    for nid in (STROKE_RESET_CCW, STROKE_RESET_CW):
        ne.append(k.neuron(
            nid, r=1e9, c=2, lam=float(integrator_lam),
            # The equal clock-induced baseline is subtracted downstream by
            # the signed opponent pair.  ``output_S0`` may expose it as
            # common-mode graded release without adding a stateful floor or
            # reset exception to the PAULA cell.
            meta={"graded_gain": float(output_gain), "graded_S0": float(output_S0)},
        ))
        # Ordinary positive and inhibitory sensory dendrites.
        sy.append(k.syn(nid, 0, abs(float(input_gain)), 1)); ex.append(k.ext(nid, 0))
        sy.append(k.syn(nid, 1, -abs(float(input_gain)), 1)); ex.append(k.ext(nid, 1))
        # Explicit inhibitory reset from the clock interneuron.  The gain is
        # deliberately a normal synapse weight, not a branch in neuron.tick.
        sy.append(k.syn(nid, 2, -abs(float(reset_gain)), 1))
        conns.append(k.conn(STROKE_RESET_CLOCK, nid, 2,
                            20000 + STROKE_RESET_CLOCK))
        sy.append(k.term(nid, 20000 + nid))

    if float(cross_inhib) > 0.0:
        # Optional ordinary opponent interneuronal inhibition.  It is kept
        # off by default: a configuration that enables it is a circuit
        # hypothesis to evaluate, rather than an asserted P-EN mechanism.
        for source, target in (
            (STROKE_RESET_CCW, STROKE_RESET_CW),
            (STROKE_RESET_CW, STROKE_RESET_CCW),
        ):
            sy.append(k.syn(target, 3, -abs(float(cross_inhib)), 1))
            conns.append(k.conn(source, target, 3, 20000 + source))

    if pb_stroke_reset:
        if float(pb_reset_gain) <= 0.0:
            raise ValueError("pb_stroke_reset requires a positive pb_reset_gain")
        # The PB bridge relays are ordinary graded PAULA cells.  Their slow
        # membrane is useful for cancelling alternating gait half-strokes but
        # must not grow across a sustained turn.  The existing recurrent
        # reset-clock interneuron supplies an ordinary inhibitory synapse at
        # each stroke boundary—no counter, snapshot, or host-side averaging.
        for nid in cc.PB_CL + cc.PB_CR:
            sid=_nsyn(sy,nid)
            sy.append(k.syn(nid,sid,-abs(float(pb_reset_gain)),1))
            conns.append(k.conn(STROKE_RESET_CLOCK,nid,sid,20000 + STROKE_RESET_CLOCK))

    if pb_phase_update:
        if int(period) != len(cc.PB_PHASE_CLOCK):
            raise ValueError("pb_phase_update currently requires the 44-tick explicit phase population")
        if not cc.PB_PHASE_SENSOR_SYNS or not cc.PB_PHASE_CLOCK_SYNS:
            raise ValueError("pb_phase_update requires PB phase cells from central_complex.parts")
        if not 0 <= int(pb_phase_start) < int(period):
            raise ValueError("pb_phase_start must be within one CPG period")
        if float(pb_phase_sensor_gain) <= 0.0 or float(pb_phase_sensor_max) <= 0.0:
            raise ValueError("pb_phase_update requires positive fast-sensor gain and cap")
        # These two ordinary graded PAULA cells are fast signed vestibular
        # afferents. Their bounded tonic release preserves the local direction
        # signal at a selected gait phase instead of accumulating an entire
        # stroke in a special sampler. The cap is an ordinary graded-neuron
        # output range, not a host normalisation or a heading estimate.
        for nid in (PHASE_VEST_CCW, PHASE_VEST_CW):
            ne.append(k.neuron(
                nid, r=1e9, c=2, lam=1.0,
                meta={"graded_gain": float(pb_phase_sensor_gain), "graded_S0": 0.0,
                      "graded_max": float(pb_phase_sensor_max), "pb_phase_fast_vestibular": True},
            ))
            sy.append(k.syn(nid, 0, 1.0, 1)); ex.append(k.ext(nid, 0))
            sy.append(k.term(nid, 20000 + nid))
        # A clock spike fans out through 44 dendritic delays, giving one
        # normal PAULA phase interneuron at each gait tick.  The PB gates
        # receive only their declared three-cell phase window; no Python
        # counter, sampled value, or special neuron state participates.
        for phase, nid in enumerate(cc.PB_PHASE_CLOCK):
            ne.append(k.neuron(nid, r=0.6, c=2, lam=3,
                               meta={"pb_phase_index": phase, "pb_phase_clock": True}))
            sy.append(k.syn(nid, 0, 4.0, phase + 1))
            conns.append(k.conn(STROKE_RESET_CLOCK, nid, 0, 20000 + STROKE_RESET_CLOCK))
            sy.append(k.term(nid, 20000 + nid))
        for gates, sensor in ((cc.PB_PHASE_CL, PHASE_VEST_CCW),
                              (cc.PB_PHASE_CR, PHASE_VEST_CW)):
            for nid in gates:
                sid=cc.PB_PHASE_SENSOR_SYNS[nid]
                conns.append(k.conn(sensor, nid, sid, 20000 + sensor))
                for offset, clock_sid in enumerate(cc.PB_PHASE_CLOCK_SYNS[nid]):
                    phase=(int(pb_phase_start) + offset) % int(period)
                    clock_nid=cc.PB_PHASE_CLOCK[phase]
                    conns.append(k.conn(clock_nid, nid, clock_sid, 20000 + clock_nid))

    for i in range(cc.NR):
        for p in range(cc.NP):
            for target, positive, negative in (
                (cc.CL[i][p], STROKE_RESET_CCW, STROKE_RESET_CW),
                (cc.CR[i][p], STROKE_RESET_CW, STROKE_RESET_CCW),
            ):
                j = _nsyn(sy, target)
                sy.append(k.syn(target, j, abs(float(pen_gain)), 1))
                conns.append(k.conn(positive, target, j, 20000 + positive))
                positive_synapse = j
                j += 1
                sy.append(k.syn(target, j, -abs(float(pen_gain)), 1))
                conns.append(k.conn(negative, target, j, 20000 + negative))
                _set_conjunctive_velocity_port(ne, target, positive_synapse, j)
    return ne, sy, conns, ex


def vestibular_pulse_parts(ne, sy, conns, ex, tau=30.0, r=2.0,
                           input_gain=0.04, gain=1.4, cross_inhib=1.5):
    """Attach a PAULA event-rate vestibular transducer to the P-EN gates.

    The raw MuJoCo gyro has large, alternating rowing transients.  The older
    graded opponent integrates those values but still supplies a continuous
    drive to every P-EN cell.  Here each signed half-wave is integrated by a
    *spiking* vestibular cell.  Its mutually exclusive event output gates the
    normal RING -> P-EN coincidence circuit, so a vestibular event can shift
    only the currently active bump rather than exciting the whole ring.

    ``input_gain`` is merely the physical gyroscope's units-to-current
    calibration.  There is no heading, turn choice, world target, or motor
    command in this transducer; its only input is raw signed physical yaw.
    """
    for nid in (VEST_CCW, VEST_CW):
        ne.append(k.neuron(nid, r=float(r), c=2, lam=float(tau)))
        sy.append(k.syn(nid, 0, float(input_gain), 1)); ex.append(k.ext(nid, 0))
        sy.append(k.term(nid, 20000+nid))
    # Mutual inhibition suppresses alternating common-mode rowing strokes:
    # whichever signed rate has integrated farther vetoes the opposite event.
    for source, target in ((VEST_CCW, VEST_CW), (VEST_CW, VEST_CCW)):
        sy.append(k.syn(target, 1, -abs(float(cross_inhib)), 1))
        conns.append(k.conn(source, target, 1, 20000+source))
    for i in range(cc.NR):
        for p in range(cc.NP):
            for target, source in ((cc.CL[i][p], VEST_CCW), (cc.CR[i][p], VEST_CW)):
                j = _nsyn(sy, target)
                sy.append(k.syn(target, j, abs(float(gain)), 1))
                conns.append(k.conn(source, target, j, 20000+source))
    return ne, sy, conns, ex


def vestibular_comparator_parts(ne, sy, conns, ex, tau=20.0,
                                comparator_r=0.45, comparator_gain=1.0,
                                pen_gain=1.4, cross_inhib=1.5):
    """Low-pass signed yaw before generating mutually exclusive P-EN events.

    A rowing stroke alternates positive and negative raw yaw at much higher
    amplitude than its net rotation.  Rectifying first therefore causes both
    directional P-EN banks to fire.  This circuit performs the biologically
    necessary order entirely in PAULA: two graded rate afferents integrate
    the signed half-waves, then two spiking comparators receive their
    *difference*.  Only the winning comparator gates a local P-EN shift.
    """
    for nid in (VEST_POS, VEST_NEG):
        ne.append(k.neuron(nid, r=1e9, c=2, lam=float(tau),
                           meta={"graded_gain": 1.0, "graded_S0": 0.0}))
        sy.append(k.syn(nid, 0, 1.0, 1)); ex.append(k.ext(nid, 0))
        sy.append(k.term(nid, 20000+nid))
    for nid, positive, negative in (
        (VEST_CCW, VEST_POS, VEST_NEG),
        (VEST_CW, VEST_NEG, VEST_POS),
    ):
        ne.append(k.neuron(nid, r=float(comparator_r), c=2, lam=3))
        sy.append(k.syn(nid, 0, abs(float(comparator_gain)), 1))
        conns.append(k.conn(positive, nid, 0, 20000+positive))
        sy.append(k.syn(nid, 1, -abs(float(comparator_gain)), 1))
        conns.append(k.conn(negative, nid, 1, 20000+negative))
        sy.append(k.syn(nid, 2, -abs(float(cross_inhib)), 1))
        # Its reciprocal source is added after the two cells are declared;
        # terminal ids do not depend on declaration order.
        source = VEST_CW if nid == VEST_CCW else VEST_CCW
        conns.append(k.conn(source, nid, 2, 20000+source))
        sy.append(k.term(nid, 20000+nid))
    for i in range(cc.NR):
        for p in range(cc.NP):
            for target, source in ((cc.CL[i][p], VEST_CCW), (cc.CR[i][p], VEST_CW)):
                j = _nsyn(sy, target)
                sy.append(k.syn(target, j, abs(float(pen_gain)), 1))
                conns.append(k.conn(source, target, j, 20000+source))
    return ne, sy, conns, ex


def vestibular_notch_parts(ne, sy, conns, ex, window=40, input_gain=1000.0,
                           comparator_r=0.12, comparator_gain=1.0,
                           pen_gain=1.4, cross_inhib=1.5):
    """PAULA full-stroke signed-gyro notch followed by opponent P-EN gates.

    Each graded afferent receives the same signed gyro current twice: once
    immediately and once through an inhibitory dendritic delay.  With a
    near-lossless membrane this is a finite moving sum over ``window`` ticks,
    not a host-language filter.  The paired afferents use opposite signs, so
    their positive releases encode the two signs of *net* yaw only after the
    high-amplitude periodic stroke has cancelled.
    """
    if int(window) < 2:
        raise ValueError("vestibular notch window must be at least two ticks")
    for nid in (VEST_NET_CCW, VEST_NET_CW):
        ne.append(k.neuron(nid, r=1e9, c=2, lam=100000.0,
                           meta={"graded_gain": 1.0, "graded_S0": 0.0}))
        sy.append(k.syn(nid, 0, abs(float(input_gain)), 1)); ex.append(k.ext(nid, 0))
        sy.append(k.syn(nid, 1, -abs(float(input_gain)), int(window))); ex.append(k.ext(nid, 1))
        sy.append(k.term(nid, 20000+nid))
    for nid, positive, negative in (
        (VEST_CCW, VEST_NET_CCW, VEST_NET_CW),
        (VEST_CW, VEST_NET_CW, VEST_NET_CCW),
    ):
        ne.append(k.neuron(nid, r=float(comparator_r), c=2, lam=3))
        sy.append(k.syn(nid, 0, abs(float(comparator_gain)), 1))
        conns.append(k.conn(positive, nid, 0, 20000+positive))
        sy.append(k.syn(nid, 1, -abs(float(comparator_gain)), 1))
        conns.append(k.conn(negative, nid, 1, 20000+negative))
        sy.append(k.syn(nid, 2, -abs(float(cross_inhib)), 1))
        source = VEST_CW if nid == VEST_CCW else VEST_CCW
        conns.append(k.conn(source, nid, 2, 20000+source))
        sy.append(k.term(nid, 20000+nid))
    for i in range(cc.NR):
        for p in range(cc.NP):
            for target, source in ((cc.CL[i][p], VEST_CCW), (cc.CR[i][p], VEST_CW)):
                j = _nsyn(sy, target)
                sy.append(k.syn(target, j, abs(float(pen_gain)), 1))
                conns.append(k.conn(source, target, j, 20000+source))
    return ne, sy, conns, ex


def vestibular_notch_graded_parts(ne, sy, conns, ex, window=44, input_gain=1000.0,
                                  pen_gain=3.0):
    """Continuous PAULA notch release to the existing P-EN AND gates.

    This is an alternative to :func:`vestibular_notch_parts`, not a change to
    its accepted physical calibration. The paired graded, delayed gyro
    afferents remain the entire full-stroke filter; their continuous release,
    rather than thresholded comparator spikes, drives the local P-EN gates.
    """
    if int(window) < 2:
        raise ValueError("vestibular notch window must be at least two ticks")
    for nid in (VEST_NET_CCW, VEST_NET_CW):
        ne.append(k.neuron(nid, r=1e9, c=2, lam=100000.0,
                           meta={"graded_gain": 1.0, "graded_S0": 0.0}))
        sy.append(k.syn(nid, 0, abs(float(input_gain)), 1)); ex.append(k.ext(nid, 0))
        sy.append(k.syn(nid, 1, -abs(float(input_gain)), int(window))); ex.append(k.ext(nid, 1))
        sy.append(k.term(nid, 20000+nid))
    for i in range(cc.NR):
        for p in range(cc.NP):
            for target, source in ((cc.CL[i][p], VEST_NET_CCW), (cc.CR[i][p], VEST_NET_CW)):
                j = _nsyn(sy, target)
                sy.append(k.syn(target, j, abs(float(pen_gain)), 1))
                conns.append(k.conn(source, target, j, 20000+source))
    return ne, sy, conns, ex


def vestibular_notch_graded_opponent_parts(ne, sy, conns, ex, window=44,
                                           input_gain=1000.0, opponent_gain=40.0,
                                           opponent_lam=4.0, pen_gain=3.0,
                                           normalized=False, normalized_gain=1.0,
                                           lead_gain=0.0):
    """Continuous, signed PAULA full-stroke notch for prolonged turns.

    A full-gait window contains substantial positive yaw *and* negative yaw,
    so the two rectified notch afferents both have a positive release even
    when their difference encodes a small net turn.  The original graded
    route projected both releases positively to P-EN and therefore lost that
    signed residual over long turns.  This variant first computes
    ``CCW_notch - CW_notch`` and its reciprocal in two non-spiking PAULA
    opponent cells, then drives the local P-EN gates with the rectified graded
    differences.  No host-side filter, sign choice, heading, or turn enters.
    """
    if int(window) < 2:
        raise ValueError("vestibular notch window must be at least two ticks")
    for nid in (VEST_NET_CCW, VEST_NET_CW):
        ne.append(k.neuron(nid, r=1e9, c=2, lam=100000.0,
                           meta={"graded_gain": 1.0, "graded_S0": 0.0}))
        sy.append(k.syn(nid, 0, abs(float(input_gain)), 1)); ex.append(k.ext(nid, 0))
        sy.append(k.syn(nid, 1, -abs(float(input_gain)), int(window))); ex.append(k.ext(nid, 1))
        sy.append(k.term(nid, 20000+nid))
    for nid, positive, negative in (
        (VEST_OPP_CCW, VEST_NET_CCW, VEST_NET_CW),
        (VEST_OPP_CW, VEST_NET_CW, VEST_NET_CCW),
    ):
        metadata={"graded_gain": 1.0, "graded_S0": 0.0, "opponent_lead_gain": float(lead_gain)}
        if normalized:
            metadata.update({
                "opponent_normalized_gain": float(normalized_gain),
                "opponent_positive_synapse": 0,
                "opponent_negative_synapse": 1,
            })
        ne.append(k.neuron(nid, r=1e9, c=2, lam=float(opponent_lam), meta=metadata))
        sy.append(k.syn(nid, 0, abs(float(opponent_gain)), 1))
        conns.append(k.conn(positive, nid, 0, 20000 + positive))
        sy.append(k.syn(nid, 1, -abs(float(opponent_gain)), 1))
        conns.append(k.conn(negative, nid, 1, 20000 + negative))
        sy.append(k.term(nid, 20000 + nid))
    for i in range(cc.NR):
        for p in range(cc.NP):
            for target, source in ((cc.CL[i][p], VEST_OPP_CCW), (cc.CR[i][p], VEST_OPP_CW)):
                j = _nsyn(sy, target)
                # The complete PAULA gyro route appends its actual velocity
                # current after the ring (0) and direct-drive (1) ports. An
                # opt-in conjunctive P-EN must gate on this input, not on the
                # intentionally zero direct port.
                _set_conjunctive_velocity_port(ne, target, j)
                sy.append(k.syn(target, j, abs(float(pen_gain)), 1))
                conns.append(k.conn(source, target, j, 20000 + source))
    return ne, sy, conns, ex

# w_hs_shift 1.3 -> 0.0: MEASURED PER-TICK in the body. Visual shift drive DEGRADES compass tracking
# (per-tick r(omega,ring) 0.609 -> 0.204 as w_hs_shift goes 0 -> 0.8 at k_ang 0.9; at k_ang 0.45 it also
# collapses ring liveness to 20% of ticks and INVERTS the slope to -0.355). Vestibular drive alone at
# k_ang=0.65 is what reaches median heading error 13-16 deg. Visual anchoring needs a LEARNED
# cue->heading map, not a raw HS->P-EN shift; this stays 0 until that exists.
def parts(w_vr=4.0, r_vr=0.6, w_anchor=2.0, w_veto=-2.0, veto_far=4, w_hs_shift=0.0, w_eff=0.0, w_rly_eff=0.0,
          w_pe_vr=4.0, w_pe_ring=-4.0, r_pe=0.45,
          u_self=5.0, u_prev=2.4, u_fill=1.0, r_u=1.6, u_adv=25, u_drain=2.0, u_drop=0.8,
          w_unc_mode=0.5, w_musf_mode=0.5, **kw):
    # An explicit component tuple is a strict topology profile.  ``None`` is
    # deliberately retained as the compatibility/full-brain path used by the
    # older experiments.  Versioned agents never rely on a decoder mask: the
    # populations below are simply not emitted into the PAULA network.
    enabled = kw.pop("enabled_components", None)
    strict = enabled is not None
    selected = set(enabled or ())
    has_compass = (not strict) or "navigation.heading_ring" in selected
    has_path_integration = (not strict) or "navigation.path_integration" in selected
    has_visual = (not strict) or "vision.visual_cortex" in selected
    has_uncertainty = (not strict) or has_compass
    has_memory = (not strict) or "learning.mushroom_body" in selected
    has_arbiter = (not strict) or "arbitration.foraging_exploration" in selected
    has_metabolic = (not strict) or "body.metabolic_organs" in selected
    # The historical full-brain path stays byte-compatible by default.  V4
    # opts in structurally through its component tuple; legacy experiments may
    # request the pair explicitly with ``obstacle_components=True``.
    has_obstacle = ("sensory.obstacle_proximity" in selected) or bool(kw.get("obstacle_components", False))
    has_obstacle_reflex = ("motor.obstacle_reflex" in selected) or bool(kw.get("obstacle_components", False))

    # The profile is passed to the navigation fragment explicitly.  This is
    # intentionally a small, data-only interface: all control remains PAULA
    # synapses once the fragment has been assembled.
    profile = {
        "strict": strict,
        "compass": has_compass,
        "path_integration": has_path_integration,
        "visual": has_visual,
        "uncertainty": has_uncertainty,
        "memory": has_memory,
        "arbiter": has_arbiter,
        "metabolic": has_metabolic,
        "obstacle": has_obstacle,
        "obstacle_reflex": has_obstacle_reflex,
    }
    # forward compass keywords (Delta-7, shift bank, gains) so the ring can be configured from the agent
    _cck=dict(d7=True, r_d7=0.3, w_ring_d7=0.9,      # r_d7=1.4 left Delta-7 SILENT (0 spikes/300 ticks)
              w_d7_ring=-2.0, d7_out=8.0, w_gi_ring=-1.2,
              r_conj_lo=1.35, r_conj_hi=1.75,        # stagger matched to the drive's real range
              w_tonic=0.12)
    _cck.update({k2:v for k2,v in kw.items() if k2 in (
        "d7","w_ring_d7","r_d7","lam_d7","w_d7_ring","d7_in","d7_out","d7_cos","d7_uniform","d7_to_shift","w_d7_shift","w_d7_d7",
        "w_gi_ring","w_ring_gi","r_gi","w_push","w_push_ccw","w_push_cw","w_pull","d_push","d_push_hi","sh_bank","sh_max","shift_relay","shift_relay_gain","shift_relay_S0","shift_relay_lam",
        "r_conj_lo","r_conj_hi","w_conj","lam_conj","w_lr_inhib","bank_wta","jitter","r_pg","w_pg","w_pg_s","w_tonic",
        "graded_shift","graded_shift_gain","graded_shift_S0","graded_shift_max","conjunctive_graded_shift","conjunctive_graded_S0","conjunctive_ring_tau","conjunctive_velocity_tau","conjunctive_velocity_lead_gain",
        "w_peg","r_peg","lam_peg","d_peg","w_peg_ring","p_enb_trailing","lam_penb","w_peg_penb","penb_conjunctive_gain","penb_ring_tau","penb_velocity_tau","w_penb_ring","d_penb","p_ena_bank","pena_copies","pena_gain","pena_ring_tau","pena_velocity_taus","w_pena_ring","d_pena","pb_eb_bridge","r_pb","lam_pb","w_pb_peg","d_pb","pb_shift_gain","pb_shift_S0","pb_shift_lam","d_pb_shift","w_pb_cross_inhib","w_pb_opponent","pb_phase_update","r_pb_phase","lam_pb_phase","w_pb_phase_ring","w_pb_phase_sensor","w_pb_phase_clock","pb_phase_width",
        "w_self","w_nbr","lam_ring","r_ring","graded_ring","graded_ring_gain","graded_ring_S0","graded_ring_max",
        "w_r_shift", "k_pi", "lam_acc")})
    if has_compass:
        ne,sy,conns,ex = cc.parts(**_cck)
    else:
        ne,sy,conns,ex = [], [], [], []
    vest_modes=sum(bool(kw.get(name)) for name in (
        "vestibular_opponent", "vestibular_opponent_bank", "vestibular_phase_locked", "vestibular_stroke_reset", "vestibular_pulse", "vestibular_comparator", "vestibular_notch",
        "vestibular_notch_graded", "vestibular_notch_graded_opponent", "sensorimotor_estimator", "sensorimotor_estimator_v2",
    ))
    if vest_modes > 1:
        raise ValueError("choose at most one vestibular representation")
    if kw.get("relay_efference_only") and vest_modes:
        raise ValueError("relay_efference_only cannot be combined with a raw vestibular representation")
    if kw.get("pb_phase_update") and not kw.get("vestibular_stroke_reset"):
        raise ValueError("pb_phase_update requires vestibular_stroke_reset")
    if has_compass and kw.get("sensorimotor_estimator"):
        sensorimotor_estimator_parts(
            ne, sy, conns, ex,
            sensory_windows=kw.get("sensorimotor_sensory_windows", (36, 40, 44, 48)),
            prediction_taus=kw.get("sensorimotor_prediction_taus", (6.0, 12.0, 24.0, 48.0)),
            sensory_input_gain=float(kw.get("sensorimotor_sensory_input_gain", 1000.0)),
            sensory_weight=float(kw.get("sensorimotor_sensory_weight", 1.0)),
            prediction_input_gain=float(kw.get("sensorimotor_prediction_input_gain", 6.0)),
            prediction_weight=float(kw.get("sensorimotor_prediction_weight", 1.0)),
            error_weight=float(kw.get("sensorimotor_error_weight", 1.0)),
            update_gain=float(kw.get("sensorimotor_update_gain", 1.0)),
            update_lam=float(kw.get("sensorimotor_update_lam", 4.0)),
            pen_gain=float(kw.get("sensorimotor_pen_gain", 1.0)),
        )
    elif has_compass and kw.get("sensorimotor_estimator_v2"):
        sensorimotor_estimator_v2_parts(
            ne, sy, conns, ex,
            sensory_taus=kw.get("sensorimotor_v2_sensory_taus", (6.0, 12.0, 24.0, 48.0)),
            prediction_taus=kw.get("sensorimotor_v2_prediction_taus", (6.0, 12.0, 24.0, 48.0)),
            sensory_input_gain=float(kw.get("sensorimotor_v2_sensory_input_gain", 0.12)),
            sensory_weight=float(kw.get("sensorimotor_v2_sensory_weight", 1.0)),
            prediction_input_gain=float(kw.get("sensorimotor_v2_prediction_input_gain", 6.0)),
            prediction_weight=float(kw.get("sensorimotor_v2_prediction_weight", 1.0)),
            error_weight=float(kw.get("sensorimotor_v2_error_weight", 1.0)),
            update_gain=float(kw.get("sensorimotor_v2_update_gain", 1.0)),
            update_lam=float(kw.get("sensorimotor_v2_update_lam", 4.0)),
            pen_gain=float(kw.get("sensorimotor_v2_pen_gain", 1.0)),
            phase_gated_lead=bool(kw.get("sensorimotor_v2_phase_gated_lead", False)),
            phase_copies=int(kw.get("sensorimotor_v2_phase_copies", SM3_PHASE_COPIES)),
            phase_gate_gain=float(kw.get("sensorimotor_v2_phase_gate_gain", 1.0)),
            phase_gate_tau=float(kw.get("sensorimotor_v2_phase_gate_tau", 1.0)),
            p_enb_trailing=bool(kw.get("p_enb_trailing", False)),
            p_enb_update_gain=float(kw.get("p_enb_update_gain", 1.0)),
            p_ena_bank=bool(kw.get("p_ena_bank", False)),
            p_ena_update_gain=float(kw.get("p_ena_update_gain", 1.0)),
        )
    elif has_compass and kw.get("vestibular_opponent"):
        vestibular_opponent_parts(ne, sy, conns, ex,
                                  tau=float(kw.get("vop_tau", 60.0)),
                                  gain=float(kw.get("vop_gain", 1.0)))
    elif has_compass and kw.get("vestibular_opponent_bank"):
        vestibular_opponent_bank_parts(
            ne, sy, conns, ex,
            taus=kw.get("vop_bank_taus", (8.0, 12.0, 18.0, 28.0, 44.0, 68.0, 104.0, 160.0)),
            gain=float(kw.get("vop_bank_gain", 1.0)),
        )
    elif has_compass and kw.get("vestibular_phase_locked"):
        vestibular_phase_locked_parts(
            ne, sy, conns, ex,
            period=int(kw.get("vop_phase_period", 44)),
            input_gain=float(kw.get("vop_phase_input_gain", 1.0)),
            output_gain=float(kw.get("vop_phase_output_gain", 0.02)),
            pen_gain=float(kw.get("vop_phase_pen_gain", 1.0)),
            clock_threshold=float(kw.get("vop_phase_clock_threshold", 0.0)),
            hold_ticks=int(kw.get("vop_phase_hold_ticks", 0)),
        )
    elif has_compass and kw.get("vestibular_stroke_reset"):
        if kw.get("pb_stroke_reset") and not kw.get("pb_eb_bridge"):
            raise ValueError("pb_stroke_reset requires pb_eb_bridge")
        if kw.get("pb_phase_update") and not kw.get("pb_eb_bridge"):
            raise ValueError("pb_phase_update requires pb_eb_bridge")
        vestibular_stroke_reset_parts(
            ne, sy, conns, ex,
            period=int(kw.get("vop_reset_period", 44)),
            input_gain=float(kw.get("vop_reset_input_gain", 1.0)),
            reset_gain=float(kw.get("vop_reset_gain", 40.0)),
            integrator_lam=float(kw.get("vop_reset_lam", 44.0)),
            output_gain=float(kw.get("vop_reset_output_gain", 1.0)),
            output_S0=float(kw.get("vop_reset_output_S0", 0.0)),
            pen_gain=float(kw.get("vop_reset_pen_gain", 1.0)),
            cross_inhib=float(kw.get("vop_reset_cross_inhib", 0.0)),
            pb_stroke_reset=bool(kw.get("pb_stroke_reset", False)),
            pb_reset_gain=float(kw.get("pb_reset_gain", 0.0)),
            pb_phase_update=bool(kw.get("pb_phase_update", False)),
            pb_phase_start=int(kw.get("pb_phase_start", 11)),
            pb_phase_sensor_gain=float(kw.get("pb_phase_sensor_gain", 1.0)),
            pb_phase_sensor_max=float(kw.get("pb_phase_sensor_max", 4.0)),
        )
    elif has_compass and kw.get("vestibular_pulse"):
        vestibular_pulse_parts(
            ne, sy, conns, ex,
            tau=float(kw.get("vop_pulse_tau", 30.0)),
            r=float(kw.get("vop_pulse_r", 2.0)),
            input_gain=float(kw.get("vop_pulse_input_gain", 0.04)),
            gain=float(kw.get("vop_pulse_gain", 1.4)),
            cross_inhib=float(kw.get("vop_pulse_cross_inhib", 1.5)),
        )
    elif has_compass and kw.get("vestibular_comparator"):
        vestibular_comparator_parts(
            ne, sy, conns, ex,
            tau=float(kw.get("vop_comp_tau", 20.0)),
            comparator_r=float(kw.get("vop_comp_r", 0.45)),
            comparator_gain=float(kw.get("vop_comp_gain", 1.0)),
            pen_gain=float(kw.get("vop_comp_pen_gain", 1.4)),
            cross_inhib=float(kw.get("vop_comp_cross_inhib", 1.5)),
        )
    elif has_compass and kw.get("vestibular_notch"):
        vestibular_notch_parts(
            ne, sy, conns, ex,
            window=int(kw.get("vop_notch_window", 40)),
            input_gain=float(kw.get("vop_notch_input_gain", 1000.0)),
            comparator_r=float(kw.get("vop_notch_r", 0.12)),
            comparator_gain=float(kw.get("vop_notch_gain", 1.0)),
            pen_gain=float(kw.get("vop_notch_pen_gain", 1.4)),
            cross_inhib=float(kw.get("vop_notch_cross_inhib", 1.5)),
        )
    elif has_compass and kw.get("vestibular_notch_graded"):
        vestibular_notch_graded_parts(
            ne, sy, conns, ex,
            window=int(kw.get("vop_notch_graded_window", 44)),
            input_gain=float(kw.get("vop_notch_graded_input_gain", 1000.0)),
            pen_gain=float(kw.get("vop_notch_graded_pen_gain", 3.0)),
        )
    elif has_compass and kw.get("vestibular_notch_graded_opponent"):
        vestibular_notch_graded_opponent_parts(
            ne, sy, conns, ex,
            window=int(kw.get("vop_notch_graded_opp_window", 44)),
            input_gain=float(kw.get("vop_notch_graded_opp_input_gain", 1000.0)),
            opponent_gain=float(kw.get("vop_notch_graded_opp_gain", 40.0)),
            opponent_lam=float(kw.get("vop_notch_graded_opp_lam", 4.0)),
            pen_gain=float(kw.get("vop_notch_graded_opp_pen_gain", 3.0)),
            normalized=bool(kw.get("vop_notch_graded_opp_normalized", False)),
            normalized_gain=float(kw.get("vop_notch_graded_opp_normalized_gain", 1.0)),
            lead_gain=float(kw.get("vop_notch_graded_opp_lead_gain", 0.0)),
        )
    _vck={k2:v for k2,v in kw.items() if k2 in ("hs_mod","w_hs","r_hs_lo","r_hs_hi","lam_hs","w_hs_anti","hs_prof",
          "w_hs_opp","w_emd","r_emd","d_emd","lam_emd","w_emd_opp")}
    if has_visual:
        vc.parts(ne=ne,sy=sy,conns=conns,ex=ex,**_vck)
    # nv.parts took NO kwargs, so CPU4/CPU1 knobs (w_drain, d_adv, w_prev, cd_cut...) were unreachable
    # from the agent -- the third allowlist gap of this kind. Forward them explicitly.
    _cpk={k2:v for k2,v in kw.items() if k2 in ("w_drain","w_prev","d_adv","w_drive","r_lad","w_self",
          "w_self_drop","cd_cut","cd_pow","cd_src","cd_neg","w_cd","r_cd","w_opp","r_opp","seed_amp",
          "w_lgi","r_lgi","lam_lgi","w_lad_lgi","w_cpu1")}
    # ``pistone_opp`` replaces, rather than supplements, the legacy CPU4
    # ladder as the source of OPP.  Leaving the ladder connections live made
    # a HOME spike ambiguous: it could have come from a birth-seeded CPU4
    # latch instead of the graded Stone memory.  An explicit ``w_opp`` keeps
    # its normal exploratory meaning; otherwise the legacy source is silent.
    if kw.get("pistone_opp") and "w_opp" not in _cpk:
        _cpk["w_opp"]=0.0
    if kw.get("pistone_opp") and "w_cpu1" not in _cpk:
        _cpk["w_cpu1"]=0.0
    if has_path_integration:
        nv.parts(ne=ne,sy=sy,conns=conns,ex=ex,wire_muscles=False,**_cpk)
    # REDESIGNED ACCUMULATOR, opt-in via accum=True. Attaches to the SAME CD population as the ladder,
    # so both can be read in the same run and compared in the body. The ladder is structurally BINARY
    # (w_self<=3.2 -> dead, 5.0 -> permanent latch), which is why it cannot sum two legs; this one is a
    # leaky integrator on a circular array with local excitation + global inhibition and NO drain.
    # STONE-mechanism PI memory (opt-in): uniform excitatory speed + INHIBITION from the heading bump
    # + constant leak, on graded analog cells. Replaces the CPU4 latch-ladder, which can only
    # accumulate and therefore cannot represent net displacement.
    if has_path_integration and kw.get("pistone"):
        _pk={k2:v for k2,v in kw.items() if k2 in ("w_speed","w_inhib","lam_mem","ring_width",
             "graded","graded_gain","graded_S0")}
        # The speed port is a current into a slow (lam=8000+) analog
        # integrator.  Its historical unit weight produced a valid membrane
        # profile, but a release far below the downstream OPP threshold.  The
        # opt-in defaults place the graded signal in the usable range; callers
        # can still sweep either value explicitly.
        _pk.setdefault("w_speed",8.0)
        _pk.setdefault("w_inhib",1.0)
        pstone.parts(ne,sy,conns,ex,**_pk)
        # Drive the EXISTING opponent cells from this memory instead of leaving them on the broken
        # ladder. Opt-in so the ladder path stays measurable side by side in the same run.
        if kw.get("pistone_opp"):
            # A graded release is numerically much smaller than one PAULA
            # spike.  128 is the lowest calibrated gain that makes the
            # memory-driven OPP population beat the concurrent uncertainty
            # population in the composed agent (the former 8 was inert).
            pstone.wire_opp(ne,sy,conns,ex,OPP=nv.OPP,w_opp=float(kw.get("w_opp_stone",128.0)))
            # The same memory must also be the input to CPU1; otherwise HOME
            # can win while the legacy CPU4 comparator has no vector to turn
            # on.  CPU1's ring inputs remain intact and provide the heading
            # gate, so this stays a neural vector-to-turn computation.
            pstone.wire_cpu1(ne,sy,conns,ex,HL=nv.HL,HR=nv.HR,
                              w_cpu1=float(kw.get("w_cpu1_stone",128.0)))
            if "w_musf_mode" not in kw:
                w_musf_mode=1.0
    if has_path_integration and kw.get("accum"):
        _ak={k2:v for k2,v in kw.items() if k2 in ("w_cd_acc","r_acc","lam_acc","w_acc_self",
             "w_acc_nbr","w_acc_agi","r_agi","w_acc_to_agi","lam_agi","r_acc_span")}
        pacc.parts(ne,sy,conns,ex,CD=nv.CD,CDM=nv.CDM,**_ak)  # CPU4 + CPU1 (muscles mode-gated below)
    # Without path integration, HOME has no causal source.  The strict V3
    # profile therefore composes a three-way FORAGE/EXPLORE/SLEEP WTA rather
    # than retaining an un-driven HOME population.
    arbiter_home = has_path_integration
    arbiter_groups = (ar.MODE if arbiter_home else (ar.MODE[0], ar.MODE[2])) if has_arbiter else ()
    if has_arbiter:
        ane,asy,acon,aex = ar.build(
            sleep=bool(kw.get("metabolic_sleep", False)), home=arbiter_home
        )  # arbiter (modes + hunger; V3 adds sleep)
        ne+=ane; sy+=asy; conns+=acon; ex+=aex
    if has_arbiter and has_metabolic and kw.get("metabolic_sleep"):
        met.parts(
            ne, sy, conns, ex,
            w_sleep_gut=float(kw.get("w_sleep_gut", 1.0)),
            w_sleep_energy=float(kw.get("w_sleep_energy", 0.65)),
            w_sleep_low=float(kw.get("w_sleep_low", -1.0)),
            w_forage_low=float(kw.get("w_forage_low", 1.0)),
            w_explore_energy=float(kw.get("w_explore_energy", 1.0)),
            w_sleep_hazard=float(kw.get("w_sleep_hazard", -4.0)),
        )
    if kw.get("relay_efference_population"):
        relay_efference_population_parts(
            ne, sy, conns,
            taus=kw.get("relay_efference_taus", (6.0, 12.0, 24.0, 48.0)),
            input_gain=float(kw.get("relay_efference_input_gain", 6.0)),
            pen_gain=float(kw.get("relay_efference_pen_gain", 1.0)),
        )
    # ---- SPIKING RELAY EFFERENCE COPY ----
    # TL/TR encode an INTENTION to turn. The RELAYS are the last spiking stage before the muscles --
    # RLY[MLp]+RLY[MLr] gate the agent's RIGHT paddle, RLY[MRp]+RLY[MRr] the LEFT -- so their asymmetry
    # is a spiking proxy for the motor output that actually produces yaw. The graded muscle membranes
    # would be the ideal signal but they never spike, and propagation is spike-gated, so the relays are
    # the closest observable the substrate allows.
    if has_compass and w_rly_eff:
        for i in range(cc.NR):
            for p in range(cc.NP):
                # ML* drive turns the body CCW (measured +132 deg), so ML relays -> CL, MR relays -> CR
                for cid,srcs in ((cc.CL[i][p],(MLp,MLr)),(cc.CR[i][p],(MRp,MRr))):
                    s2=_nsyn(sy,cid)
                    for m in srcs:
                        for g in RLY[m]:
                            sy.append(k.syn(cid,s2,w_rly_eff,1))
                            conns.append(k.conn(g,cid,s2,20000+g)); s2+=1
    # ---- EFFERENCE COPY (built, MEASURED NOT TO WORK HERE, default off) ----
    # von Holst & Mittelstaedt 1950; Green et al. 2017; Kim, Fisher & Maimon 2015. This is how the fly
    # does it and how the verified prototype did it (omega_cmd went to the body AND the compass).
    # It does not transfer to this body: TL/TR are opponent neurons driven by the ODOUR GRADIENT, so
    # they encode an INTENTION to turn, not a turn RATE, while the realised yaw is dominated by stroke
    # mechanics. Measured at 300 steps: TL->CL slope -0.40, flipped TL->CR slope +0.06, stronger -0.67,
    # against +0.36 with it off. The graded muscle asymmetry WOULD be the proportional signal, but
    # graded cells never spike so they cannot drive anything (spike-gated propagation). The spiking
    # relay populations are the remaining candidate proxy.
    # The reafference principle: an animal does not recover its rotation from raw sensors, it takes a
    # copy of its own motor command and predicts the consequence. In the fly the compass bump is
    # updated by an angular-velocity signal derived largely from efference copy of the turn command
    # arriving via the lateral accessory lobe (Green et al. 2017 Nature; Turner-Evans et al. 2017;
    # Kim, Fisher & Maimon 2015 for the HS-cell version). The verified cx_navigator prototype did
    # exactly this -- it handed omega_cmd to the body AND to the compass -- which is why it reached a
    # 0.97 tracking ratio while the embodied agent, trying to estimate rotation from a stroke-corrupted
    # yaw signal, could not.
    # TL/TR are the opponent turn commands. Measured sign: TL gates the MR* pair (the agent's LEFT
    # paddle) so TL -> turn left -> CCW -> CL; TR -> CW -> CR. This is a neuron->neuron connection,
    # so it is strictly MORE PAULA-pure than the vestibular transducer it supplements.
    if has_compass and w_eff:
        for i in range(cc.NR):
            for p in range(cc.NP):
                # MEASURED, not derived: TL->CL gave slope -0.40 (bump running backwards against the
                # body), so the correct pairing is the opposite one. The double inversion in the motor
                # naming (padL sits on the agent's right) makes the paper derivation unreliable here.
                for cid,src in ((cc.CL[i][p],TR),(cc.CR[i][p],TL)):
                    s2=_nsyn(sy,cid)
                    sy.append(k.syn(cid,s2,w_eff,1))
                    conns.append(k.conn(src,cid,s2,20000+src))
    # ---- VISUAL SELF-MOTION -> the compass shift (fly HS -> P-EN; Green, Vijayan & Maimon 2017) ----
    # The proprioceptive turn signal spans 0.4-35 rad/s while the shift circuit is linear only in a
    # narrow band, so no single gain can serve both. The HS pooling is COMPRESSIVE by construction
    # (measured: 0.4/1.2/3.0 rad/s -> +24/+59/+72), which is exactly the conditioning the shift needs.
    # This is an additional, INDEPENDENT estimate of self-rotation, not an anchor: it says "I am
    # turning this fast", never "I am facing this way".
    if has_compass and has_visual and w_hs_shift:
        for i in range(cc.NR):
            for p in range(cc.NP):
                for cid,src in ((cc.CL[i][p],vc.HS_CCW),(cc.CR[i][p],vc.HS_CW)):
                    s2=_nsyn(sy,cid)
                    for h in range(vc.NHS):
                        sy.append(k.syn(cid,s2,w_hs_shift,1))
                        conns.append(k.conn(src(h),cid,s2,vc.T(src(h)))); s2+=1
    # ---- VISUAL RING + GABAergic anchoring (a sighting inhibits every heading it contradicts) ----
    src_of={j:[] for j in range(cc.NR)} if has_compass and has_visual else {}
    if has_compass and has_visual:
        for a in range(vc.NV1AZ):
            implied=w3.SUN_AZIMUTH-vc.az_of(a,vc.NV1AZ)
            src_of[int(round((implied%(2*np.pi))/(2*np.pi)*cc.NR))%cc.NR].append(a)
        for j in range(cc.NR):
            if not src_of[j]: continue
            ne.append(k.neuron(VR[j],r=r_vr,c=2,lam=6)); idx=0
            for a in src_of[j]:
                sy.append(k.syn(VR[j],idx,w_vr,1)); conns.append(k.conn(vc.SAL(a),VR[j],idx,vc.T(vc.SAL(a)))); idx+=1
            sy.append(k.term(VR[j],20000+VR[j]))
        for j in range(cc.NR):
            if not src_of[j]: continue
            for i in range(cc.NR):
                d=abs(i-j); d=min(d,cc.NR-d)
                w=w_anchor if d<=1 else (w_veto if d>=veto_far else 0.0)
                if w==0.0: continue
                nid=cc.RING[i]; s2=_nsyn(sy,nid)
                sy.append(k.syn(nid,s2,w,1)); conns.append(k.conn(VR[j],nid,s2,20000+VR[j]))
    # ---- SELF-MODEL: prediction error = vision asserts heading j while the bump is elsewhere ----
    if has_compass and has_visual:
        for j in range(cc.NR):
            if not src_of[j]: continue
            ne.append(k.neuron(PE[j],r=r_pe,c=2,lam=8))
            sy.append(k.syn(PE[j],0,w_pe_vr,1));   conns.append(k.conn(VR[j],PE[j],0,20000+VR[j]))
            sy.append(k.syn(PE[j],1,w_pe_ring,1)); conns.append(k.conn(cc.RING[j],PE[j],1,cc.Tt(cc.RING[j])))
            sy.append(k.term(PE[j],20000+PE[j]))
    # ---- CONFIDENCE: uncertainty ladder. fills while MOVING (ext), drained by seeing a landmark (VR) ----
    if has_uncertainty:
        for i,nid in enumerate(UNC):
            ne.append(k.neuron(nid,r=r_u,c=2,lam=2)); j=0
            sy.append(k.syn(nid,j,u_self-(i/max(NU-1,1))*u_drop,1)); conns.append(k.conn(nid,nid,j,20000+nid)); j+=1
            if i>0:
                sy.append(k.syn(nid,j,u_prev,u_adv)); conns.append(k.conn(UNC[i-1],nid,j,20000+UNC[i-1])); j+=1
            sy.append(k.syn(nid,j,u_fill,1)); ex.append(k.ext(nid,j)); j+=1                 # fill port (movement)
            for jj in range(cc.NR):                                                          # drain: any landmark fix
                if src_of.get(jj):
                    sy.append(k.syn(nid,j,-u_drain,1)); conns.append(k.conn(VR[jj],nid,j,20000+VR[jj])); j+=1
            if i==0:
                global UNC_SEED_SYN; UNC_SEED_SYN=j
                sy.append(k.syn(nid,j,6.0,1)); ex.append(k.ext(nid,j))                       # birth seed
            sy.append(k.term(nid,20000+nid))
    # ---- BELIEFS -> ARBITER DRIVE PORTS (the modes are now driven by the brain, not by python) ----
    if has_arbiter:
        dsyn=ar.mode_drive_syn(0,0)
        if has_path_integration:
            for i,nid in enumerate(ar.MODE[1]):                     # HOME <- |home vector| (CPU4 opponent cells)
                s2=_nsyn(sy,nid)
                for c in range(nv.NC):
                    sy.append(k.syn(nid,s2,w_musf_mode,1)); conns.append(k.conn(nv.OPP[c],nid,s2,nv.T(nv.OPP[c]))); s2+=1
        for i,nid in enumerate(ar.MODE[2]):                     # EXPLORE <- uncertainty + surprise
            s2=_nsyn(sy,nid)
            if has_uncertainty:
                for u in UNC:
                    sy.append(k.syn(nid,s2,w_unc_mode,1)); conns.append(k.conn(u,nid,s2,20000+u)); s2+=1
            if has_compass and has_visual:
                for j in range(cc.NR):
                    if src_of.get(j):
                        sy.append(k.syn(nid,s2,w_unc_mode,1)); conns.append(k.conn(PE[j],nid,s2,20000+PE[j])); s2+=1
        for i,nid in enumerate(ar.MODE[0]):                     # FORAGE <- hunger
            s2=_nsyn(sy,nid)
            for h in ar.HUNGER:
                sy.append(k.syn(nid,s2,w_unc_mode,1)); conns.append(k.conn(h,nid,s2,ar.T(h))); s2+=1
    _alk={k2:v for k2,v in kw.items() if k2 in ("r_orn","w_orn","w_pn","w_al","r_al","w_al_pool","r_pn")}
    _stk={k2:v for k2,v in kw.items() if k2 in ("w_trig","w_latch","w_kill","r_stg","lam_stg","d_us","d_reflex","mod_cort","mod_dopa","w_teach_stg","w_rfx")}
    if has_memory:
        sting_parts(ne,sy,conns,ex,**_stk)
        al_parts(ne,sy,conns,ex,**_alk)
    _mbk={k2:v for k2,v in kw.items() if k2 in ("r_kc","w_pk","w_apl_kc","r_apl","w_kc_apl",
          "w_km0","eta","kappa","rh_decay","w_teach","w_tref_cort","r_mbon","w_av_mbon","r_avoid","lam_avoid","w_kc_max",
          # vAC (visual calyx). This dict is an ALLOWLIST: a name missing here is silently dropped,
          # which is how mb_parts kwargs were inert once before.
          "vac","w_chr_vpn","r_vpn","w_vpn_vkc","r_vkc","w_vapl_vkc","r_vapl","w_vkc_vapl","w_vkm0")}
    if has_memory:
        mb_parts(ne,sy,conns,ex,**_mbk)                             # PORTED mushroom body (learned valence)
    # navcore_parts took NO kwargs at all, so any new knob here would have been silently inert -- the
    # same trap as the mb_parts allowlist. Forward explicitly.
    _nvk={k2:v for k2,v in kw.items() if k2 in (
        "trise", "w_trise", "w_tox", "w_sd", "w_riseinh", "w_ton", "w_pool",
        # MBON/AVOID reaches the body only through this escape-to-relay gate.
        # Keep it configurable for causal calibration; previously a caller's
        # value was silently discarded by this allowlist.
        "w_avoid_steer", "w_forage_steer", "turn_probe_ports",
        "metabolic_sleep", "w_sleep_veto",
        # Opt-in learned-valence/lateral-horn route.  Preserve all structural
        # and calibration fields so an experiment cannot silently test the
        # baseline network instead.
        "mb_lh", "w_lh_avoid", "w_lh_sensor", "r_lh", "w_lh_turn",
    )}
    _nvk.update({
        "mode_groups": arbiter_groups,
        "home_group": ar.MODE[1] if arbiter_home else (),
        "sleep_group": ar.SLEEP_MODE if has_arbiter and has_metabolic and kw.get("metabolic_sleep") else (),
        "home_sources": (nv.HL, nv.HR) if has_path_integration else ((), ()),
        "avoid_source": AVOID if has_memory else False,
        "arbiter_enabled": has_arbiter,
        "path_integration_enabled": has_path_integration,
        "memory_enabled": has_memory,
    })
    navcore_parts(ne,sy,conns,ex,**_nvk)                        # FAITHFUL neural_agent navigation core
    if has_obstacle:
        obs.parts(ne, sy, conns, ex,
                  onset_delay=int(kw.get("obstacle_onset_delay", 6)),
                  proximity_gain=float(kw.get("obstacle_proximity_gain", 1.0)),
                  onset_gain=float(kw.get("obstacle_onset_gain", 1.5)))
    if has_obstacle_reflex:
        obm.parts(ne, sy, conns, ex,
                  turn_gain=float(kw.get("obstacle_turn_gain", 3.4)),
                  onset_turn_gain=float(kw.get("obstacle_onset_turn_gain", 2.2)),
                  brake_gain=float(kw.get("obstacle_brake_gain", -1.15)),
                  wall_gain=float(kw.get("obstacle_wall_gain", 0.8)),
                  relays=RLY)
    if not strict:
        belief_parts(ne,sy,conns,ex)                                # PORTED spiking AIF belief core (BL/BR/UBEL/DL/DR)
    return ne,sy,conns,ex

# =====================================================================================================
# NAVIGATION CORE — a FAITHFUL port of neural_agent.py (the verified 12/12 navigator). Earlier I replaced
# this with two opponent cells and a kinematic body, and re-derived every failure it was built to avoid.
# The real architecture, kept intact:
#   FL/FR log-threshold sensor POPULATIONS, divisively normalised by POOL  (no saturation, no deadzone)
#   TL/TR OPPONENT turn neurons with WTA cross-inhibition                  (fires on the L/R DIFFERENCE)
#   RISE population (staggered delay windows) -> SEARCH spiral             (klinokinesis)
#   ENGINE (coprime pacemakers) -> smooth tonic power
#   CPG ring -> RELAY populations (distributed thresholds) -> graded muscles
#      *** steering WEAKENS THE DRIVE through the relays, it does not inhibit the muscle membrane ***
#   POOL also BRAKES both sides near food, so the agent converges instead of overshooting.
# Higher brain (mushroom body, compass/PI homing, arbiter) steers through the SAME relay gate.
# =====================================================================================================
NPS=10; SENS_THR=np.exp(np.linspace(np.log(0.25),np.log(4.5),NPS))
FLp=[83300+i for i in range(NPS)]; FRp=[83320+i for i in range(NPS)]
TXL=[83340+i for i in range(NPS)]; TXR=[83360+i for i in range(NPS)]   # toxin sensor populations
POOL=83380; STEER=83381; TL=83382; TR=83383
# ``SEARCH`` is the background curved-search command.  It is structurally
# separate from STEER so FORAGE can gate exploration without vetoing the
# independent TRISE/AVOID hazard-escape pathway carried by STEER.
SEARCH=87650
HTL=83384; HTR=83385                                   # HOME-mode turn commands (from CPU1)
RISEP=[83390+i for i in range(6)]; RISE_DEL=[35,50,65,80,95,110]
# TOXIN rise detector -> run termination. Opponent steering (TXL/TXR -> TR/TL) is structurally blind to
# a HEAD-ON toxin: both sides are driven equally, TL and TR cancel through their mutual WTA, and the
# agent walks past a ~6000-spike danger signal (measured: min-distance change +0.03/+0.05/-0.02 on 3
# seeds vs a food positive control of -0.95..-1.19).
# Drosophila larval chemotaxis uses TWO mechanisms (Gomez-Marin & Louis; klinotaxis review PMC4132367):
# turning rate follows the LATERAL gradient component, while RUN TERMINATION is driven by the TEMPORAL
# change along the direction of travel. A raw intensity SUM is neither -- which is why the obvious
# "TXL+TXR -> STEER" is wrong. This is the temporal-derivative arm, built from the SAME primitive the
# food pathway already uses (RISEP: +w immediate, -w delayed), with the OPPOSITE sign into STEER:
# rising food INHIBITS the spiral (w_riseinh<0, keep going), so rising toxin must EXCITE it (turn away).
TPOOL=87500; TRISE=[87510+i for i in range(6)]      # ids >=87500: max pre-existing id is 86021, vAC owns 87000-87400
ENGN=[83400+i for i in range(5)]; ENG_C=[2,3,5,7,11]
CPG_PERIOD=10; LAM_M=6      # STROKE RATE. The cycle is PERIOD*4 ticks, and the relays only pass drive on a
                            # CPG spike, so a slow gait starves the muscles: at PERIOD=40 the agent crept
                            # (CPG 0.048/tick, closest approach 2.33, never fed). Measured: 20 -> 1.68,
                            # 12 -> 0.67 REACHED, 8 -> 0.28 REACHED.
CPGP=[85000+i for i in range(4)]
MLp,MLr,MRp,MRr = 85010,85011,85012,85013
NRLY=6; THR_R=np.linspace(0.5,4.2,NRLY)
RLY={MLp:[85100+i for i in range(NRLY)], MLr:[85110+i for i in range(NRLY)],
     MRp:[85120+i for i in range(NRLY)], MRr:[85130+i for i in range(NRLY)]}

def navcore_parts(ne,sy,conns,ex, w_cpg=8.0, w_norm=-0.9, w_pool=0.5, w_sd=2.5, w_wta=-7.0, w_ton=2.5,
                  w_riseinh=-4.0, w_closeinh=-0.35, w_rc=14.0, w_relay_inh=-4.0, w_steer_relay=-5.0,
                  trise=True, w_trise=3.0,    # toxin rise -> STEER (run termination), enabled for head-on hazard escape.
                  w_brake=-0.7, w_tox=2.2, w_avoid_steer=3.0, w_mode_veto=-2.0, w_home_turn=1.4,
                  w_forage_steer=0.0, turn_probe_ports=False,
                  mb_lh=True, w_lh_avoid=3.0, w_lh_sensor=1.2,
                  r_lh=0.8, w_lh_turn=10.0, metabolic_sleep=False,
                  w_sleep_veto=-8.0, mode_groups=None, sleep_group=(),
                  home_group=None,
                  home_sources=None, avoid_source=None,
                  arbiter_enabled=True, path_integration_enabled=True,
                  memory_enabled=True):
    # ``None`` retains the legacy full-brain wiring. Strict version profiles
    # pass empty source groups for omitted components, so no dangling
    # HOME/AVOID/arbiter projections are emitted.
    if mode_groups is None:
        mode_groups = ar.MODE
    mode_groups = tuple(tuple(group) for group in mode_groups)
    sleep_group = tuple(sleep_group)
    if home_group is None:
        home_group = mode_groups[1] if len(mode_groups) >= 3 else ()
    home_group = tuple(home_group)
    if home_sources is None:
        home_sources = (nv.HL, nv.HR)
    home_left, home_right = tuple(home_sources)
    if avoid_source is None:
        # Default argument is the historical full-brain route.  Strict V1
        # passes ``False`` explicitly to keep the learned-danger population
        # out of its topology.
        avoid_source = AVOID
    avoid_enabled = avoid_source is not False and memory_enabled
    # ENGINE: coprime pacemakers -> one crude tonic drive becomes a smooth, dense power train
    for e,c in zip(ENGN,ENG_C):
        ne.append(k.neuron(e,r=0.6,c=c,lam=5)); sy.extend([k.syn(e,0,2.0,1),k.term(e,20000+e)]); ex.append(k.ext(e,0))
    # CPG ring: propagation delay sets phase spacing; one-time birth impulse then self-sequences
    for nid in CPGP:
        ne.append(k.neuron(nid,r=0.6,lam=3,c=CPG_PERIOD)); sy.extend([k.syn(nid,0,4.0,CPG_PERIOD),k.term(nid,20000+nid)])
    for i,nid in enumerate(CPGP): conns.append(k.conn(CPGP[(i-1)%4],nid,0,20000+CPGP[(i-1)%4]))
    ex.append(k.ext(CPGP[0],0))
    # MUSCLES: graded, one synapse per relay (summed drive)
    for m in (MLp,MLr,MRp,MRr):
        ne.append(k.neuron(m,r=1e9,lam=LAM_M,delta_decay=1.0,c=2))
        for j in range(NRLY): sy.append(k.syn(m,j,w_cpg/NRLY,1))
        sy.append(k.term(m,20000+m))
    # SENSORS: log-threshold populations, each divisively normalised by POOL
    for pop in (FLp,FRp,TXL,TXR):
        for i,nid in enumerate(pop):
            ne.append(k.neuron(nid,r=float(SENS_THR[i]),c=2,lam=4))
            sy.extend([k.syn(nid,0,1.0,1),k.syn(nid,1,w_norm,1),k.term(nid,20000+nid)])
            ex.append(k.ext(nid,0)); conns.append(k.conn(POOL,nid,1,20000+POOL))
    ne.append(k.neuron(POOL,r=0.5,c=1,lam=3)); pj=0
    for nid in FLp+FRp:
        sy.append(k.syn(POOL,pj,w_pool,1)); conns.append(k.conn(nid,POOL,pj,20000+nid)); pj+=1
    sy.append(k.term(POOL,20000+POOL))
    # TURN neurons: OPPONENT tropotaxis (fires on the L/R DIFFERENCE) + WTA. Toxin enters CROSSED (avoid).
    for nid in (TL,TR): ne.append(k.neuron(nid,r=0.6,c=2,lam=4))
    tj={TL:0,TR:0}
    def addsyn(nid,w): j=tj[nid]; sy.append(k.syn(nid,j,w,1)); tj[nid]+=1; return j
    for s_ in FLp: conns.append(k.conn(s_,TL,addsyn(TL, w_sd),20000+s_))
    for s_ in FRp: conns.append(k.conn(s_,TL,addsyn(TL,-w_sd),20000+s_))
    for s_ in FRp: conns.append(k.conn(s_,TR,addsyn(TR, w_sd),20000+s_))
    for s_ in FLp: conns.append(k.conn(s_,TR,addsyn(TR,-w_sd),20000+s_))
    for s_ in TXR: conns.append(k.conn(s_,TL,addsyn(TL, w_tox),20000+s_))   # toxin RIGHT -> turn away
    for s_ in TXL: conns.append(k.conn(s_,TR,addsyn(TR, w_tox),20000+s_))
    # MODE VETO — only HOME overrides. FORAGE and EXPLORE must NOT veto each other: in the proven
    # navigator, search (STEER spiral) and approach (TL/TR tropotaxis) run SIMULTANEOUSLY and are
    # arbitrated inside the sensory circuit (RISE suppresses STEER as odour rises). Putting a mode-WTA
    # between them destroyed foraging outright (measured: EXPLORE 98% of ticks, 1 food in 12000 steps).
    if arbiter_enabled and home_group:
        for o in home_group:
            conns.append(k.conn(o,TL,addsyn(TL,w_mode_veto),ar.T(o)))
            conns.append(k.conn(o,TR,addsyn(TR,w_mode_veto),ar.T(o)))
    conns.append(k.conn(TR,TL,addsyn(TL,w_wta),20000+TR)); conns.append(k.conn(TL,TR,addsyn(TR,w_wta),20000+TL))
    if turn_probe_ports:
        # A component runner may deliver a predeclared current into the
        # existing spiking turn neurons.  The port is not created in the
        # default organism and it never maps a Python turn to body pose.
        global TURN_PROBE_SYN
        TURN_PROBE_SYN={}
        for nid in (TL,TR):
            sid=addsyn(nid,1.0)
            TURN_PROBE_SYN[nid]=sid
            ex.append(k.ext(nid,sid))
    sy.append(k.term(TL,20000+TL)); sy.append(k.term(TR,20000+TR))
    # HOME turn commands: CPU1 comparator, vetoed unless HOME mode is active
    if path_integration_enabled and home_left and home_right:
        for nid,src in ((HTL,home_left),(HTR,home_right)):
            ne.append(k.neuron(nid,r=0.9,c=2,lam=4)); j=0
            for h in src: sy.append(k.syn(nid,j,w_home_turn,1)); conns.append(k.conn(h,nid,j,nv.T(h))); j+=1
            for o in ((mode_groups[0] if len(mode_groups) > 0 else ()) +
                      (mode_groups[2] if len(mode_groups) > 2 else ())):
                sy.append(k.syn(nid,j,w_mode_veto,1)); conns.append(k.conn(o,nid,j,ar.T(o))); j+=1
            if metabolic_sleep:
                for o in sleep_group:
                    sy.append(k.syn(nid,j,w_sleep_veto,1)); conns.append(k.conn(o,nid,j,ar.T(o))); j+=1
            sy.append(k.term(nid,20000+nid))
    # RISE population -> SEARCH spiral (klinokinesis), plus AVOID (learned danger) driving escape
    for r_,dl in zip(RISEP,RISE_DEL):
        ne.append(k.neuron(r_,r=0.6,c=2,lam=3))
        sy.extend([k.syn(r_,0,3.4,1),k.syn(r_,1,-3.4,dl),k.term(r_,20000+r_)])
        conns.extend([k.conn(POOL,r_,0,20000+POOL),k.conn(POOL,r_,1,20000+POOL)])
    if trise:
        # TPOOL mirrors POOL but pools the TOXIN sensor populations instead of the food ones.
        ne.append(k.neuron(TPOOL,r=0.5,c=1,lam=3)); tp=0
        for nid in TXL+TXR:
            sy.append(k.syn(TPOOL,tp,w_pool,1)); conns.append(k.conn(nid,TPOOL,tp,20000+nid)); tp+=1
        sy.append(k.term(TPOOL,20000+TPOOL))
        for r_,dl in zip(TRISE,RISE_DEL):          # same delay bank as RISEP: 35..110 ticks
            ne.append(k.neuron(r_,r=0.6,c=2,lam=3))
            sy.extend([k.syn(r_,0,3.4,1),k.syn(r_,1,-3.4,dl),k.term(r_,20000+r_)])
            conns.extend([k.conn(TPOOL,r_,0,20000+TPOOL),k.conn(TPOOL,r_,1,20000+TPOOL)])
    # Background curved search: ENGINE produces the primitive, food rise
    # suppresses it while directed tropotaxis takes over, and FORAGE can
    # suppress it while EXPLORE releases it.  Keep this distinct from the
    # hazard channel below: a hungry animal must still escape a toxin.
    ne.append(k.neuron(SEARCH,r=0.6,c=2,lam=5)); sj=0
    for e in ENGN: sy.append(k.syn(SEARCH,sj,w_ton,1)); conns.append(k.conn(e,SEARCH,sj,20000+e)); sj+=1
    for r_ in RISEP: sy.append(k.syn(SEARCH,sj,w_riseinh,1)); conns.append(k.conn(r_,SEARCH,sj,20000+r_)); sj+=1
    for src in FLp+FRp: sy.append(k.syn(SEARCH,sj,w_closeinh,1)); conns.append(k.conn(src,SEARCH,sj,20000+src)); sj+=1
    if arbiter_enabled and home_group:
        for o in home_group:                                  # HOME also suppresses background search
            sy.append(k.syn(SEARCH,sj,w_mode_veto,1)); conns.append(k.conn(o,SEARCH,sj,ar.T(o))); sj+=1
    # SEARCH formerly ran identically under FORAGE and EXPLORE, reducing the
    # WTA to a decoded label.  This projection makes FORAGE suppress only the
    # background search route; EXPLORE releases it.
    if w_forage_steer and arbiter_enabled and len(mode_groups) > 0:
        for o in mode_groups[0]:
            sy.append(k.syn(SEARCH,sj,w_forage_steer,1)); conns.append(k.conn(o,SEARCH,sj,ar.T(o))); sj+=1
    if metabolic_sleep:
        for o in sleep_group:
            sy.append(k.syn(SEARCH,sj,w_sleep_veto,1)); conns.append(k.conn(o,SEARCH,sj,ar.T(o))); sj+=1
    sy.append(k.term(SEARCH,20000+SEARCH))

    # Independent danger/avoidance escape.  It deliberately has no engine
    # tonic and no FORAGE veto: temporal toxin rise and learned danger must be
    # able to reorient the body in every motivational state.
    ne.append(k.neuron(STEER,r=0.6,c=2,lam=5)); si=0
    if trise:                                   # rising TOXIN excites the spiral (opposite sign to food)
        for r_ in TRISE: sy.append(k.syn(STEER,si,w_trise,1)); conns.append(k.conn(r_,STEER,si,20000+r_)); si+=1
    if avoid_enabled:
        sy.append(k.syn(STEER,si,w_avoid_steer,1)); conns.append(k.conn(avoid_source,STEER,si,20000+avoid_source)); si+=1
    if arbiter_enabled and home_group:
        for o in home_group:                                   # do not spiral-search while homing
            sy.append(k.syn(STEER,si,w_mode_veto,1)); conns.append(k.conn(o,STEER,si,ar.T(o))); si+=1
    sy.append(k.term(STEER,20000+STEER))

    # Learned valence -> lateral horn -> motor choice.  MBONs do not preserve
    # source side, so AVOID alone can only drive a directionless escape.  In
    # flies, MB output converges with innate olfactory pathways in the lateral
    # horn (Owald et al. 2015; Pinho et al. 2018).  Here each staggered LH
    # cell receives learned aversion plus an opponent antennal comparison.
    # The lateral horn naturally has an innate sensory contribution, but the
    # learned AVOID input must substantially recruit it before contact.  The
    # two turn populations retain their usual WTA dynamics, yielding an
    # evasive reorientation rather than a scripted turn direction.
    if mb_lh and avoid_enabled:
        for gates, ipsi, contra, target in (
            (LH_LEFT, FLp, FRp, TR),
            (LH_RIGHT, FRp, FLp, TL),
        ):
            for gi, gate in enumerate(gates):
                ne.append(k.neuron(gate, r=float(r_lh) + 0.15 * gi, c=2, lam=4))
                sj = 0
                sy.append(k.syn(gate, sj, w_lh_avoid, 1))
                conns.append(k.conn(avoid_source, gate, sj, 20000 + avoid_source)); sj += 1
                for sensor in ipsi:
                    sy.append(k.syn(gate, sj, w_lh_sensor, 1))
                    conns.append(k.conn(sensor, gate, sj, 20000 + sensor)); sj += 1
                for sensor in contra:
                    sy.append(k.syn(gate, sj, -w_lh_sensor, 1))
                    conns.append(k.conn(sensor, gate, sj, 20000 + sensor)); sj += 1
                sy.append(k.term(gate, 20000 + gate))
                conns.append(k.conn(gate, target, addsyn(target, w_lh_turn), 20000 + gate))

    # RELAYS: CPG excites all (distributed thresholds); steering INHIBITS -> a graded fraction passes
    for m,ph in ((MLp,CPGP[0]),(MLr,CPGP[2]),(MRp,CPGP[0]),(MRr,CPGP[2])):
        left = m in (MLp,MLr)
        for j,g in enumerate(RLY[m]):
            ne.append(k.neuron(g,r=float(THR_R[j]),c=2,lam=3))
            sy.append(k.syn(g,0,w_rc,1)); conns.append(k.conn(ph,g,0,20000+ph)); si=1
            turn_src = TR if left else TL                       # opposite-side turn neuron gates this paddle
            sy.append(k.syn(g,si,w_relay_inh,1)); conns.append(k.conn(turn_src,g,si,20000+turn_src)); si+=1
            if path_integration_enabled:
                home_src = HTR if left else HTL
                sy.append(k.syn(g,si,w_relay_inh,1)); conns.append(k.conn(home_src,g,si,20000+home_src)); si+=1
            if left:
                for source in (SEARCH, STEER):
                    sy.append(k.syn(g,si,w_steer_relay,1)); conns.append(k.conn(source,g,si,20000+source)); si+=1
            if metabolic_sleep:
                for o in sleep_group:
                    sy.append(k.syn(g,si,w_sleep_veto,1)); conns.append(k.conn(o,g,si,ar.T(o))); si+=1
            sy.append(k.syn(g,si,w_brake,1)); conns.append(k.conn(POOL,g,si,20000+POOL)); si+=1
            sy.append(k.term(g,20000+g))
            conns.append(k.conn(g,m,j,20000+g))
    return ne,sy,conns,ex


class AIFAgent3D:
    def __init__(self, tref_upper=2.0, seed=1, nr=None, config=None,
                 tonic_amp=None, tonic_gate=None, k_prop=None, k_ang=None, w_cap=None, **kw):
        """Build the embodied brain with one explicit shared configuration.

        Existing construction keywords remain supported and override the matching
        config setting.  Unknown exploratory keywords continue to flow to
        ``parts`` exactly as before; only the common baseline is centralised.
        """
        if nr and nr!=cc.NR: set_ring(nr)   # angular resolution of the heading ring
        if config is None:
            config=DEFAULT_EMBODIED_CONFIG
        if not isinstance(config, EmbodiedAgentConfig):
            # This module is also loaded by path in legacy runners.  A caller
            # may therefore hold the otherwise-identical package import
            # (``simulations.active_inference.embodied_config``), whose class
            # identity differs from our path-loaded module.  Accept that
            # exact dataclass shape, then normalise it locally.
            fields=getattr(config,"__dataclass_fields__",{})
            expected=set(EmbodiedAgentConfig.__dataclass_fields__)
            if set(fields)==expected:
                config=EmbodiedAgentConfig(**{name:getattr(config,name) for name in expected})
            else:
                raise TypeError("config must be an EmbodiedAgentConfig")
        config_fields=set(config.__dataclass_fields__)
        selected_components=kw.pop("components", None)
        self._strict = selected_components is not None
        selected_set = set(selected_components or ())
        # Keep every strict V3/V4 construction on the same calibrated
        # PAULA arbiter route as the versioned InteroceptiveV3 entrypoint.
        # Without this default, direct harnesses used the historical 0.5
        # hunger projection while the V3 entrypoint used 0.8, so two tests
        # labelled “V3” were actually different neural parameterisations.
        # An explicit caller override still wins; the legacy full-brain path
        # is intentionally left byte-compatible.
        if selected_components is not None and "arbitration.foraging_exploration" in selected_set:
            kw.setdefault("w_unc_mode", 0.8)
        self._has_compass = (not self._strict) or "navigation.heading_ring" in selected_set
        self._has_path_integration = (not self._strict) or "navigation.path_integration" in selected_set
        self._has_visual = (not self._strict) or "vision.visual_cortex" in selected_set
        self._has_memory = (not self._strict) or "learning.mushroom_body" in selected_set
        self._has_arbiter = (not self._strict) or "arbitration.foraging_exploration" in selected_set
        self._has_metabolic = (not self._strict) or "body.metabolic_organs" in selected_set
        self._has_obstacle = ("sensory.obstacle_proximity" in selected_set) or bool(kw.get("obstacle_components", False))
        self._has_obstacle_reflex = ("motor.obstacle_reflex" in selected_set) or bool(kw.get("obstacle_components", False))
        self._has_uncertainty = (not self._strict) or self._has_compass
        self._arbiter_home = self._has_path_integration
        self._mode_groups = (
            (ar.MODE if self._arbiter_home else (ar.MODE[0], ar.MODE[2]))
            if self._has_arbiter else ()
        )
        self._mode_names = (
            (list(MODES) if self._arbiter_home else ["FORAGE", "EXPLORE"])
            + (["SLEEP"] if "arbitration.metabolic_sleep" in selected_set else [])
            if self._has_arbiter else []
        )
        self.component_profile = {
            "strict": self._strict,
            "components": tuple(selected_components or ()),
            "compass": self._has_compass,
            "path_integration": self._has_path_integration,
            "visual": self._has_visual,
            "memory": self._has_memory,
            "arbiter": self._has_arbiter,
            "metabolic": self._has_metabolic,
            "obstacle": self._has_obstacle,
            "obstacle_reflex": self._has_obstacle_reflex,
            "uncertainty": self._has_uncertainty,
            "arbiter_home": self._arbiter_home,
            "mode_names": tuple(self._mode_names),
        }
        config_overrides={name:kw.pop(name) for name in tuple(kw) if name in config_fields}
        if config_overrides:
            config=config.with_overrides(**config_overrides)
        # Component selection is a build-time composition API, but the same
        # selection must also configure the closed-loop transducers.  Keep the
        # metadata path and the direct V3 config path equivalent; otherwise a
        # selected SLEEP population would be built while its afferents and
        # runtime observation remained disabled.
        if selected_components is not None and "arbitration.metabolic_sleep" in selected_components:
            config=config.with_overrides(metabolic_sleep=True)
        self.config=config
        build_kw=config.build_kwargs()
        build_kw.update(kw)
        # Preserve the effective PAULA construction parameters beside the
        # topology.  Experiments may still record their causal override
        # separately, but a trace can no longer hide a default such as the V3
        # hunger projection or a TRISE gain behind an empty ``build`` dict.
        self.build_parameters = dict(build_kw)
        # ``pistone`` is structural: when requested it adds the graded PI
        # memory cells and their speed-input ports.  Keep that fact on the
        # agent so BOTH supported stepping surfaces (``tick`` for controlled
        # probes and ``run_episode`` for MuJoCo) drive the same transducer.
        # Previously experiments had to mutate ``_pistone`` by hand after
        # construction, leaving an apparently enabled circuit inert in normal
        # use.  The default remains off.
        self._pistone=bool(build_kw.get("pistone",False))
        self._vestibular_opponent=bool(build_kw.get("vestibular_opponent",False))
        self._vestibular_opponent_bank=bool(build_kw.get("vestibular_opponent_bank",False))
        self._vestibular_phase_locked=bool(build_kw.get("vestibular_phase_locked",False))
        self._vestibular_stroke_reset=bool(build_kw.get("vestibular_stroke_reset",False))
        self._sensorimotor_estimator=bool(build_kw.get("sensorimotor_estimator",False))
        self._sensorimotor_estimator_v2=bool(build_kw.get("sensorimotor_estimator_v2",False))
        self._p_enb_trailing=bool(build_kw.get("p_enb_trailing",False))
        self._p_ena_bank=bool(build_kw.get("p_ena_bank",False))
        self._pb_phase_update=bool(build_kw.get("pb_phase_update",False))
        self._relay_efference_only=bool(build_kw.get("relay_efference_only",False))
        self._vestibular_pulse=bool(build_kw.get("vestibular_pulse",False))
        self._vestibular_comparator=bool(build_kw.get("vestibular_comparator",False))
        self._vestibular_notch=bool(build_kw.get("vestibular_notch",False))
        self._vestibular_notch_graded=bool(build_kw.get("vestibular_notch_graded",False))
        self._vestibular_notch_graded_opponent=bool(build_kw.get("vestibular_notch_graded_opponent",False))
        self._conjunctive_graded_shift=bool(build_kw.get("conjunctive_graded_shift",False))
        self._normalized_vestibular_opponent=bool(build_kw.get("vop_notch_graded_opp_normalized",False))
        self._lead_vestibular_opponent=bool(build_kw.get("vop_notch_graded_opp_lead_gain",0.0))
        self._metabolic_sleep=bool(build_kw.get("metabolic_sleep", False)) and self._has_metabolic and self._has_arbiter
        self.obstacle_config={
            "turn_gain": float(build_kw.get("obstacle_turn_gain", 3.4)),
            "onset_turn_gain": float(build_kw.get("obstacle_onset_turn_gain", 2.2)),
            "brake_gain": float(build_kw.get("obstacle_brake_gain", -1.15)),
            "wall_gain": float(build_kw.get("obstacle_wall_gain", 0.8)),
            "onset_delay": int(build_kw.get("obstacle_onset_delay", 6)),
            "proximity_gain": float(build_kw.get("obstacle_proximity_gain", 1.0)),
            "onset_gain": float(build_kw.get("obstacle_onset_gain", 1.5)),
        }
        self.net,self.core=k.load(
            k.build(*composer.compose_brain(parts, build_kw, components=selected_components)),
            neuron_class=(
                k.PhaseLockedGradedNeuron if self._vestibular_phase_locked else
                k.ConjunctiveGradedNeuron
            ) if self._conjunctive_graded_shift or self._normalized_vestibular_opponent or self._lead_vestibular_opponent or self._vestibular_phase_locked else None,
        )
        self.nb={i:u for i,u in self.net.network.neurons.items()}
        # t_ref = c*num_inputs cripples ANY high-fan-in cell, not just visual pooling: a veto relay carries
        # ~12 inhibitory synapses -> t_ref 26 -> it can barely fire even when its pathway is wide open.
        ids=list(vc.VIS_IDS())+list(VR)+list(PE)+list(UNC)+[n for v in RLY.values() for n in v]
        ids+=FLp+FRp+TXL+TXR+RISEP+ENGN+[POOL,SEARCH,STEER,TL,TR,HTL,HTR,ORN_F,ORN_T,ALN]
        ids+=OBL+OBR+OBDL+OBDR+[OBS_LEFT,OBS_RIGHT,OBS_BRAKE,OBS_WALL]
        ids+=LH_LEFT+LH_RIGHT+[VEST_CCW,VEST_CW,VEST_POS,VEST_NEG,VEST_NET_CCW,VEST_NET_CW,VEST_OPP_CCW,VEST_OPP_CW]+VEST_BANK_CCW+VEST_BANK_CW+EFF_BANK_CCW+EFF_BANK_CW+SM_SENS_CCW+SM_SENS_CW+SM_PRED_CCW+SM_PRED_CW+[SM_PE_CCW,SM_PE_CW,SM_UPDATE_CCW,SM_UPDATE_CW]+SM2_SENS_CCW+SM2_SENS_CW+SM2_PRED_CCW+SM2_PRED_CW+[SM2_PE_CCW,SM2_PE_CW,SM2_UPDATE_CCW,SM2_UPDATE_CW]+SM3_LEAD_CCW+SM3_LEAD_CW+[STROKE_CLOCK,STROKE_CCW,STROKE_CW,STROKE_RESET_CLOCK,STROKE_RESET_CCW,STROKE_RESET_CW]
        ids+=TRISE+[TPOOL]+[nv.LGI]+list(pacc.ACC)+list(pacc.ACCM)+list(nv.CDM)+[pacc.AGI]                   # TPOOL pools all 20 toxin sensors -> t_ref would be 20
        ids+=VPN+VKC+[VAPL]                  # vAC: VAPL pools all 40 vKCs, so t_ref = c*num_inputs
                                             # would sit at 40 and gate the sparsening inhibitor off
                                             # entirely; VPN pools 8 chroma cells for the same reason.
        ids+=PN+KC+[APL,STG_T,STG_F,RFX]     # PN/KC were MISSING: at t_ref=c*(KIN+1)=10 the KC layer
                                             # under-samples and the odour-specific groups vanish
        for m in range(3): ids+=list(ar.MODE[m])
        if self._metabolic_sleep:
            ids+=list(ar.SLEEP_MODE)+met.GUT_AFFERENTS+met.ENERGY_AFFERENTS+met.LOW_ENERGY_AFFERENTS+met.DIGESTION_AFFERENTS
        ids+=list(ar.HUNGER)+[nv.OPP[c] for c in range(nv.NC)]+[BL,BR,UBEL,DL,DR]
        ids+=[n for i2 in range(cc.NR) for p2 in range(cc.NP)
                for n in (cc.CL[i2][p2],cc.CR[i2][p2])]   # fan-in grew: keep t_ref sane
        if self._p_enb_trailing:
            ids+=cc.PENB_CL+cc.PENB_CR
        if self._p_ena_bank:
            ids += [nid for columns in (cc.PENA_CL, cc.PENA_CR) for column in columns for nid in column]
        for nid in ids:
            if nid in self.nb:
                self.nb[nid].upper_t_ref_bound=tref_upper; self.nb[nid].lower_t_ref_bound=1.0
                self.nb[nid].t_ref=tref_upper
        # MBON-specific: the aversive population needs a SMALL t_ref so infrequent inputs read as acausal
        # (that is what makes it forget when danger is gone); tref_upper=8 was the verified sweet spot.
        for m in MBONP:
            if m in self.nb:
                self.nb[m].upper_t_ref_bound=8.0; self.nb[m].lower_t_ref_bound=1.0; self.nb[m].t_ref=8.0
        self._mb_pp=[self.nb[m].postsynaptic_points for m in MBONP if m in self.nb]
        self._mb_clampidx=list(range(NKC))+[TEACH]
        self._kicked=False; self._fed=0; self._phase_clock_seeded=False; self._stroke_reset_clock_seeded=False
        self.tonic_amp=float(config.tonic_amp if tonic_amp is None else tonic_amp)
        self.tonic_gate=float(config.tonic_gate if tonic_gate is None else tonic_gate)
        self.k_prop=float(config.k_prop if k_prop is None else k_prop)
        self.k_ang=float(config.k_ang if k_ang is None else k_ang)
        self.w_cap=float(config.w_cap if w_cap is None else w_cap)   # k_ang 0.45 -> 0.65: MEASURED PER-TICK in the
        self.wz_tau=float(config.wz_tau)
        self.wz_comp=float(config.wz_comp)
        # Observational only: the exact raw gyro sample delivered to the
        # vestibular PAULA ports on the most recent neural tick. Tick hooks
        # run after physics, so reading ``world.yaw_rate()`` there is one
        # sample too late for a faithful isolated replay.
        self.last_gyro_yaw_rate=0.0
        self.last_obstacle_afferents={"left":0.0,"right":0.0,"left_onset":0.0,"right_onset":0.0,
                                      "distance_left":None,"distance_right":None,"contact":0.0}
        self.last_metabolic_afferents={"gut_load":0.0,"energy_store":float(self.world.energy_store if hasattr(self,"world") else 0.42),"low_energy":0.58,"digestion_rate":0.0}
        # body against a time-varying omega. The vestibular gain is sharply nonlinear -- 0.45 gives
        # tracking slope 0.085 (bump barely moves), 0.9 gives 2.43 (overshoot), and the slope~1.0
        # crossing is 0.55-0.65. With d_emd=16 this takes median heading error from 70-112 deg to
        # 15.8/13.1/15.3 deg on seeds 11/23/44 (59.8/62.9/76.7% of ticks under 20 deg).
        # w_hs_shift stays 0: visual shift drive DEGRADES tracking (per-tick r 0.609 -> 0.204).; self._us=0; self._us_mod=np.zeros(2)
        self.world=w3.World3D(seed=seed); self.img=self.world.retina(); self.t=0
    def mb_weight_min(self):
        """probe: the lowest KC->MBON weight (must never go negative now that w_min=0.0 is enforced)"""
        if not self._has_memory or not self._mb_pp:
            raise RuntimeError("mushroom-body component is not part of this agent version")
        return min(float(pp[i].u_i.info) for pp in self._mb_pp for i in range(NKC))
    def mbon_rate(self, odor_id, ticks=40, orn_gain=3.0):
        """Probe the LEARNED aversion for an odorant = summed spikes of the aversive-MBON population.
        The odour MUST be presented through the receptor cells, exactly as in life. Injecting into PN
        synapse 0 (which this used to do) now writes into the ORN_F -- i.e. FOOD-weighted -- connection,
        so probing 'toxin' actually drove the food pathway and the result came out inverted."""
        if not self._has_memory:
            raise RuntimeError("mushroom-body component is not part of this agent version")
        tot=0
        for _ in range(ticks):
            self.net.set_external_input(ORN_F,0, orn_gain if odor_id==0 else 0.0)
            self.net.set_external_input(ORN_T,0, orn_gain if odor_id==1 else 0.0)
            self.core.do_tick()
            tot+=sum(1 for m in MBONP if self.nb[m].O>0)
        self.net.set_external_input(ORN_F,0,0.0); self.net.set_external_input(ORN_T,0,0.0)
        return tot
    # ---------- birth seeds (one-time) ----------
    def birth(self, ticks=35):
        ss={o%cc.NR for o in range(-cc.NB,cc.NB+1)}
        for _ in range(ticks):
            # per-cell seed index (cc.SEEDMAP): this agent runs d7=True, so Delta-7 coverage makes
            # each ring cell's synapse layout differ and one global index seeds the WRONG synapse.
            if self._has_compass:
                for i,r in enumerate(cc.RING): self.net.set_external_input(r,cc.SEEDMAP[r], 4.0 if i in ss else 0.0)
            if self._has_path_integration:
                for c in range(nv.NC): self.net.set_external_input(nv.LAD[c][0],3,6.0)
            if self._has_arbiter:
                self.net.set_external_input(ar.HUNGER[0],ar.hunger_seed_syn(),6.0)
            if self._has_uncertainty:
                self.net.set_external_input(UNC[0],self._useed(),6.0)
            if _==2:
                self.net.set_external_input(CPGP[0],0,5.0)   # one-time CPG kick (birth seed)
                if self._vestibular_phase_locked:
                    self.net.set_external_input(STROKE_CLOCK,STROKE_CLOCK_SYN,5.0)
                    self._phase_clock_seeded=True
                if self._vestibular_stroke_reset:
                    self.net.set_external_input(STROKE_RESET_CLOCK,STROKE_RESET_CLOCK_SYN,5.0)
                    self._stroke_reset_clock_seeded=True
            self.core.do_tick()
            if _==3: self.net.set_external_input(CPGP[0],0,0.0)
        if self._has_compass:
            for r in cc.RING: self.net.set_external_input(r,cc.SEEDMAP[r],0.0)
        if self._has_path_integration:
            for c in range(nv.NC): self.net.set_external_input(nv.LAD[c][0],3,0.0)
        if self._has_arbiter:
            self.net.set_external_input(ar.HUNGER[0],ar.hunger_seed_syn(),0.0)
        if self._has_uncertainty:
            self.net.set_external_input(UNC[0],self._useed(),0.0)
    def _useed(self):  return UNC_SEED_SYN
    def _ufill(self,i): return 2 if i>0 else 1
    def drive_metabolic_afferents(self):
        """Write the physical metabolic state to V3 PAULA sensory ports."""
        if not self._metabolic_sleep:
            return None
        state=self.world.metabolic_state()
        self.last_metabolic_afferents=dict(state)
        channels=(
            (met.GUT_AFFERENTS, state["gut_load"]),
            (met.ENERGY_AFFERENTS, state["energy_store"]),
            (met.LOW_ENERGY_AFFERENTS, state["low_energy"]),
            (met.DIGESTION_AFFERENTS, min(1.0, state["digestion_rate"] / max(self.world.digestion_rate,1e-9))),
        )
        for ids,value in channels:
            for i,nid in enumerate(ids):
                gain = (float(self.config.metabolic_energy_afferent_gain)
                        if ids is met.ENERGY_AFFERENTS else 2.0)
                self.net.set_external_input(nid,0,max(0.0,float(value))*gain)
        return state

    def metabolic_hunger_inputs(self, state=None):
        """Convert body interoception into the existing PAULA hunger ports.

        This is a transducer only.  The body reports energy deficit and gut
        load; the hunger ladder integrates those currents and the arbiter
        decides which mode wins.  V3 previously wrote zero to both hunger
        ports, leaving the ladder unrelated to the physical meal.
        """
        if not self._metabolic_sleep:
            return 0.0, 0.0
        state = self.last_metabolic_afferents if state is None else state
        fill = (float(self.config.metabolic_hunger_fill_base)
                + float(self.config.metabolic_hunger_low_gain) * float(state.get("low_energy", 0.0)))
        drain = min(2.5, float(self.config.metabolic_hunger_gut_gain)
                    * max(0.0, float(state.get("gut_load", 0.0))))
        return max(0.0, fill), max(0.0, drain)

    def drive_obstacle_afferents(self, values=None):
        """Drive V4's raw whisker populations from a physical body sample.

        This method performs only the sanctioned transduction step.  It never
        computes a turn, changes a relay, or writes a pose.  Passing a mapping
        is useful for isolated replay; normal embodiment leaves ``values`` as
        ``None`` so the sample comes from ``World3D.obstacle_proximity``.
        """
        if not self._has_obstacle:
            return None
        if values is None:
            values = self.world.obstacle_proximity() if hasattr(self.world, "obstacle_proximity") else {}
        values = dict(values or {})
        left = max(0.0, float(values.get("left", 0.0)))
        right = max(0.0, float(values.get("right", 0.0)))
        for nid in OBL:
            self.net.set_external_input(nid, 0, left)
        for nid in OBR:
            self.net.set_external_input(nid, 0, right)
        def finite_distance(key):
            try:
                value = float(values.get(key, float("inf")))
            except (TypeError, ValueError):
                return None
            return value if np.isfinite(value) else None
        self.last_obstacle_afferents = {
            "left": left, "right": right,
            "left_onset": max(0.0, float(values.get("left_onset", 0.0))),
            "right_onset": max(0.0, float(values.get("right_onset", 0.0))),
            # Distances are observational metadata only.  The PAULA input
            # remains the bilateral normalized current above; preserving the
            # raw finite range makes embodied replays auditable without
            # serializing JSON Infinity when no barrier is in range.
            "distance_left": finite_distance("distance_left"),
            "distance_right": finite_distance("distance_right"),
            "contact": float(values.get("contact", 0.0)),
        }
        return self.last_obstacle_afferents
    def effective_config_manifest(self):
        """Return the configuration actually driving this instance, JSON-ready."""
        manifest=self.config.with_overrides(
            tonic_amp=self.tonic_amp, tonic_gate=self.tonic_gate,
            k_prop=self.k_prop, k_ang=self.k_ang, w_cap=self.w_cap,
            wz_tau=self.wz_tau, wz_comp=self.wz_comp,
        ).manifest()
        manifest["obstacle"] = dict(self.obstacle_config)
        manifest["obstacle"]["enabled"] = bool(self._has_obstacle and self._has_obstacle_reflex)
        return manifest

    def ring_tonic_current(self, ccw, cw):
        """Maintenance current for this tick; zero gate means legacy ungated drive."""
        if not self.tonic_amp:
            return 0.0
        if self.tonic_gate <= 0.0:
            return self.tonic_amp
        gate=max(0.0, 1.0-(abs(ccw)+abs(cw))/max(self.tonic_gate,1e-9))
        return self.tonic_amp*gate

    def drive_ring_tonic(self, ccw, cw):
        """Set the same external tonic input used by every closed-loop runner."""
        if not self._has_compass:
            return 0.0
        tonic=self.ring_tonic_current(ccw,cw)
        if tonic:
            for r in cc.RING:
                self.net.set_external_input(r,cc.TONICSYN[r],tonic)
        return tonic

    # ---------- one closed-loop tick ----------
    def tick(self, ccw, cw, speed, hunger_fill=0.6, eat=0.0, vision=True, obstacle=None):
        if vision and self._has_visual: vc.drive_from_image(self.net,self.img)
        if self._has_obstacle:
            self.drive_obstacle_afferents(obstacle)
        # RING TONIC. cc.parts wires a tonic synapse per ring cell (cc.TONICSYN) as an EXTERNAL input,
        # but an external input delivers NOTHING unless it is set every tick -- and nothing ever set it,
        # so w_tonic was multiplied by zero at every value (0.12 and 1.2 gave byte-identical traces).
        # The ring was therefore held up by w_self/lam_ring = 1.4/2 = 0.70 against r_ring=0.9 plus its
        # two neighbours: a bump >=3 cells wide survives, a narrower one collapses irreversibly. Measured
        # in the body over 8 seeds with ZERO angular velocity, the bump died at tick 36/36/36/51 on
        # 4 of 8 seeds and never recovered. Injecting the tonic gives the ring the floor it was designed
        # to have. Same class of bug as SEEDSYN: a synapse that exists, looks configured, delivers nothing.
        # The same configuration-owned external drive is used by run_episode and the live UI.
        self.drive_ring_tonic(ccw,cw)
        if self._sensorimotor_estimator:
            signed=float(ccw)-float(cw)
            self.last_gyro_yaw_rate=signed
            for nid in SM_SENS_CCW:
                self.net.set_external_input(nid, 0, signed)
                self.net.set_external_input(nid, 1, signed)
            for nid in SM_SENS_CW:
                self.net.set_external_input(nid, 0, -signed)
                self.net.set_external_input(nid, 1, -signed)
            ccw = cw = 0.0
        elif self._sensorimotor_estimator_v2:
            signed=float(ccw)-float(cw)
            self.last_gyro_yaw_rate=signed
            positive=max(0.0, signed)
            negative=max(0.0, -signed)
            for nid in SM2_SENS_CCW:
                self.net.set_external_input(nid, 0, positive)
                self.net.set_external_input(nid, 1, negative)
            for nid in SM2_SENS_CW:
                self.net.set_external_input(nid, 0, negative)
                self.net.set_external_input(nid, 1, positive)
            ccw = cw = 0.0
        elif self._relay_efference_only:
            self.last_gyro_yaw_rate=float(ccw)-float(cw)
            ccw = cw = 0.0
        elif self._vestibular_opponent:
            # Controlled direct ticks carry a signed vestibular current as the
            # established ``ccw``/``cw`` pair.  The full physical runner below
            # supplies the corresponding split raw yaw-rate instead.
            self.last_gyro_yaw_rate=float(ccw)-float(cw)
            self.net.set_external_input(VEST_CCW, 0, max(0.0, float(ccw)))
            self.net.set_external_input(VEST_CW, 0, max(0.0, float(cw)))
            ccw = cw = 0.0
        elif self._vestibular_opponent_bank:
            self.last_gyro_yaw_rate=float(ccw)-float(cw)
            for nid in VEST_BANK_CCW:
                self.net.set_external_input(nid, 0, max(0.0, float(ccw)))
            for nid in VEST_BANK_CW:
                self.net.set_external_input(nid, 0, max(0.0, float(cw)))
            ccw = cw = 0.0
        elif self._vestibular_phase_locked:
            self.last_gyro_yaw_rate=float(ccw)-float(cw)
            for nid, positive, negative in ((STROKE_CCW, ccw, cw), (STROKE_CW, cw, ccw)):
                self.net.set_external_input(nid, 0, max(0.0, float(positive)))
                self.net.set_external_input(nid, 1, max(0.0, float(negative)))
            ccw = cw = 0.0
        elif self._vestibular_stroke_reset:
            self.last_gyro_yaw_rate=float(ccw)-float(cw)
            for nid, positive, negative in ((STROKE_RESET_CCW, ccw, cw), (STROKE_RESET_CW, cw, ccw)):
                self.net.set_external_input(nid, 0, max(0.0, float(positive)))
                self.net.set_external_input(nid, 1, max(0.0, float(negative)))
            if self._pb_phase_update:
                self.net.set_external_input(PHASE_VEST_CCW, 0, max(0.0, float(ccw)))
                self.net.set_external_input(PHASE_VEST_CW, 0, max(0.0, float(cw)))
            ccw = cw = 0.0
        elif self._vestibular_pulse:
            # Direct controlled ticks use the supplied signed vestibular
            # current.  The full physical loop below instead supplies the
            # raw MuJoCo yaw-rate through the same two sensory ports.
            self.net.set_external_input(VEST_CCW, 0, max(0.0, float(ccw)))
            self.net.set_external_input(VEST_CW, 0, max(0.0, float(cw)))
            ccw = cw = 0.0
        elif self._vestibular_comparator:
            self.net.set_external_input(VEST_POS, 0, max(0.0, float(ccw)))
            self.net.set_external_input(VEST_NEG, 0, max(0.0, float(cw)))
            ccw = cw = 0.0
        elif self._vestibular_notch or self._vestibular_notch_graded or self._vestibular_notch_graded_opponent:
            signed=float(ccw)-float(cw)
            self.last_gyro_yaw_rate=signed
            for nid, current in ((VEST_NET_CCW, signed), (VEST_NET_CW, -signed)):
                self.net.set_external_input(nid, 0, current)
                self.net.set_external_input(nid, 1, current)
            ccw = cw = 0.0
        if self._has_compass:
            for i in range(cc.NR):
                for p in range(cc.NP):
                    self.net.set_external_input(cc.CL[i][p],1,ccw); self.net.set_external_input(cc.CR[i][p],1,cw)
                self.net.set_external_input(cc.PG[i],1,speed)
        if self._has_uncertainty:
            for i,nid in enumerate(UNC): self.net.set_external_input(nid,self._ufill(i),speed*1.2)
        if self._metabolic_sleep:
            state = self.drive_metabolic_afferents()
            hunger_fill, eat = self.metabolic_hunger_inputs(state)
        if self._has_arbiter:
            for i,nid in enumerate(ar.HUNGER):
                self.net.set_external_input(nid,ar.hunger_fill_syn(i),hunger_fill)
                self.net.set_external_input(nid,ar.hunger_drain_syn(i),eat)
        if self._pistone and self._has_path_integration:
            pstone.drive(self.net,speed)
        self.core.do_tick(); self.t+=1
    # ---------- readouts (measurement only) ----------
    def sample(self, n=20, **dr):
        acc=dict(mode=np.zeros(3), unc=0, pe=0, ring=np.zeros(cc.NR), sal=np.zeros(vc.NV1AZ))
        for _ in range(n):
            self.tick(**dr)
            for m in range(3): acc["mode"][m]+=sum(1 for nid in ar.MODE[m] if nid in self.nb and self.nb[nid].O>0)
            acc["unc"]+=sum(1 for nid in UNC if nid in self.nb and self.nb[nid].O>0)
            acc["pe"] +=sum(1 for j in range(cc.NR) if PE[j] in self.nb and self.nb[PE[j]].O>0)
            for i,r in enumerate(cc.RING):
                if r in self.nb: acc["ring"][i]+= self.nb[r].O>0
            for a in range(vc.NV1AZ):
                if vc.SAL(a) in self.nb: acc["sal"][a]+= self.nb[vc.SAL(a)].O>0
        return acc
    def heading(self, acc):
        r=acc["ring"]
        if r.sum()==0: return None
        return float(np.arctan2(float(np.sum(r*np.sin(cc.PHI))),float(np.sum(r*np.cos(cc.PHI)))))
    def winner(self, acc):
        return MODES[int(np.argmax(acc["mode"]))] if acc["mode"].max()>0 else "none"
    def muscles(self):
        return (float(self.nb[nv.MUS_L].S), float(self.nb[nv.MUS_R].S), float(self.nb[nv.MUS_F].S))

if __name__=="__main__":
    np.random.seed(0)
    print("AIF AGENT 3D — one PAULA brain: vision + compass + PI + self-model + confidence + EFE arbiter")
    ag=AIFAgent3D(); print(f"  network: {len(ag.nb)} neurons"); ag.birth()
    w=ag.world; import mujoco
    w.data.qpos[w.jyaw]=np.radians(35.0); mujoco.mj_forward(w.model,w.data); ag.img=w.retina()
    print("A) SELF-MODEL: prediction error when the heading belief disagrees with vision")
    a=ag.sample(20,ccw=0,cw=0,speed=0.0); print(f"   after birth (bump ~aligned): PE={a['pe']:3d}  heading={np.degrees(ag.heading(a) or 0):+6.1f}")
    print("B) CONFIDENCE: uncertainty fills while moving, drains when a landmark is seen")
    a=ag.sample(40,ccw=0,cw=0,speed=1.0,vision=False); print(f"   moving, no vision : UNC={a['unc']:3d}")
    a=ag.sample(40,ccw=0,cw=0,speed=1.0,vision=False); print(f"   moving, no vision : UNC={a['unc']:3d}")
    a=ag.sample(40,ccw=0,cw=0,speed=0.0,vision=True);  print(f"   landmark in view  : UNC={a['unc']:3d}  (should FALL)")
    print("C) ARBITER: which mode do the agent's own beliefs select?")
    a=ag.sample(30,ccw=0,cw=0,speed=0.5); print(f"   modes={np.round(a['mode'],1)} -> {ag.winner(a)}   (UNC={a['unc']}, PE={a['pe']})")
    print("@@@AIF3D DONE@@@")


# =====================================================================================================
# BEHAVIOURAL GATING — veto relays, so the mode that WINS actually selects what the body does.
# gated_agent.py established that VETOING the wrong-mode relay is far more robust than AND-gating the
# right one (a veto only needs the losing mode to be silent; an AND-gate needs its threshold tuned).
# Three steering pathways converge on the same graded muscles, each vetoed by the other two modes:
#     FORAGE  : opponent chemotaxis  (turn toward the antenna with more food odour)
#     HOME    : CPU1 home-vector comparator (the path-integration steering already verified)
#     EXPLORE : a tonic turn bias -> spiral search (the verified klinokinesis engine)
# =====================================================================================================
# ---- PORTED FROM neural_agent.py (the verified 12/12 navigator). I originally hand-rolled a 2-cell
# chemotaxis here and re-derived the exact failures these populations exist to prevent (saturation at high
# concentration, deadzone at low). "A population per function" is the project's own rule; this restores it.
# ---- MUSHROOM BODY (ported from mushroom_agent.py): the LEARNING system. PN->KC->MBON, plastic.
NPN=16; NKC=160; KIN=4; NMBON=8      # KIN=4: with 6-of-16 odorant patterns, a KC sampling 6 PNs almost
                                     # always hears BOTH odours (measured: food-only KC group EMPTY, so
                                     # there was nothing odour-specific to associate). At KIN=4 the groups
                                     # come apart: food-only 11, toxin-only 10, shared 2.
PN =[84000+i for i in range(NPN)]
KC =[84100+i for i in range(NKC)]
APL=84400                                                  # global KC inhibitor -> sparse odour codes
MBONP=[84500+i for i in range(NMBON)]                      # aversive MBON population (staggered thresholds)
AVOID=84600                                                # learned-danger output -> drives escape
DOP=NKC; TEACH=NKC+1; GATE=NKC+2                                       # neuromodulator port; aversive teaching (shock) port
MBON_R=np.linspace(0.7,1.6,NMBON)
ORN_F=84700; ORN_T=84701; ALN=84702        # antennal lobe: receptor cells + lateral-inhibition pool
STG_T=84710; STG_F=84711; RFX=84712        # sting latches: US window (toxin/food) and nociceptive reflex
_rng_mb=np.random.default_rng(2)
_rng_kc=np.random.default_rng(7)
KC_PRE={kc: sorted(_rng_kc.choice(NPN,size=KIN,replace=False).tolist()) for kc in KC}

# ---- VENTRAL ACCESSORY CALYX: the VISUAL pathway into the mushroom body ----
# Vogt, Aso, Hige, Knapek, Ichinose, Friedrich et al. 2016 (eLife 5:e14009), "Direct neural pathways
# convey distinct visual information to Drosophila mushroom bodies": a small subset of KCs responds to
# VISUAL but not olfactory stimulation, and their dendrites form a ventral accessory calyx (vAC) that is
# anatomically DISTINCT from the main olfactory calyx. Two types of visual projection neuron (VPN) run
# directly from the optic lobes to the vAC and are differentially required for COLOUR vs BRIGHTNESS
# memories. Crucially gamma-d and gamma-m KCs share the same dopaminergic valence modulation, so the
# visual and olfactory calyces converge on the SAME MBONs under the SAME teaching signal.
# Hence: CH_RG -> colour VPN -> separate visual KC population -> existing aversive MBONs. NOT chroma ->
# MBON directly, and NOT chroma mixed into the olfactory KCs (those cells do not answer to vision).
# MEASURED (renders, one object at 2.2 ahead): a toxin adds 16.2 to the image R-G sum where food adds
# 1.6 -- a 10x separation, peak 0.812 vs 0.227. The B-(R+G)/2 channel is NOT wired because it measured
# as a pure SKY detector here (EMPTY scores highest at 141.55; max identically 0.222 in all scenes),
# carrying zero food/toxin information for these two colours.
# ID RANGE: 87000+ is chosen because the highest id already in the built agent is 86021. The first
# attempt used 85000/85100, which COLLIDES with CPGP (85000-85003, the engine pacemakers) and RLY
# (85100+, the relay cells) -- the builder silently kept the incumbents, so VPN[0:4] and 24 of the 40
# vKCs resolved to MOTOR neurons and the vAC->MBON synapses would have injected walking rhythm straight
# into the aversive MBON. Nothing would have crashed. Verify every population exists before measuring.
NVPN=10; NVKC=40; VKIN=3
VPN =[87000+i for i in range(NVPN)]     # colour projection neurons (optic lobe -> vAC)
VKC =[87100+i for i in range(NVKC)]     # vAC Kenyon cells: VISUAL only, separate from the olfactory KC
VAPL=87400                              # vAC's own sparsening inhibitor. NOT the olfactory APL: adding
                                        # fan-in to APL would shift t_ref = c*num_inputs and disturb the
                                        # verified odour sparsening.
_rng_vkc=np.random.default_rng(7)
VKC_PRE={v: sorted(_rng_vkc.choice(NVPN,size=VKIN,replace=False).tolist()) for v in VKC}
def _pat():
    """SPARSE BINARY receptor pattern: an odorant drives 6 of 16 receptor types and NOTHING else.
    A dense pattern (rand()*0.9+0.1, which is what I first wrote) makes every PN respond to every odorant --
    measured PN overlap 0.93 -- so the KC layer has nothing to separate and the MBON learns to fear
    everything. Receptor specificity is what makes the identity code separable."""
    v=np.zeros(NPN); v[_rng_mb.choice(NPN,size=6,replace=False)]=1.0; return v
ODOR_PAT=[_pat(),_pat()]
while np.array_equal(ODOR_PAT[0],ODOR_PAT[1]): ODOR_PAT[1]=_pat()
# ---- SPIKING ACTIVE-INFERENCE CORE (ported from paula_aif.py) ----
# BL/BR integrate-to-bound evidence accumulators (Bayesian SPRT) with mutual inhibition and self-excitation
# memory; U is tonic and inhibited by BOTH beliefs, so it is high exactly when belief is UNRESOLVED
# (epistemic value); DL/DR are confidence = belief for one side suppressed by the other (pragmatic value).
# This is a DIFFERENT uncertainty from the UNC ladder: UNC = "I don't know where I am" (path-integration
# error), U = "I don't know where the food is" (unresolved sensory belief). Both drive EXPLORE.
BL,BR = 86000,86001
UBEL  = 86010
DL,DR = 86020,86021

def belief_parts(ne,sy,conns,ex, r_b=0.5, lam_b=10, w_ev=1.1, w_inh=-12.0, w_self=6.0, d_self=3,
                 w_tonic=1.4, w_uinh=-9.0, w_bd=3.2, w_dinh=-9.0):
    """BL/BR evidence accumulators -> U (epistemic) and DL/DR (pragmatic), ported from paula_aif.py.
    Evidence here is the agent's own graded odour populations rather than a T-maze cue."""
    for nid,other,src in ((BL,BR,FLp),(BR,BL,FRp)):
        ne.append(k.neuron(nid,r=r_b,c=2,lam=lam_b)); j=0
        for cell in src[:6]:                                   # syn block 0: sensory evidence
            sy.append(k.syn(nid,j,w_ev,1)); conns.append(k.conn(cell,nid,j,20000+cell)); j+=1
        sy.append(k.syn(nid,j,w_inh,1)); conns.append(k.conn(other,nid,j,20000+other)); j+=1   # mutual inhibition
        sy.append(k.syn(nid,j,w_self,d_self)); conns.append(k.conn(nid,nid,j,20000+nid))       # self-excitation memory
        sy.append(k.term(nid,20000+nid))
    ne.append(k.neuron(UBEL,r=0.6,c=2,lam=5))
    sy.append(k.syn(UBEL,0,w_tonic,1)); ex.append(k.ext(UBEL,0))
    sy.append(k.syn(UBEL,1,w_uinh,1)); conns.append(k.conn(BL,UBEL,1,20000+BL))
    sy.append(k.syn(UBEL,2,w_uinh,1)); conns.append(k.conn(BR,UBEL,2,20000+BR))
    sy.append(k.term(UBEL,20000+UBEL))
    for nid,pos,neg in ((DL,BL,BR),(DR,BR,BL)):
        ne.append(k.neuron(nid,r=0.6,c=2,lam=5))
        sy.append(k.syn(nid,0,w_bd,1));   conns.append(k.conn(pos,nid,0,20000+pos))
        sy.append(k.syn(nid,1,w_dinh,1)); conns.append(k.conn(neg,nid,1,20000+neg))
        sy.append(k.term(nid,20000+nid))
    return ne,sy,conns,ex

def sting_parts(ne,sy,conns,ex, w_trig=4.0, w_latch=3.0, w_kill=-9.0, r_stg=0.6, lam_stg=3,
                d_us=46, d_reflex=238, mod_cort=(2.6,0.0), mod_dopa=(0.0,2.6), w_teach_stg=5.0, w_rfx=4.0):
    """STING LATCHES — how long a contact keeps stimulating, done with neurons instead of python counters.
    A 1-tick contact pulse (the sensor) starts a self-excitating latch; the latch's OWN spikes come back as
    inhibition through a DENDRITIC DELAY of d ticks and switch it off, so the window length is set by the
    dendrite. A decaying membrane cannot serve as the timer here because PAULA resets S to 0 on every spike.
    The latch terminal carries the NEUROMODULATOR in its u_o.mod, so cortisol/dopamine is delivered by a
    neuron rather than injected by python."""
    for nid,d,mod in ((STG_T,d_us,mod_cort),(STG_F,d_us,mod_dopa),(RFX,d_reflex,(0.0,0.0))):
        ne.append(k.neuron(nid,r=r_stg,c=2,lam=lam_stg)); j=0
        sy.append(k.syn(nid,j,w_trig,1)); ex.append(k.ext(nid,j)); j+=1          # contact pulse (sensor)
        sy.append(k.syn(nid,j,w_latch,1)); conns.append(k.conn(nid,nid,j,20000+nid)); j+=1   # self-latch
        sy.append(k.syn(nid,j,w_kill,d)); conns.append(k.conn(nid,nid,j,20000+nid))          # delayed self-kill
        sy.append({"neuron_id":nid,"terminal_id":20000+nid,"type":"presynaptic",
                   "distance_from_hillock":1,"u_o":{"info":1.0,"mod":[float(mod[0]),float(mod[1])]},
                   "u_i_retro":1.0})
    return ne,sy,conns,ex

def al_parts(ne,sy,conns,ex, r_orn=0.45, w_orn=1.0, w_pn=3.2, w_al=-0.5, r_al=0.5, w_al_pool=0.5, r_pn=0.6):
    """ANTENNAL LOBE — the identity code, computed by neurons instead of numpy.
    ORN_F/ORN_T are receptor cells driven by the two odorant channels (concentration is the only thing the
    transducer supplies). PN[i] reads them through FIXED receptor-affinity weights (ODOR_PAT), and the AL
    pool inhibits every PN in proportion to TOTAL receptor drive -- lateral-inhibition gain control, the same
    POOL/APL pattern used elsewhere here. The result is that PN codes IDENTITY (the mixture ratio) rather
    than intensity, which is what numpy used to do with `idv = (cg/tot)*PAT0 + (cb/tot)*PAT1`."""
    for nid in (ORN_F,ORN_T):
        ne.append(k.neuron(nid,r=r_orn,c=2,lam=4))
        sy.extend([k.syn(nid,0,w_orn,1),k.term(nid,20000+nid)]); ex.append(k.ext(nid,0))
    ne.append(k.neuron(ALN,r=r_al,c=1,lam=3)); j=0
    for src in (ORN_F,ORN_T):
        sy.append(k.syn(ALN,j,w_al_pool,1)); conns.append(k.conn(src,ALN,j,20000+src)); j+=1
    sy.append(k.term(ALN,20000+ALN))
    for i,p_ in enumerate(PN):
        ne.append(k.neuron(p_,r=r_pn,c=2,lam=4)); j=0
        sy.append(k.syn(p_,j,w_pn*float(ODOR_PAT[0][i]),1)); conns.append(k.conn(ORN_F,p_,j,20000+ORN_F)); j+=1
        sy.append(k.syn(p_,j,w_pn*float(ODOR_PAT[1][i]),1)); conns.append(k.conn(ORN_T,p_,j,20000+ORN_T)); j+=1
        sy.append(k.syn(p_,j,w_al,1)); conns.append(k.conn(ALN,p_,j,20000+ALN)); j+=1
        sy.append(k.term(p_,20000+p_))
    return ne,sy,conns,ex

def mb_parts(ne,sy,conns,ex, r_kc=1.0, w_pk=1.1, w_apl_kc=-1.6, r_apl=1.2, w_kc_apl=0.6,
             w_km0=0.02, eta=0.08, kappa=0.9, rh_decay=0.12, w_teach=5.0, w_tref_cort=30.0, w_kc_max=None,
             r_avoid=1.1, lam_avoid=6, w_av_mbon=0.9,
             # calibrated: VPN/vKC fire 39/126 spikes on a TOXIN and EXACTLY 0 on food or an empty
             # scene (14 distinct vKC recruited). The first guess (0.9/1.3/1.1) left VPN silent --
             # the toxin lights only a few grid cells sparsely, so coincidence never reached threshold.
             vac=False, w_chr_vpn=2.0, r_vpn=0.6, w_vpn_vkc=2.6, r_vkc=1.0,
             w_vapl_vkc=-1.6, r_vapl=1.2, w_vkc_vapl=0.6, w_vkm0=0.02):
    """PN -> KC (sparse, APL-normalised) -> plastic aversive MBON population -> AVOID.
    Learns each odorant's valence from experience: a TOXIN contact releases cortisol AND drives the MBONs to
    fire (the shock), so the currently-active KCs potentiate -> 'this smell = danger'. FOOD releases dopamine,
    which suppresses aversion. Innate reflex handles the immediate sting; this is what makes it LEARNED."""
    for kc in KC:
        ne.append(k.neuron(kc,r=r_kc,c=2,lam=4)); j=0
        for pi in KC_PRE[kc]:
            pn=PN[pi]; sy.append(k.syn(kc,j,w_pk,1)); conns.append(k.conn(pn,kc,j,20000+pn)); j+=1
        sy.append(k.syn(kc,j,w_apl_kc,1)); conns.append(k.conn(APL,kc,j,20000+APL))
        sy.append(k.term(kc,20000+kc))
    ne.append(k.neuron(APL,r=r_apl,c=1,lam=3)); j=0
    for kc in KC: sy.append(k.syn(APL,j,w_kc_apl,1)); conns.append(k.conn(kc,APL,j,20000+kc)); j+=1
    sy.append(k.term(APL,20000+APL))
    if vac:
        # colour VPNs: each pools the R-G opponent cells over a band of azimuth, all elevations.
        per=max(1,vc.NAZ//NVPN)
        for vi,vp in enumerate(VPN):
            ne.append(k.neuron(vp,r=r_vpn,c=2,lam=3)); j=0
            for a in range(vi*per, min((vi+1)*per, vc.NAZ)):
                for e in range(vc.NEL):
                    src=vc.CH_RG(a,e)
                    sy.append(k.syn(vp,j,w_chr_vpn,1)); conns.append(k.conn(src,vp,j,vc.T(src))); j+=1
            sy.append(k.term(vp,20000+vp))
        # vAC Kenyon cells: sparse sampling of VPNs, sparsened by their OWN inhibitor (not APL).
        for v in VKC:
            ne.append(k.neuron(v,r=r_vkc,c=2,lam=4)); j=0
            for pi in VKC_PRE[v]:
                sy.append(k.syn(v,j,w_vpn_vkc,1)); conns.append(k.conn(VPN[pi],v,j,20000+VPN[pi])); j+=1
            sy.append(k.syn(v,j,w_vapl_vkc,1)); conns.append(k.conn(VAPL,v,j,20000+VAPL))
            sy.append(k.term(v,20000+v))
        ne.append(k.neuron(VAPL,r=r_vapl,c=1,lam=3)); j=0
        for v in VKC: sy.append(k.syn(VAPL,j,w_vkc_vapl,1)); conns.append(k.conn(v,VAPL,j,20000+v)); j+=1
        sy.append(k.term(VAPL,20000+VAPL))
    for mi,mbon in enumerate(MBONP):
        # w_min=0.0 enforces SIGN-CONSTANCY (Dale's law) in the neuron model itself: reward_hebb LTD may
        # weaken a KC->MBON synapse to silence but never invert it into an inhibitory one. This replaces the
        # per-step python weight clamp that used to walk 1288 synapses every agent step.
        nd=k.neuron(mbon,r=float(MBON_R[mi]),c=2,lam=5,plasticity="reward_hebb",
                    eta_post=eta,kappa=kappa,rh_decay=rh_decay,w_min=0.0,
                    **({'w_max':w_kc_max} if w_kc_max is not None else {}),nm_internal=1.0)
        nd["params"]["nm_reward_index"]=0        # cortisol potentiates aversive memory
        nd["params"]["nm_stress_index"]=1        # dopamine (food) suppresses it
        nd["params"]["w_tref"]=[float(w_tref_cort),0.0]   # cortisol opens the learning window
        ne.append(nd)
        for i,kc in enumerate(KC): sy.append(k.syn(mbon,i,w_km0,1)); conns.append(k.conn(kc,mbon,i,20000+kc))
        if vac:
            # vAC KC -> the SAME aversive MBONs (Vogt: gamma-d and gamma-m share dopaminergic valence).
            # Indices start at NKC+3 because DOP=NKC, TEACH=NKC+1 and GATE=NKC+2 are HARD-CODED synapse
            # ids; anything overlapping them would silently overwrite the teaching or gating input.
            # These synapses are plastic for free: plasticity is declared on the MBON, and reward_hebb
            # scales each update by that synapse's own presynaptic activity, which is exactly what makes
            # the odour-specific KC groups come apart.
            for vi,v in enumerate(VKC):
                idx=NKC+3+vi
                sy.append(k.syn(mbon,idx,w_vkm0,1)); conns.append(k.conn(v,mbon,idx,20000+v))
        # neuromodulator now arrives FROM THE STING LATCHES (nm_internal=1.0 switches on the neuron->neuron
        # neuromodulation path that neuron.py previously ignored). No python injection.
        # TONIC PLASTICITY GATE. Hige, Aso, Modi, Rubin & Turner 2015 (Neuron 88:985-998): KC->MBON
        # plasticity requires PAIRING with dopaminergic activity -- depression happens only in the
        # compartment innervated by the ACTIVATED DAN. Here nm = max(0, 1 + kappa*(m_r - m_s)) sits at
        # 1.0 with no dopamine, so the Hebbian term ran free and erased the weights (measured: -20 of
        # 65.6 over 220 steps with ZERO toxin contacts). Driving a tonic STRESS level m_s holds nm at
        # 0, so with rh_decay=0 the update is exactly zero until a teaching event lifts m_r.
        sy.append(k.syn(mbon,GATE,0.0,1)); ex.append(k.ext(mbon,GATE))
        sy.append(k.syn(mbon,DOP,0.0,1))
        conns.append(k.conn(STG_T,mbon,DOP,20000+STG_T)); conns.append(k.conn(STG_F,mbon,DOP,20000+STG_F))
        sy.append(k.syn(mbon,TEACH,w_teach,1))                # the shock drives the aversive MBONs to fire
        conns.append(k.conn(STG_T,mbon,TEACH,20000+STG_T))
        sy.append(k.term(mbon,20000+mbon))
    ne.append(k.neuron(AVOID,r=r_avoid,c=2,lam=lam_avoid))
    for mi,mbon in enumerate(MBONP):
        sy.append(k.syn(AVOID,mi,w_av_mbon,1)); conns.append(k.conn(mbon,AVOID,mi,20000+mbon))
    sy.append(k.syn(AVOID,NMBON,4.0,1)); conns.append(k.conn(RFX,AVOID,NMBON,20000+RFX))  # innate reflex, neural
    sy.append(k.term(AVOID,20000+AVOID))
    return ne,sy,conns,ex

def run_episode(ag, steps=2500, sub=16, render_every=8, gain=5.0, tox_gain=6.0, orn_gain=3.0,
                sated_steps=40, log_every=250, tick_hook=None, render_ticks=4, gate_tonic=0.0,
                vision=True, neural_input_hook=None):
    # NOTE: there is no kinematic fallback any more. Since the navigation core was ported, steering IS
    # inhibition of the CPG->muscle drive through the relay populations -- there is no turn scalar to hand
    # to a kinematic body. Locomotion is muscle-driven or it does not happen.
    """Advance a coupled organism, one sensory/neural/physics cycle per tick.

    ``sub`` batches observations, not biology. Contact is delivered on the
    next cycle after physical consumption; synaptic delays remain in PAULA.
    Legacy ``sated_steps`` and ``render_every`` use the original 16-tick unit,
    independent of caller batching. No velocity or heading is prescribed.
    """
    import mujoco
    # Keep the legacy three-mode API for V1/V2, while exposing the fourth
    # PAULA population in V3.  The body and the arbiter are still one neural
    # network; this is only a naming/observation boundary for the live loop.
    mode_names = list(ag._mode_names)
    if ag._metabolic_sleep and "SLEEP" not in mode_names:
        mode_names.append("SLEEP")
    mode_groups = list(ag._mode_groups)
    if ag._metabolic_sleep:
        mode_groups.append(ar.SLEEP_MODE)
    w=ag.world; mode_ticks={m:0 for m in mode_names}; hist=[]
    # Angular proprioception is a physical-to-neural transducer.  Its filter
    # settings belong to the same effective configuration as the ring gain;
    # experiments frequently replace ``ag.world`` after construction, so set
    # them here immediately before the closed-loop ticks begin.
    w.wz_tau=ag.wz_tau
    w.wz_comp=ag.wz_comp
    # render_every counts AGENT STEPS, and with steps=1 per call `0 % n == 0` always fires, so the
    # retina was re-rendered exactly once per 16 neural ticks no matter what was passed. Everything
    # visual -- V1, V2, salience and above all the motion detectors -- was therefore looking at a
    # frozen image that jumped every 16 ticks. render_ticks re-renders inside the sub-loop instead,
    # giving the eye its own clock at (or near) the neural rate. 0 keeps the old behaviour.
    for t in range(steps):
        msum=[0 for _ in mode_names]
        for _ in range(sub):
            retinal_period = render_ticks or max(1, render_every * 16)
            if vision and ag._has_visual and ag.t % retinal_period == 0: ag.img=w.retina()
            if vision and ag._has_visual: vc.drive_from_image(ag.net,ag.img)                       # optional sensory pathway
            od=w.odour(); oL,oR=od[w3.GOOD]; tLc,tRc=od[w3.BAD]
            cg,cb=w.odour_identity()
            ev=w.take_event()
            trig_t = 1.0 if ev=="toxin" else 0.0
            trig_f = 1.0 if ev=="food" else 0.0
            if trig_f and not ag._metabolic_sleep:
                ag._fed=sated_steps*16
            for e in ENGN: ag.net.set_external_input(e,0,1.0)                   # engine tonic
            for nid in FLp: ag.net.set_external_input(nid,0,gain*oL)            # odour (broad, tropotaxis)
            for nid in FRp: ag.net.set_external_input(nid,0,gain*oR)
            if ag._metabolic_sleep:
                # V3's toxin transducer is a local high-threshold receptor:
                # weak distant plume activity must not cancel every food
                # approach, while the near-field signal still reaches the
                # ordinary crossed-turn/TRISE escape circuit.
                toxin_gain = tox_gain * float(ag.config.metabolic_toxin_gain)
                toxin_threshold = float(ag.config.metabolic_toxin_threshold)
                tLc_in = max(0.0, float(tLc) - toxin_threshold)
                tRc_in = max(0.0, float(tRc) - toxin_threshold)
            else:
                toxin_gain = tox_gain
                tLc_in, tRc_in = tLc, tRc
            for nid in TXL: ag.net.set_external_input(nid,0,toxin_gain*tLc_in)       # toxin (short range)
            for nid in TXR: ag.net.set_external_input(nid,0,toxin_gain*tRc_in)
            obstacle = ag.drive_obstacle_afferents(w.obstacle_proximity()) if ag._has_obstacle else None
            if ag._has_memory:
                ag.net.set_external_input(ORN_F,0,float(cg)*orn_gain)      # receptor cells; the ANTENNAL
                ag.net.set_external_input(ORN_T,0,float(cb)*orn_gain)      # LOBE computes the identity code
                ag.net.set_external_input(STG_T,0,trig_t); ag.net.set_external_input(STG_F,0,trig_f)
                ag.net.set_external_input(RFX,0,trig_t)          # contact = a ONE-TICK pulse; the latch times it
            # Observational record of the actual sensory currents delivered
            # this tick.  Experiment hooks may read this after the neural and
            # physics update; it never feeds back into the circuit.
            ag.last_sensor_drives={
                "food_left":float(gain*oL), "food_right":float(gain*oR),
                "toxin_left":float(toxin_gain*tLc_in), "toxin_right":float(toxin_gain*tRc_in),
                "odor_identity_food":float(cg*orn_gain), "odor_identity_toxin":float(cb*orn_gain),
                "obstacle_left":float(obstacle["left"]) if obstacle is not None else 0.0,
                "obstacle_right":float(obstacle["right"]) if obstacle is not None else 0.0,
                "obstacle_left_onset":float(obstacle["left_onset"]) if obstacle is not None else 0.0,
                "obstacle_right_onset":float(obstacle["right_onset"]) if obstacle is not None else 0.0,
                "obstacle_contact":float(obstacle["contact"]) if obstacle is not None else 0.0,
                "vision_enabled":bool(vision and ag._has_visual),
            }
            trig_t=trig_f=0.0

            if ag._has_memory:
                for _m in MBONP:                       # tonic stress -> holds the plasticity gate shut
                    if _m in ag.nb:
                        ag.net.set_external_input(_m,GATE,gate_tonic,mod=np.array([gate_tonic,0.0]))
            if ag._has_uncertainty and UBEL in ag.nb:
                ag.net.set_external_input(UBEL,0,1.0)                               # epistemic baseline
            if ag._has_uncertainty:
                for i2,nid in enumerate(UNC):
                    if nid in ag.nb: ag.net.set_external_input(nid,ag._ufill(i2),1.0)
            if ag._metabolic_sleep:
                metabolic_state = ag.drive_metabolic_afferents()
                metabolic_fill, metabolic_drain = ag.metabolic_hunger_inputs(metabolic_state)
                for i2,nid in enumerate(ar.HUNGER):
                    ag.net.set_external_input(nid,ar.hunger_fill_syn(i2),metabolic_fill)
                    ag.net.set_external_input(nid,ar.hunger_drain_syn(i2),metabolic_drain)
            elif ag._has_arbiter:
                for i2,nid in enumerate(ar.HUNGER):                        # legacy hunger ladder
                    ag.net.set_external_input(nid,ar.hunger_fill_syn(i2),0.5)
                    ag.net.set_external_input(nid,ar.hunger_drain_syn(i2),2.5 if ag._fed>0 else 0.0)
                ag._fed=max(0,ag._fed-1)
            spd=min(1.5, w.speed()*ag.k_prop)                                    # PROPRIOCEPTION -> PI
            # ANGULAR proprioception -> the P-EN shift cells. This was missing: run_episode drove the
            # speed gate but never the shift, so the heading bump could not rotate and sat at its birth
            # column for the whole life. CPU4 was then integrating speed x cos(constant) -- a straight
            # line in a fixed direction -- which is why the home vector never matched the real one.
            # CALIBRATED, not guessed. The verified prototype (cx_navigator) advances the body by
            # kyaw*omega_cmd DEGREES per tick with kyaw=0.36, and the shift circuit is tuned to that.
            # Here one neural tick is one mj_step of 0.004 s, so the body turns degrees(yaw_rate*dt)
            # per tick and the equivalent drive is that divided by kyaw: k_ang = degrees(dt)/kyaw
            # = 0.637, and open-loop calibration confirms it: at drive 1.2 the bump rotates 0.34
            # deg/tick, matching the body. The CAP is the other half of the fix. Measured: drive 1.2
            # tracks, drive 1.5 makes the bump RUN AWAY BACKWARDS at -30 deg/tick -- the shift cells
            # saturate and the wave inverts. The verified prototype clips omega_cmd to +-1.2 for
            # exactly this reason. The CAP is 0.9, not 1.2: the compass is only LINEAR in a narrow
            # band -- previously measured drift/yaw ratio 0.97 at drive 0.9, 0.69 at 1.1, 1.62 at 1.2 --
            # and clipping the prototype to +-0.9 is what took its homing from 422 to 10 units (98%
            # closed). Above the band the PI banks into the wrong CPU4 columns. Faster turns now
            # saturate gracefully (bump lags and blurs, the biological disorientation response) and
            # vision re-anchors the ring afterwards through the GABAergic visual ring.
            # NOTE: the stroke-notch boxcar (w.wz_box = CPG_PERIOD*4) suppresses the stroke 3.5x at
            # the median and 12x at p95, but a boxcar of n ticks carries an n/2 GROUP DELAY -- 20 ticks
            # here -- and that lag inverted the measured tracking (slope +0.36 -> -0.53). Left available
            # but OFF; a zero-phase notch (or a shorter one matched to a half cycle) is the way to keep
            # the suppression without the lag.
            if ag._sensorimotor_estimator:
                wz=w.yaw_rate()
                ag.last_gyro_yaw_rate=float(wz)
                for nid in SM_SENS_CCW:
                    ag.net.set_external_input(nid, 0, wz)
                    ag.net.set_external_input(nid, 1, wz)
                for nid in SM_SENS_CW:
                    ag.net.set_external_input(nid, 0, -wz)
                    ag.net.set_external_input(nid, 1, -wz)
                ccw = cw = 0.0
            elif ag._sensorimotor_estimator_v2:
                wz=w.yaw_rate()
                ag.last_gyro_yaw_rate=float(wz)
                positive=max(0.0, float(wz))
                negative=max(0.0, -float(wz))
                for nid in SM2_SENS_CCW:
                    ag.net.set_external_input(nid, 0, positive)
                    ag.net.set_external_input(nid, 1, negative)
                for nid in SM2_SENS_CW:
                    ag.net.set_external_input(nid, 0, negative)
                    ag.net.set_external_input(nid, 1, positive)
                ccw = cw = 0.0
            elif ag._relay_efference_only:
                # The compass receives only the PAULA relay population here;
                # raw yaw remains trace data, not an unlogged host filter.
                ag.last_gyro_yaw_rate=float(w.yaw_rate())
                ccw = cw = 0.0
            elif ag._vestibular_opponent:
                # The physical rate enters only as two signed sensory
                # currents.  The temporal smoothing and subtraction are the
                # graded PAULA cells wired above, rather than a host filter.
                wz=w.yaw_rate()
                ag.last_gyro_yaw_rate=float(wz)
                ag.net.set_external_input(VEST_CCW, 0, max(0.0, wz))
                ag.net.set_external_input(VEST_CW, 0, max(0.0, -wz))
                ccw = cw = 0.0
            elif ag._vestibular_opponent_bank:
                wz=w.yaw_rate()
                ag.last_gyro_yaw_rate=float(wz)
                for nid in VEST_BANK_CCW:
                    ag.net.set_external_input(nid, 0, max(0.0, wz))
                for nid in VEST_BANK_CW:
                    ag.net.set_external_input(nid, 0, max(0.0, -wz))
                ccw = cw = 0.0
            elif ag._vestibular_phase_locked:
                wz=w.yaw_rate()
                ag.last_gyro_yaw_rate=float(wz)
                for nid, positive, negative in ((STROKE_CCW, wz, -wz), (STROKE_CW, -wz, wz)):
                    ag.net.set_external_input(nid, 0, max(0.0, positive))
                    ag.net.set_external_input(nid, 1, max(0.0, negative))
                ccw = cw = 0.0
            elif ag._vestibular_stroke_reset:
                wz=w.yaw_rate()
                ag.last_gyro_yaw_rate=float(wz)
                for nid, positive, negative in ((STROKE_RESET_CCW, wz, -wz), (STROKE_RESET_CW, -wz, wz)):
                    ag.net.set_external_input(nid, 0, max(0.0, positive))
                    ag.net.set_external_input(nid, 1, max(0.0, negative))
                if ag._pb_phase_update:
                    ag.net.set_external_input(PHASE_VEST_CCW, 0, max(0.0, wz))
                    ag.net.set_external_input(PHASE_VEST_CW, 0, max(0.0, -wz))
                ccw = cw = 0.0
            elif ag._vestibular_pulse:
                # Raw signed gyroscope -> PAULA event-rate vestibular cells.
                # The current scale is declared structurally at construction;
                # Python neither filters toward a target nor chooses a turn.
                wz=w.yaw_rate()
                ag.net.set_external_input(VEST_CCW, 0, max(0.0, wz))
                ag.net.set_external_input(VEST_CW, 0, max(0.0, -wz))
                ccw = cw = 0.0
            elif ag._vestibular_comparator:
                # Raw signed gyro first enters slow graded afferents; PAULA
                # comparators, not Python rectification, decide which P-EN
                # direction receives an event.
                wz=w.yaw_rate()
                ag.net.set_external_input(VEST_POS, 0, max(0.0, wz))
                ag.net.set_external_input(VEST_NEG, 0, max(0.0, -wz))
                ccw = cw = 0.0
            elif ag._vestibular_notch or ag._vestibular_notch_graded or ag._vestibular_notch_graded_opponent:
                # The two delay-line afferents receive raw yaw with opposite
                # polarity; their PAULA dendritic subtraction implements the
                # full-stroke moving sum before directional comparison.
                wz=w.yaw_rate()
                ag.last_gyro_yaw_rate=float(wz)
                for nid, current in ((VEST_NET_CCW, wz), (VEST_NET_CW, -wz)):
                    ag.net.set_external_input(nid, 0, current)
                    ag.net.set_external_input(nid, 1, current)
                ccw = cw = 0.0
            elif ag._has_compass:
                wzf=w.yaw_rate_net()                  # advances the filter; call exactly once per tick
                ccw=min(ag.w_cap, max(0.0, wzf)*ag.k_ang)
                cw =min(ag.w_cap, max(0.0,-wzf)*ag.k_ang)
            else:
                # Strict reactive/memory/metabolic profiles intentionally omit
                # the compass transducer.  Their motor circuit receives no
                # angular-current ports.
                ccw = cw = 0.0
            # Configuration-owned ring maintenance.  This used to be a hidden
            # hard-coded 1.0 injection here while direct ticks and the live UI
            # defaulted to zero, making nominally identical agents different.
            ag.drive_ring_tonic(ccw,cw)
            if ag._has_compass:
                for i2 in range(cc.NR):
                    for p2 in range(cc.NP):
                        ag.net.set_external_input(cc.CL[i2][p2],1,ccw)
                        ag.net.set_external_input(cc.CR[i2][p2],1,cw)
                    ag.net.set_external_input(cc.PG[i2],1,spd)
            if getattr(ag,'_pistone',False) and ag._has_path_integration: pstone.drive(ag.net,spd)
            if not ag._kicked and ag.t>2:
                ag.net.set_external_input(CPGP[0],0,5.0)
                if ag._vestibular_phase_locked and not ag._phase_clock_seeded:
                    ag.net.set_external_input(STROKE_CLOCK,STROKE_CLOCK_SYN,5.0)
                    ag._phase_clock_seeded=True
                if ag._vestibular_stroke_reset and not ag._stroke_reset_clock_seeded:
                    ag.net.set_external_input(STROKE_RESET_CLOCK,STROKE_RESET_CLOCK_SYN,5.0)
                    ag._stroke_reset_clock_seeded=True
                ag._kicked=True       # one-time CPG birth seed
            if neural_input_hook is not None:
                # Experimental stimulus port only.  It is called before the
                # PAULA tick so a component runner can prescribe a neural
                # current; it is never used by ordinary agent operation.
                neural_input_hook(ag)
            ag.core.do_tick(); ag.t+=1
            for i2, group in enumerate(mode_groups):   # accumulate activity across the sub-loop;
                msum[i2]+=sum(1 for nid in group if ag.nb[nid].O>0)  # sampling one tick in 16 misses
            w.act_muscles(ag.nb[MLp].S,ag.nb[MLr].S,ag.nb[MRp].S,ag.nb[MRr].S)   # sparse firing entirely
            # OBSERVER HOOK: called once per NEURAL tick, after the tick and the physics. Purely for
            # instrumentation (live streaming, per-tick rasters); with tick_hook=None nothing changes.
            if tick_hook is not None: tick_hook(ag)
        m=msum
        if m and max(m)>0: mode_ticks[mode_names[int(np.argmax(m))]]+=1
        if (t+1)%log_every==0:
            unc=sum(1 for nid in UNC if nid in ag.nb and ag.nb[nid].O>0)
            hist.append((t+1,w.eaten,w.tox_hits,round(w.dist_home(),1)))
            print(f"   step{t+1:5d}: eaten={w.eaten:2d} toxinX={w.tox_hits:2d} dist_home={w.dist_home():5.1f} "
                  f"mode={mode_names[int(np.argmax(m))] if m and max(m)>0 else '-':7s} UNC={unc}",flush=True)
    return hist, mode_ticks

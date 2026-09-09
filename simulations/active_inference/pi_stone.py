"""PI_STONE — path-integration memory built to the mechanism described for the bee central complex
(Stone et al. 2017, Current Biology), replacing the CPU4 latch-ladder.

WHY THE LADDER CANNOT WORK (measured, 8000 ticks, minefield/23, 2x2 on ring x drain):
  Its rungs are BISTABLE latches (self 5.0 vs threshold 1.6), so the only available "subtract" is
  "erase". Subtraction was implemented as an ANTIPODAL DRAIN from CD[c+NC/2].
    live ring + drain  -> fill 3.4   (antipodal co-drive 96% of windows: the drain erases everything)
    live ring, no drain-> fill 80.0 but span 1.53 (12/12 columns saturate: no differential left)
    dead ring + drain  -> fill 24.1, span 5.19 -- but that span encodes WHERE THE FROZEN COMPASS IS
                          STUCK, not where the agent travelled.
  So: working compass -> flat profile; broken compass -> large meaningless span. The ladder only ever
  ACCUMULATES; fill_c integrates speed*[cos(heading-phi_c)]+ = PATH LENGTH along phi_c, not net
  displacement. Out-and-back fills c and c+6 equally and the difference vanishes.

THE MECHANISM IN THE LITERATURE (differs from the ladder on every point):
  - speed (TN) arrives as UNIFORM EXCITATION to every column, not per-column cosine drive
  - heading (TB1) arrives as INHIBITION, so memory grows in columns OPPOSITE the current heading
  - there is a CONSTANT DECAY on all memory cells (a uniform leak, never an antipodal subtraction)
  - the memory units are GRADED (sigmoid rate), not latches
  Bidirectionality is free: heading th inhibits column th while th+180 accumulates; reverse the
  heading and the roles swap, so the DIFFERENCE tracks net displacement. It REQUIRES a working
  compass, because the inhibition IS the bump -- the opposite of the ladder's failure mode.

IMPLEMENTATION IN THIS SUBSTRATE:
  Spiking cells cannot hold a graded value (spiking resets S; self-excitation is bistable -- measured
  forget<=6.0 / latch>=7.0, no middle). The sanctioned pattern for a graded variable here is the
  ANALOG cell used by the graded muscles and by XACC/YACC: r=1e9 so it NEVER spikes, so S is never
  reset, and S is read directly as the value. lam sets the leak: dS = (dt/lam)(-S + I), so lam IS the
  "constant memory decay" the model calls for -- finite, not the 50000 used for a near-lossless
  integrator.
"""
import sys, importlib.util
import numpy as np
sys.path.insert(0, "/Users/arterialist/Projects/agi-research/neuron-model")
from paula_agent import ckit as k
_s = importlib.util.spec_from_file_location(
    "cc", "/Users/arterialist/Projects/agi-research/active-inference/simulations/active_inference/central_complex.py")
cc = importlib.util.module_from_spec(_s); _s.loader.exec_module(cc)


def T(n): return 20000 + n

NC = 12
MEM = [10400 + c for c in range(NC)]          # graded memory, one per column (analog: never spikes)
PHI_C = np.array([2 * np.pi * c / NC for c in range(NC)])


def parts(ne, sy, conns, ex, w_speed=1.0, w_inhib=1.2, lam_mem=8000, ring_width=1,
          graded=True, graded_gain=1.0, graded_S0=0.0):
    """Attach the memory to the EXISTING ring, so the identical circuit runs isolated and in the body.

    w_speed  uniform excitatory speed drive (external input, set every tick to the speed signal)
    w_inhib  inhibition from the ring cells covering this column's heading (the TB1 term)
    lam_mem  leak time constant = the model's constant memory decay
    graded   emit tonic release (GradedNeuron) as well as integrating. The pure-analog cell (r=1e9)
             integrates perfectly but NEVER FIRES, so it cannot drive OPP -- which is why OPP was
             silent and HOME was never selected. Integration is UNCHANGED by this flag (measured: the
             MEM antipodal differential is 0.1804 either way); it only adds the ability to emit.
    """
    per = cc.NR // NC                          # ring cells per memory column
    for c in range(NC):
        nid = MEM[c]
        # Never spikes either way, so S is never reset and stays a true graded value. With graded=True
        # the cell additionally releases transmitter in proportion to depolarisation (non-spiking tonic
        # release: C. elegans, insect LPTCs, bipolar cells), so downstream spiking cells can read it.
        if graded:
            ne.append(k.neuron(nid, r=1e9, c=2, lam=lam_mem,
                               meta={"graded_gain": graded_gain, "graded_S0": graded_S0}))
        else:
            ne.append(k.neuron(nid, r=1e9, c=2, lam=lam_mem))
        j = 0
        # (1) UNIFORM excitatory speed drive -- every column gets the same speed signal
        sy.append(k.syn(nid, j, w_speed, 1)); ex.append(k.ext(nid, j)); j += 1
        # (2) INHIBITION from the heading bump: the ring cells AT this column's direction.
        #     Memory therefore grows fastest in the columns OPPOSITE the current heading.
        for d in range(-ring_width, ring_width + 1):
            for p in range(per):
                i = (c * per + p + d) % cc.NR
                sy.append(k.syn(nid, j, -abs(w_inhib), 1))
                conns.append(k.conn(cc.RING[i], nid, j, cc.Tt(cc.RING[i]))); j += 1
        sy.append(k.term(nid, T(nid)))
    return ne, sy, conns, ex


def wire_opp(ne, sy, conns, ex, OPP, w_opp=8.0):
    """Drive the EXISTING opponent cells (built by cx_navigator from the ladder) from this memory.

    OPP[c] = [MEM[c] - MEM[c+NC/2]]+ is the same motif cx_navigator already uses; only the source
    changes. Two things fall out for free:
      (1) OPP is a SPIKING cell, so the home vector can finally drive MUS_F and HOME. The analog memory
          never fired, so nothing downstream could read it -- the root cause of "OPP is silent".
      (2) The uniform speed term (TN) is COMMON MODE across antipodal columns, so the subtraction
          cancels it in neurons. That is exactly what read()'s `v - v.mean()` does in Python, which
          means the home vector no longer needs a Python readout at all.

    Synapse indices are taken as a RUNNING COUNTER over what is already on each OPP cell (never
    hardcoded -- index collisions have cost five bugs here).
    """
    for c in range(NC):
        oid = OPP[c]
        j = 1 + max((s["synapse_id"] for s in sy
                     if s.get("neuron_id") == oid and s.get("type") == "postsynaptic"), default=-1)
        src_e, src_i = MEM[c], MEM[(c + NC // 2) % NC]
        sy.append(k.syn(oid, j, +abs(w_opp), 1)); conns.append(k.conn(src_e, oid, j, T(src_e))); j += 1
        sy.append(k.syn(oid, j, -abs(w_opp), 1)); conns.append(k.conn(src_i, oid, j, T(src_i))); j += 1
    return ne, sy, conns, ex


def wire_cpu1(ne, sy, conns, ex, HL, HR, w_cpu1=128.0):
    """Feed graded Stone memory to the existing heading-gated CPU1 comparators.

    ``HL[c]`` and ``HR[c]`` already receive the ring cells at the two
    quarter-turn offsets.  Replacing their CPU4-ladder input with ``MEM[c]``
    therefore preserves the comparator geometry: the pair's imbalance is the
    turn direction from current heading to the Stone home vector.  This is
    deliberately wiring, not a Python angle readout.

    The default gain is on the same scale as :func:`wire_opp`: a tonic graded
    release is much smaller than a binary PAULA spike.  It is only used by the
    opt-in Stone route; standard CPU4/CPU1 callers keep their existing input.
    """
    for c in range(NC):
        for hid in (HL[c], HR[c]):
            j = 1 + max((s["synapse_id"] for s in sy
                         if s.get("neuron_id") == hid and s.get("type") == "postsynaptic"), default=-1)
            sy.append(k.syn(hid, j, +abs(w_cpu1), 1))
            conns.append(k.conn(MEM[c], hid, j, T(MEM[c])))
    return ne, sy, conns, ex


def read_opp(nb, OPP, spikes=None):
    """Home vector from the OPP population. Prefer per-tick SPIKE counts (`spikes`, an NC-vector of
    rates over a window) -- membrane S is meaningless for a spiking cell because spiking resets it."""
    r = np.asarray(spikes, dtype=float) if spikes is not None else np.array(
        [float(nb[OPP[c]].O > 0) for c in range(NC)])
    x = float(r @ np.cos(PHI_C)); y = float(r @ np.sin(PHI_C))
    return r, float(np.degrees(np.arctan2(y, x)) % 360), float(np.hypot(x, y))


def drive(net, speed):
    """TRANSDUCER: forward speed -> the uniform excitatory term. Call once per tick."""
    for c in range(NC):
        net.set_external_input(MEM[c], 0, float(speed))


def read(nb):
    """Home vector from the graded memory. Population vector over S; the sign is negated because the
    memory accumulates OPPOSITE the heading, so it already points HOME."""
    v = np.array([float(nb[MEM[c]].S) for c in range(NC)])
    v = v - v.mean()                            # remove the uniform speed component; the DIFFERENCE carries it
    x = float(v @ np.cos(PHI_C)); y = float(v @ np.sin(PHI_C))
    return v, float(np.degrees(np.arctan2(y, x)) % 360), float(np.hypot(x, y))

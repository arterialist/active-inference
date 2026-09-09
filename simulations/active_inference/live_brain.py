"""LIVE BRAIN — run the MuJoCo agent for n ticks or indefinitely and watch every neuron fire in the
browser, on the same 3D brain as brain_space.html.

    python live_brain.py                 # serve on :8770, sim paused, press Run in the page
    python live_brain.py --port 8771 --world minefield --ticks 40000
    python live_brain.py --verify 120     # prove the status decoder never touches the network

The simulation runs in one worker thread; HTTP handlers only read the latest published snapshot, so a
slow browser can never stall the physics. The tick loop is byte-for-byte the one in run_episode: this
server drives the agent through AG.run_episode(a, steps=1), it does not reimplement any of it.

The status panel's decoder lives in this file (class Decoder below) but stays strictly OUTSIDE the
brain: it reads spikes and membrane potentials and writes nothing back. --verify demonstrates that.
"""
import importlib.util,sys,os,json,time,base64,io,threading,argparse,collections,asyncio
from pathlib import Path
try:
    import topo_live
except ImportError:  # package import (the CLI uses ``python -m``)
    from . import topo_live
import numpy as np
_LAUNCH_CWD=Path.cwd()
sys.path.insert(0,'/Users/arterialist/Projects/agi-research/neuron-model')
_HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(_HERE)
spec=importlib.util.spec_from_file_location('ag','aif_agent3d.py')
AG=importlib.util.module_from_spec(spec); spec.loader.exec_module(AG)
DEFAULT_CFG=AG.DEFAULT_EMBODIED_CONFIG
import mujoco
from PIL import Image
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from .live.adapters import NeuronDisplayAdapter, TopologyDisplayAdapter
    from .live.introspection import TraceStore
    from .live.protocol import PROTOCOL_VERSION, hello as protocol_hello, schema as protocol_schema, session as protocol_session
    from .live.versions import get_version
except ImportError:  # legacy ``python live_brain.py`` entrypoint
    from live.adapters import NeuronDisplayAdapter, TopologyDisplayAdapter
    from live.introspection import TraceStore
    from live.protocol import PROTOCOL_VERSION, hello as protocol_hello, schema as protocol_schema, session as protocol_session
    from live.versions import get_version

# ================================ EXTERNAL DECODER ============================================
def _pop_vector(rates, angles):
    """circular mean of a population laid out around a circle -> (angle_deg, resultant_length)"""
    r=np.asarray(rates,float); a=np.asarray(angles,float)
    if r.sum()<=0: return 0.0, 0.0
    x=float((r*np.cos(a)).sum()); y=float((r*np.sin(a)).sum())
    return float(np.degrees(np.arctan2(y,x))%360.0), float(np.hypot(x,y)/max(r.sum(),1e-9))

class Decoder:
    """Neural state -> human-readable states and actions. An INSTRUMENT, not part of the brain: it
    only reads a leaky spike-rate trace and the membrane S of the graded cells. It never calls
    set_external_input, never touches a weight, never steps the network. `--verify` proves it by
    running the same seed with and without decoding and comparing a hash of the whole network."""
    def __init__(self):
        self.AG=AG; self.vc=AG.vc; self.cc=AG.cc; self.nv=AG.nv; self.ar=AG.ar

    def bind(self, ag, ids):
        """ids = the neuron order of the rate trace the caller will hand to read()"""
        self.ag=ag; self.nb=ag.nb
        self.pos={n:i for i,n in enumerate(ids)}
        return self

    # ---- population readers (all read-only) ------------------------------------------------
    def _r(self, tr, ids):
        p=[self.pos[n] for n in ids if n in self.pos]
        return float(np.mean([tr[i] for i in p])) if p else 0.0
    def _each(self, tr, ids):
        return [float(tr[self.pos[n]]) if n in self.pos else 0.0 for n in ids]
    def _S(self, nid):
        return float(self.nb[nid].S) if nid in self.nb else 0.0

    def read(self, tr, mode_hint=None, ring_tr=None):
        """mode_hint: the arbiter's per-mode activity accumulated ACROSS the 16-tick sub-loop, which
        run_episode already computes and returns. The rest of this decoder samples the last tick of the
        step, which is fine for ladders, sensors and graded muscles but silently misses the arbiter's
        sparse firing -- reading one tick in sixteen showed every mode at 0%."""
        AG,vc,cc,nv,ar=self.AG,self.vc,self.cc,self.nv,self.ar
        out=[]
        profile=getattr(getattr(self, "ag", None), "component_profile", {})
        strict=bool(profile.get("strict", False))
        active_sections={"sense", "motor"}
        if not strict:
            active_sections.update({"drive", "belief", "spatial", "vision", "learn"})
        else:
            if profile.get("arbiter"): active_sections.add("drive")
            if profile.get("memory"): active_sections.add("learn")
            if profile.get("compass") or profile.get("path_integration"):
                active_sections.add("spatial")
            if profile.get("visual"): active_sections.add("vision")
        def item(sec,label,text,level=None,note=None):
            if sec not in active_sections:
                return
            out.append({"s":sec,"k":label,"v":text,
                        "b":None if level is None else max(0.0,min(1.0,float(level))),
                        "n":note})

        # ---------- INTEROCEPTION AND DRIVE ----------
        hun=self._r(tr,ar.HUNGER)
        item("drive","hunger",
             ("sated" if hun<.12 else "peckish" if hun<.35 else "hungry" if hun<.65 else "starving")
             +f"  {hun*100:.0f}%", hun, "interoceptive ladder, 12 cells; eating drains it")
        mode_names=list(getattr(self.ag, "_mode_names", ar.MODES)) if profile.get("arbiter", not strict) else []
        if mode_names and getattr(self.ag, "_metabolic_sleep", False):
            if "SLEEP" not in mode_names:
                mode_names.append("SLEEP")
        if not mode_names:
            mrates=[]
        elif mode_hint is None:
            mode_groups=list(getattr(self.ag, "_mode_groups", ar.MODE))
            if "SLEEP" in mode_names and len(mode_groups) == len(mode_names)-1:
                mode_groups.append(ar.SLEEP_MODE)
            mrates=[self._r(tr,group) for group in mode_groups]
        else:
            mrates=list(mode_hint)
            # A stale browser snapshot can briefly carry the previous
            # three-mode hint while a V3 network is rebuilding.  Pad that
            # observation rather than indexing the wrong label.
            mrates.extend([0.0] * max(0, len(mode_names)-len(mrates)))
            mrates=mrates[:len(mode_names)]
        if mrates:
            win=int(np.argmax(mrates)); tot=sum(mrates)
            marg=(mrates[win]-sorted(mrates)[-2])/max(mrates[win],1e-9) if tot>0 and len(mrates)>1 else 0.0
            item("drive","mode", (mode_names[win] if tot>1e-3 else "none")+f"  ({marg*100:.0f}% margin)",
                 mrates[win], "winner-take-all over the active PAULA mode populations, summed across the sub-loop")
            for m,nm in enumerate(mode_names):
                item("drive","  "+nm.lower(), f"{mrates[m]*100:.0f}%", mrates[m])

        # ---------- BELIEF AND SURPRISE ----------
        bl,br=self._r(tr,[AG.BL]),self._r(tr,[AG.BR])
        side="left" if bl>br else "right" if br>bl else "even"
        item("belief","food is",
             f"{side}  (L {bl*100:.0f}% / R {br*100:.0f}%)", max(bl,br),
             "BL/BR evidence accumulators with mutual inhibition")
        item("belief","unresolved", f"{self._r(tr,[AG.UBEL])*100:.0f}%", self._r(tr,[AG.UBEL]),
             "U is tonic and inhibited by both beliefs: high = belief not yet settled")
        dl,dr=self._r(tr,[AG.DL]),self._r(tr,[AG.DR])
        item("belief","confidence", f"{max(dl,dr)*100:.0f}%", max(dl,dr),
             "pragmatic value: one belief suppressed by the other")
        pe=self._r(tr,AG.PE)
        item("belief","prediction error", f"{pe*100:.0f}%", pe,
             "vision asserts a heading the compass bump disagrees with")
        unc=self._r(tr,AG.UNC)
        item("belief","positional uncertainty", f"{unc*100:.0f}%", unc,
             "UNC ladder: how stale the path integral is")

        # ---------- VESTIBULAR / SPATIAL ----------
        # The ring MUST be read on a ~12-tick window: it bursts, and it moves, so the decoder's slow
        # rate trace (0.985 decay, ~66 ticks) smears a healthy travelling bump into a flat ring and
        # reports "disoriented 0.00" while a 12-tick read of the same run gives 0.82.
        ring=self._each(ring_tr if ring_tr is not None else tr, cc.RING)
        hd,sharp=_pop_vector(ring,[2*np.pi*j/cc.NR for j in range(cc.NR)])
        item("spatial","heading (compass bump)", f"{hd:6.1f} deg", sharp,
             f"circular mean of the {cc.NR}-cell ring attractor; bar = bump sharpness")
        item("spatial","bump sharpness",
             ("sharp" if sharp>.55 else "broad" if sharp>.25 else "disoriented")+f"  {sharp:.2f}",
             sharp, "a blurred bump is the disorientation state, not a failure")
        spd=self._r(tr,nv.CD)
        item("spatial","speed (CD cells)", f"{spd*100:.0f}%", spd, "cosine-tuned speed population")
        lad=[self._r(tr,nv.LAD[c]) for c in range(nv.NC)]
        pv,pl=_pop_vector(lad,[2*np.pi*c/nv.NC for c in range(nv.NC)])
        item("spatial","accumulated path vector", f"{pv:6.1f} deg", pl,
             "CPU4 ladder levels as a population vector")
        item("spatial","home bearing", f"{(pv+180)%360:6.1f} deg", pl,
             "the accumulated vector, reversed; bar = how directional the ladder fill is")
        item("spatial","home distance (ladder fill)",
             f"mean {np.mean(lad)*100:.1f}%  peak {max(lad)*100:.1f}%  ({sum(1 for v in lad if v>.02)}"
             f"/{nv.NC} columns)", float(np.mean(lad)),
             "CPU4 accumulates the outbound path; an empty ladder means no home vector is being held")
        hl,hr=self._r(tr,nv.HL),self._r(tr,nv.HR)
        item("spatial","homing turn",
             ("left" if hl>hr else "right" if hr>hl else "straight")+f"  (L {hl*100:.0f}/R {hr*100:.0f})",
             abs(hl-hr), "CPU1 comparators")

        # ---------- CHEMOSENSATION ----------
        ol,orr=self._r(tr,AG.FLp),self._r(tr,AG.FRp)
        item("sense","odour", f"L {ol*100:.0f}%  R {orr*100:.0f}%", max(ol,orr),
             "bilateral food-odour populations, divisively normalised by POOL")
        item("sense","odour gradient",
             ("turn left" if ol>orr else "turn right" if orr>ol else "balanced")+f"  d={abs(ol-orr)*100:.0f}",
             abs(ol-orr), "the tropotaxis signal")
        tl_,tr_=self._r(tr,AG.TXL),self._r(tr,AG.TXR)
        item("sense","toxin", f"L {tl_*100:.0f}%  R {tr_*100:.0f}%", max(tl_,tr_),
             "short-range toxin receptors")
        rise=self._r(tr,AG.RISEP)
        item("sense","odour trend (RISE)",
             ("improving" if rise>.3 else "flat/worse")+f"  {rise*100:.0f}%", rise,
             "delay-line ladder: is the odour better than it was 35-110 ticks ago")
        if profile.get("obstacle"):
            ob = getattr(self.ag, "last_obstacle_afferents", {})
            olb = float(ob.get("left", self._r(tr, AG.OBL)))
            orb = float(ob.get("right", self._r(tr, AG.OBR)))
            item("sense", "obstacle proximity", f"L {olb*100:.0f}%  R {orb*100:.0f}%",
                 max(olb, orb),
                 "physical bilateral whisker/range transducer; no turn is selected here")
            item("sense", "obstacle onset",
                 f"L {float(ob.get('left_onset', 0.0))*100:.0f}%  R {float(ob.get('right_onset', 0.0))*100:.0f}%",
                 max(float(ob.get("left_onset", 0.0)), float(ob.get("right_onset", 0.0))),
                 "PAULA delayed comparison receives the same physical afferent")
            item("sense", "obstacle contact", "CONTACT" if float(ob.get("contact", 0.0)) > 0 else "clear",
                 float(ob.get("contact", 0.0)), "MuJoCo contact/proximity observation")

        # ---------- VISION ----------
        sal=[self._r(tr,[vc.SAL(a)]) for a in range(vc.NV1AZ)]
        if max(sal)>0:
            a=int(np.argmax(sal)); bearing=np.degrees(vc.az_of(a,vc.NV1AZ))
            item("vision","structure at", f"{bearing:+6.1f} deg  ({max(sal)*100:.0f}%)", max(sal),
                 "salience map argmax; + is to the left")
        else:
            item("vision","structure at", "nothing salient", 0.0)
        vr=[(j,self._r(tr,[AG.VR[j]])) for j in range(cc.NR) if AG.VR[j] in self.pos]
        if vr:
            vs=[v for _,v in vr]
            if max(vs)>0:
                j=vr[int(np.argmax(vs))][0]
                item("vision","visual heading anchor", f"{j*360.0/cc.NR:6.1f} deg", max(vs),
                     "the ring column vision is asserting (GABAergic anchoring)")
            else:
                item("vision","visual heading anchor","no sighting",0.0)
        item("vision","V2 structure", f"{self._r(tr,[vc.V2(a,e) for a in range(vc.NV1AZ) for e in range(vc.NV1EL)])*100:.0f}%",
             self._r(tr,[vc.V2(a,e) for a in range(vc.NV1AZ) for e in range(vc.NV1EL)]),
             "complex cells: orientation- and phase-tolerant structure")

        # ---------- LEARNING AND DEFENCE ----------
        kc=self._r(tr,AG.KC)
        item("learn","Kenyon sparseness", f"{kc*100:.1f}% active", min(1.0,kc*10),
             "sparse odour code; APL enforces it")
        item("learn","APL inhibition", f"{self._r(tr,[AG.APL])*100:.0f}%", self._r(tr,[AG.APL]))
        mb=self._r(tr,AG.MBONP)
        item("learn","learned danger (MBON)", f"{mb*100:.0f}%", mb,
             "aversive MBON population; rises for an odour trained with toxin")
        item("learn","avoid command", f"{self._r(tr,[AG.AVOID])*100:.0f}%", self._r(tr,[AG.AVOID]))
        for nid,nm in ((AG.STG_T,"toxin latch"),(AG.STG_F,"food latch"),(AG.RFX,"reflex latch")):
            v=self._r(tr,[nid])
            item("learn",nm, "OPEN" if v>0.05 else "closed", v,
                 "contact opens a timed latch (delayed self-inhibition closes it)")

        # ---------- MOTOR ----------
        s={n:self._S(n) for n in (AG.MLp,AG.MLr,AG.MRp,AG.MRr)}
        left=max(0.0,s[AG.MLp])+max(0.0,s[AG.MLr]); right=max(0.0,s[AG.MRp])+max(0.0,s[AG.MRr])
        # ML*/MR* are named for the body's +y/-y paddle, but forward is the -x axis, so ML* is the
        # paddle on the agent's RIGHT and MR* the one on its LEFT. Report the physical side.
        item("motor","right paddle (graded S)",
             f"protract {s[AG.MLp]:+.3f}  retract {s[AG.MLr]:+.3f}", min(1.0,left/2.0),
             "membrane potential read straight out as actuator force; these cells never spike. "
             "This is the ML* pair, which sits on the agent's RIGHT (body +y, forward is -x)")
        item("motor","left paddle (graded S)",
             f"protract {s[AG.MRp]:+.3f}  retract {s[AG.MRr]:+.3f}", min(1.0,right/2.0),
             "the MR* pair, on the agent's LEFT")
        item("motor","thrust", f"{(left+right)/2:.3f}", min(1.0,(left+right)/4.0))
        turn=left-right      # measured: driving ML* alone yaws +132 deg (counter-clockwise = agent's left)
        item("motor","turn",
             ("left" if turn>0.02 else "right" if turn<-0.02 else "straight")+f"  {turn:+.3f}",
             min(1.0,abs(turn)),
             "sign verified against the physics: ML* thrust alone turns the body counter-clockwise. "
             "Steering is inhibition of one side's muscles, never a yaw command")
        rl={m:sum(1 for n in AG.RLY[m] if self._r(tr,[n])>0.05) for m in AG.RLY}
        item("motor","relay gating",
             "  ".join(f"{'RL'[m in (AG.MRp,AG.MRr)]}{'pr'[m in (AG.MLr,AG.MRr)]}:{v}/{AG.NRLY}"
                       for m,v in rl.items()),
             float(np.mean(list(rl.values())))/AG.NRLY,
             "how many threshold-staggered relays are open per muscle")
        cpg=self._each(tr,AG.CPGP)
        item("motor","CPG phase",
             f"cell {int(np.argmax(cpg))} of {len(cpg)}" if max(cpg)>0 else "quiet",
             max(cpg), "the stroke pacemaker ring")
        item("motor","engine", f"{self._r(tr,AG.ENGN)*100:.0f}%", self._r(tr,AG.ENGN),
             "coprime pacemakers: the tonic drive that keeps the CPG alive")
        item("motor","brake (POOL)", f"{self._r(tr,[AG.POOL])*100:.0f}%", self._r(tr,[AG.POOL]),
             "normalisation pool also brakes near food so the agent converges")
        return out

# ==============================================================================================

WORLDS={
 "meadow":    dict(n_food=9,  n_tox=3,  arena=11.0, seed=11),
 "minefield": dict(n_food=7,  n_tox=26, arena=11.0, seed=22),
 "sparse":    dict(n_food=4,  n_tox=6,  arena=13.0, seed=33),
 # V4 is a deterministic causal fixture: a full-width wall and no food
 # source.  Keeping the challenge free of random respawns makes the live
 # replay agree with the acceptance harness instead of testing lucky search.
 "obstacle_detour": dict(n_food=0, n_tox=0, arena=7.0, seed=11, barrier="head_on_wall"),
 "obstacle_corner": dict(n_food=0, n_tox=0, arena=7.0, seed=11, barrier="corner"),
 "obstacle_chicane": dict(n_food=0, n_tox=0, arena=7.0, seed=11, barrier="chicane"),
 "obstacle_maze": dict(n_food=0, n_tox=0, arena=7.0, seed=11, barrier="maze"),
}
# ---------------------------------------------------------------------------------------------
# TUNABLE PARAMETERS exposed to the live UI.
#
# kind="config" -> updates EmbodiedAgentConfig before the network is built
# kind="kwarg"  -> an uncommon exploratory construction setting, forwarded explicitly
# kind="attr"   -> retained only for UI metadata such as the world seed
#
# `ship` is the value the shipped agent uses today. `fix` is the value that MEASUREMENT supports,
# where the two differ -- so "Verified fixes" in the UI is not a guess, each one has a log behind it.
#
# HAZARD: `w_self` is in BOTH the ring allowlist (_cck) and the ladder allowlist (_cpk) and would
# silently change two unrelated circuits at once. It is deliberately NOT exposed.
PARAM_SPEC = [
 # name            kind    ship    fix     lo     hi    step  group        help
 ("tonic_amp",    "config", DEFAULT_CFG.tonic_amp, 1.0,    0.0,   3.0,  0.05, "L1 ring",
  "Canonical ring-maintenance current, injected every neural tick by the same configuration used by "
  "direct ticks and closed-loop episodes. 1.0 preserves the former run_episode default; 0.0 is kept "
  "only as an explicit legacy control."),
 ("tonic_gate",   "config", DEFAULT_CFG.tonic_gate, 0.30,   0.0,  2.0,  0.05, "L1 ring",
  "Velocity gate on the tonic. 0 means ungated, matching the former closed-loop episode. Positive "
  "values suppress the tonic as shift drive rises and must be measured as a separate configuration."),
 ("w_tonic",      "config", DEFAULT_CFG.w_tonic, 0.60,   0.0,   2.0,  0.02, "L1 ring",
  "Weight of the tonic synapse. Inert unless tonic_amp > 0 (the synapse exists but nothing drives it)."),
 ("k_ang",        "config", DEFAULT_CFG.k_ang,   0.65,   0.10,  1.20, 0.01, "L2 compass",
  "Vestibular gain, omega -> shift-cell drive. Sharply NONLINEAR and non-monotonic: 0.40->ratio 0.84, "
  "0.42->1.00, 0.44->0.23, 0.46->1.35, 0.50->1.45. There is no stable unity point."),
 ("r_conj_lo",    "config", DEFAULT_CFG.r_conj_lo, 0.90,   0.50,  2.50, 0.05, "L2 compass",
  "Lowest shift-cell threshold. A ring spike alone contributes ~0.70, so at 1.35 a slow turn "
  "(drive ~0.21) can never reach threshold -- slow turns register EXACTLY ZERO degrees. At 0.90 they "
  "register 43-56 deg. Only safe now that the tonic gives the ring a floor."),
 ("r_conj_hi",    "config", DEFAULT_CFG.r_conj_hi, 1.30,   0.60,  3.00, 0.05, "L2 compass",
  "Highest shift-cell threshold. Widening the spread thins the cells covering low drive: 0.90-1.75 "
  "sends slow turns back to ratio 0.00-0.06, while 0.90-1.30 keeps them at 0.24-0.31."),
 ("w_hs_shift",   "config", DEFAULT_CFG.w_hs_shift, 0.0,    0.0,   2.0,  0.05, "L2 compass",
  "Visual (HS) drive onto the shift cells. SHIPPED 0.0 because it MEASURABLY HURTS: per-tick "
  "r(omega,ring) falls 0.609 -> 0.204 as this goes 0 -> 0.8. Raise only to reproduce that result."),
 ("d_emd",        "config", DEFAULT_CFG.d_emd, 16,     2,     32,   1,    "L2 vision",
  "Reichardt delay; sets EMD velocity tuning (peak near delta_phi/d_emd). 4 -> 16 raised per-tick "
  "r(omega,HS) from 0.347+-0.072 to 0.587+-0.031, better on all 4 seeds. NOTE: with w_hs_shift=0 this "
  "pathway is disconnected from the compass, so changing it has no behavioural effect."),
 ("w_hs_anti",    "config", DEFAULT_CFG.w_hs_anti, 1.6,    0.0,   3.0,  0.1,  "L2 vision",
  "Within-cell opponency (HS excited by preferred, inhibited by anti-preferred, pre-threshold). "
  "Uniform pooling of SIGNED flow is already the correct rotation matched filter."),
 ("r_pg",         "config", DEFAULT_CFG.r_pg, 1.7,    0.5,   3.0,  0.05, "L3 path integration",
  "Speed-gate threshold. PG is an AND of (bump, speed)."),
 ("w_pg",         "config", DEFAULT_CFG.w_pg, 2.4,    0.5,   4.0,  0.05, "L3 path integration",
  "Ring->PG weight. SHIPPED 1.4 with r_pg=1.7 gives 0.70+0.63 = 1.33 per tick -- BELOW threshold, so "
  "the gate CANNOT FIRE even with both inputs. MEASURED 2.4/2.6: PG/tick 0.63 -> 5.38, moving-vs-"
  "stopped 1.48/0.00 -> 12.60/0.03, CD/tick 0.35 -> 1.97. Verified fix, not currently shipped."),
 ("w_pg_s",       "config", DEFAULT_CFG.w_pg_s, 2.6,    0.5,   4.0,  0.05, "L3 path integration",
  "Speed->PG weight. Must stay below threshold ALONE or PG fires on speed with no bump, flattening "
  "CD's cosine tuning across all 36 columns."),
 ("k_pi",         "config", DEFAULT_CFG.k_pi, 1.0,    0.1,  500.0, 0.1,  "L3 path integration",
  "Gain into the ANALOG accumulators XACC/YACC (r=1e9, never spike; lam=50000, near-lossless; read "
  "from membrane S). MEASURED: at k_pi=1.0 they charge to only 0.006-0.016 over 3200 ticks and |S| is "
  "FLAT across distance. Needs orders of magnitude more gain -- untested, and the most promising "
  "untried L3 experiment."),
 ("cd_neg",       "config", DEFAULT_CFG.cd_neg, 0.0,    0.0,   2.0,  0.05, "L3 path integration",
  "Wires the NEGATIVE cosine lobe as inhibition, making CD a true signed cosine (Stone 2017 CPU4). "
  "Signed CD gives a clean population-vector ANGLE but nothing accumulates; RECTIFIED (0.0) "
  "accumulates but the angle blurs. The two L3 criteria separate on exactly this knob."),
 ("cd_cut",       "config", DEFAULT_CFG.cd_cut, 0.05,   0.0,   0.9,  0.05, "L3 path integration",
  "CD cosine tuning width. At 0.05 the pool spans ~170 deg so every column and its ANTIPODE fire "
  "almost equally. Narrowing it was measured WORSE (error 47.8 -> 82.2)."),
 ("w_cd_acc",     "kwarg", 1.2,    4.0,    0.5,  12.0,  0.1,  "L3 path integration",
  "CD -> ACC drive. 1.2/lam(6) = 0.20 vs threshold 0.9 = SILENT. 9.0 gives 1.50 = pure RELAY (one "
  "spike in, one out). The integrating window is between. Only matters when accum is on."),
 ("w_acc_self",   "kwarg", 0.85,   7.0,    0.0,  10.0,  0.1,  "L3 path integration",
  "ACC self-excitation. MEASURED hold test: <=6.0 retains 0% after drive stops, >=7.0 retains 100% "
  "forever. Only matters when accum is on."),
 # ---- STRUCTURAL GATES: these build or omit whole POPULATIONS, they are not weight tweaks.
 # Neuron counts measured against the former 1746-cell shipped baseline.  The
 # normal shared configuration now enables TRISE, so its current baseline is
 # 1753 cells.
 ("trise",        "config", DEFAULT_CFG.trise, 1, 0,     1,    1,    "structure (populations)",
  "+7 cells (TPOOL + 6 TRISE). Pools TXL+TXR -- the SUM of both toxin sensors -- and detects its "
  "temporal RISE through a 35-110 tick delay bank, then excites STEER. This IS the klinokinesis fix "
  "for head-on toxin blindness: lateral toxins are seen (turn sign 4/4, repulsion +1.04) but head-on "
  "ones are INVISIBLE (+-0.05 despite ~6000 TXL/TXR spikes) because TL and TR cancel in the opponent "
  "comparison. A summed channel does not cancel. Enabled by default after a physical head-on causal "
  "test; disable only as that experiment's output-path control."),
 ("w_trise",      "kwarg", 3.0,    3.0,    0.0,   8.0,  0.1,  "structure (populations)",
  "TRISE -> STEER weight. Only matters when trise is on."),
 ("vac",          "kwarg", 0,      0,      0,     1,    1,    "structure (populations)",
  "+51 cells (VPN 10, VKC 40, VAPL). Visual calyx into the mushroom body -- colour opponent channels "
  "feeding learned valence, so the agent could SEE a toxin rather than only smell/touch it. Ids sit at "
  "87000+ after an earlier collision put them on CPGP/RLY, which would have wired walking rhythm into "
  "the aversive MBON with nothing crashing. Needs an odour-ablated validation run to mean anything."),
 ("accum",        "kwarg", 0,      0,      0,     1,    1,    "structure (populations)",
  "+145 cells (ACC_G 72, ACCM_G 72, AGI). The spiking path-integration accumulator, alongside the "
  "analog XACC/YACC. Its cells are bistable: w_acc_self <=6.0 forgets everything within 900 ticks, "
  ">=7.0 latches permanently, and no graded regime exists between them."),
 ("w_peg",        "kwarg", 0.0,    0.0,    0.0,   3.0,  0.1,  "structure (populations)",
  "+36 cells (PEG). Offset-free maintenance loop through the bridge, one per ring column. Hulse 2021 "
  "reports the connectome REPLACED E-PG<->P-EG with E-PG<->E-PG, so this is the older wiring."),
 ("pb_eb_bridge", "kwarg", 0,      0,      0,     1,    1,    "structure (populations)",
  "+72 PB update cells, plus +72 bilateral maintenance cells when w_peg is on. This is an explicit "
  "PAULA P-EN -> PB -> E-PG bridge; it is a visible, opt-in experimental topology, not an accepted "
  "compass fix. The current long-turn trace fails because its slow PB directional residual accumulates."),
 ("pb_phase_update", "kwarg", 0,   0,      0,     1,    1,    "structure (populations)",
  "+118 cells: 72 PB phase-gated update relays, a 44-cell CPG phase population, and two fast gyro "
  "afferents. Each relay requires local E-PG, signed gyro, and a declared gait phase. Experimental; it "
  "is an ordinary PAULA circuit, not the special sample/hold neuron."),
 ("w_eff",        "kwarg", 0.0,    0.0,    0.0,   3.0,  0.1,  "structure (populations)",
  "+0 cells, adds synapses: efference copy from the motor side onto the compass. An internal estimate "
  "of self-motion that does not depend on vision or vestibular transduction. Never measured."),
 ("w_rly_eff",    "kwarg", 0.0,    0.0,    0.0,   3.0,  0.1,  "structure (populations)",
  "+0 cells, adds synapses: efference copy routed via the motor relays instead of directly."),
 ("sh_bank",      "kwarg", 0,      0,      0,     1,    1,    "structure (populations)",
  "+0 cells, re-wires the shift offsets so each of the NP cells per column steps a DIFFERENT distance "
  "-- a velocity-scaled bank (Turner-Evans 2017: P-EN/E-PG phase offset varies linearly with rotational "
  "velocity). Isolated it makes bump speed scale with velocity (1.97x vs body 2x); in the body it "
  "measured negative. I once killed this on a wrapped-endpoint metric bug, so treat the negative as soft."),
 ("w_lgi",        "kwarg", 0.0,    0.0,    0.0,   3.0,  0.1,  "structure (populations)",
  "+1 cell (LGI). Global inhibition across the CPU4 ladder, replacing an antipodal drain that appears "
  "in no organism. Measured binary: at 0.01 the ladder saturates 16/16 with |h|=0.00, at 0.04 it "
  "collapses to 0. Only relevant to the old ladder, not pi_accum."),
 ("d7",           "kwarg", 1,      1,      0,     1,    1,    "structure (populations)",
  "Delta-7 structured inhibition (8 cells, built by default via the agent's override). This is what "
  "keeps the ring ALIVE -- d7=False gives 0 ring spikes in 400 ticks. Turn off only to reproduce that."),
 ("metabolic_sleep", "config", DEFAULT_CFG.metabolic_sleep, 1, 0, 1, 1, "V3 interoception",
 "Builds the V3 gut/energy afferents and PAULA SLEEP population. SLEEP suppresses search/motor drive "
 "while digestion continues; this is the accepted two-seed embodied V3 composition."),
 ("obstacle_turn_gain", "kwarg", 3.4, 3.4, 0.0, 8.0, 0.1, "V4 tactile",
  "Crossed bilateral obstacle command gain onto the established TL/TR neurons."),
 ("obstacle_onset_turn_gain", "kwarg", 2.2, 2.2, 0.0, 8.0, 0.1, "V4 tactile",
  "Delayed obstacle-onset contribution to the bilateral command cells."),
 ("obstacle_brake_gain", "kwarg", -1.15, -1.15, -4.0, 0.0, 0.05, "V4 tactile",
  "Shared obstacle brake onto the graded relays; zero is a reflex-output ablation."),
 ("obstacle_wall_gain", "kwarg", 0.8, 0.8, 0.0, 3.0, 0.05, "V4 tactile",
  "Slow wall-presence gate; it does not write a pose or turn directly."),
 ("seed",         "attr",  -1,     -1,     -1,    999,  1,    "world",
  "Agent/world seed. -1 = use the selected world's own seed. Ring death is seed-dependent: it dies "
  "on 11/77/5/13 and survives on 23/91/7 (44 partial), so sweep seeds before believing any result."),
]
PARAM_BY_NAME = {p[0]: p for p in PARAM_SPEC}

# STRUCTURAL GATES get their own selector in the UI, because they are not tuning: each builds or omits
# a POPULATION, so flipping one changes what the brain IS. `cells` is the measured neuron delta against
# the former 1746-cell shipped baseline (verified by construction, not estimated). `on` is the value that
# enables the part -- for the weight-gated ones (w_peg, w_lgi, w_eff, w_rly_eff) "enabled" means a
# sensible non-zero weight rather than a boolean.
STRUCT_GATES = {
  "trise":     {"cells": "+7",   "on": 1,   "label": "Toxin-rise klinokinesis",
                "sub": "TPOOL + 6 TRISE -> STEER. Sums TXL+TXR so a HEAD-ON toxin is visible; "
                       "the opponent comparison cancels it. Enabled by default; physical causal evidence is maintained."},
  "vac":       {"cells": "+51",  "on": 1,   "label": "Visual calyx (vAC)",
                "sub": "VPN 10 + VKC 40 + VAPL. Colour opponent channels into the mushroom body, so "
                       "toxins could be SEEN rather than only smelled and touched."},
  "accum":     {"cells": "+145", "on": 1,   "label": "Spiking PI accumulator",
                "sub": "ACC_G 72 + ACCM_G 72 + AGI, alongside the analog XACC/YACC. Cells are "
                       "bistable: forgets below w_acc_self 6.0, latches above 7.0."},
  "w_peg":     {"cells": "+36",  "on": 1.0, "label": "P-EG maintenance loop",
                "sub": "Offset-free loop through the bridge, one per column. Hulse 2021 reports the "
                       "connectome REPLACED E-PG<->P-EG with E-PG<->E-PG, so this is older wiring."},
  "pb_eb_bridge": {"cells": "+72", "on": 1, "label": "Explicit PB/EB update bridge",
                "sub": "P-EN -> bilateral PB update relays -> E-PG. With P-EG enabled it also adds "
                       "bilateral E-PG -> PB maintenance tracts. Experimental and presently negative in the "
                       "long embodied turn; exposed so the diagram shows what is actually built."},
  "pb_phase_update": {"cells": "+118", "on": 1, "label": "CPG-phase-gated PB update",
                "sub": "A 44-cell PAULA phase population gates local PB update relays with fast signed gyro "
                       "and local E-PG drive. A separate experimental hypothesis; it requires the PB/EB bridge."},
  "w_lgi":     {"cells": "+1",   "on": 0.8, "label": "Ladder global inhibitor",
                "sub": "Normalises the CPU4 ladder instead of an antipodal drain that exists in no "
                       "organism. Measured binary: saturates at 0.01, collapses at 0.04."},
  "w_eff":     {"cells": "synapses only", "on": 1.0, "label": "Efference copy -> compass",
                "sub": "Motor-side estimate of self-motion, needing neither vision nor vestibular "
                       "transduction. Adds synapses, no new cells. Never measured."},
  "w_rly_eff": {"cells": "synapses only", "on": 1.0, "label": "Efference copy via relays",
                "sub": "Same idea, routed through the motor relays instead of directly."},
  "sh_bank":   {"cells": "rewires shift", "on": 1, "label": "Velocity-scaled shift bank",
                "sub": "Each of the NP cells per column steps a DIFFERENT distance (Turner-Evans "
                       "2017). Isolated: bump speed scales with velocity. In body: measured negative, "
                       "but I once killed this on a wrapped-endpoint metric bug, so treat that softly."},
  "d7":        {"cells": "8 (on by default)", "on": 1, "label": "Delta-7 inhibitors",
                "sub": "What keeps the ring ALIVE: d7 off gives 0 ring spikes in 400 ticks. "
                       "Disable only to reproduce that failure."},
  "metabolic_sleep": {"cells": "+22", "on": 1, "label": "V3 metabolic SLEEP",
                       "sub": "Gut, usable-energy, low-energy and digestion afferents plus the fourth PAULA "
                              "SLEEP population. It suppresses search/motor drive while digestion continues."},
}

def jpg(arr,q=58):
    b=io.BytesIO(); Image.fromarray(arr.astype(np.uint8)).save(b,format="JPEG",quality=q)
    return base64.b64encode(b.getvalue()).decode()

class Sim:
    def __init__(self, world="meadow", sub=16, version="v1"):
        self.version_spec=get_version(version)
        self.version=self.version_spec.id
        if self.version == "v4" and world == "meadow":
            # V4's catalyst is the barrier world.  Callers can still select a
            # legacy meadow explicitly through the command protocol.
            world = "obstacle_detour"
        self.sub=sub; self.lock=threading.Lock()
        self.cmd_lock=threading.Lock()
        self.running=False; self.budget=0      # ticks still owed; <0 means indefinitely
        self.pending=world                     # the worker builds the world, see below
        self.snap={"ready":False}
        self.world_name=world; self.built=False
        self.trace=None
        self.trace_error=None
        # live parameter overrides, name -> value. Empty = shipped defaults.
        self.params={}
        self.build_err=None
        # the wiring diagram must track STRUCTURAL changes; regenerated off-thread after every build
        self.topo=None; self.topo_ver=0; self.topo_busy=False; self.topo_err=None
        # EVERYTHING is constructed on the worker thread. A mujoco.Renderer binds the GL context of the
        # thread that creates it, so a renderer built here and used in loop() blocks forever on first use.
        threading.Thread(target=self.loop,daemon=True).start()

    # ---- construction -------------------------------------------------------------------------
    def build(self, name):
        cfg=WORLDS[name]
        # PARAMETER APPLICATION. Shared settings first update the single
        # EmbodiedAgentConfig; uncommon exploratory knobs remain explicit
        # constructor kwargs.  This prevents the former split in which the
        # episode loop used tonic=1 while direct/live ticks constructed tonic=0.
        kwargs={}; config_overrides={}
        config_fields=set(DEFAULT_CFG.__dataclass_fields__)
        for nm,val in self.params.items():
            spec=PARAM_BY_NAME.get(nm)
            if spec is None: continue
            if nm=="seed": continue
            # BOOLEAN GATES build or omit whole populations. They must be passed as real bools:
            # `sh_bank=0.0` is falsy so it happens to work, but `d7=0.0` vs False and `trise=1.0` vs
            # True are the kind of mismatch that silently half-configures a circuit.
            if nm in ("accum","trise","vac","sh_bank","d7","metabolic_sleep"):
                val=bool(round(val))
            if nm=="d_emd": val=int(val)
            (config_overrides if nm in config_fields else kwargs)[nm]=val
        seed=int(self.params.get("seed",-1))
        if seed < 0: seed=cfg["seed"]
        np.random.seed(seed)
        self.build_err=None
        embodied_config=DEFAULT_CFG.with_overrides(**config_overrides)
        # All versions use the same PAULA graph builder.  The selected
        # composition only changes the declared build fragments (V3 adds the
        # real metabolic afferents/SLEEP population); it is never a Python
        # policy layer.
        a=AG.AIFAgent3D(seed=seed,config=embodied_config,
                        components=self.version_spec.components, **kwargs)
        a.world=AG.w3.World3D(n_food=cfg["n_food"],n_tox=cfg["n_tox"],arena=cfg["arena"],seed=seed,
                              barrier=cfg.get("barrier"))
        a.img=a.world.retina(); a.birth()
        if getattr(a, "_has_obstacle", False):
            # Publish the physical starting observation without injecting it
            # into the just-seeded network; the first closed-loop tick will
            # sample and drive the afferents in the normal order.
            a.last_obstacle_afferents = dict(a.world.obstacle_proximity())
        self.a=a; self.w=a.world; self.cfg=cfg; self.world_name=name
        self.ids=sorted(a.net.network.neurons.keys())        # SAME order export_brain.py uses
        self.units=[a.nb[i] for i in self.ids]
        self.act=np.zeros(len(self.ids),dtype=np.float32)
        self.rate=np.zeros(len(self.ids),dtype=np.float32)
        self.ring_tr=np.zeros(len(self.ids),dtype=np.float32)   # 12-tick ring readout
        self.pub_every=16       # the heavy JSON snapshot rides the agent-step boundary; the raster
                                # goes out on EVERY tick over the websocket instead
        self.frames=collections.deque(maxlen=4096)   # (tick, packed raster) awaiting broadcast
        self.frame_evt=threading.Event()
        # The binary websocket remains the low-latency raster path.  TraceStore is the seekable
        # microscope path: it retains post-tick intracellular, postsynaptic-point, and physical
        # transducer state for a bounded window without changing the PAULA network.
        self.trace=TraceStore(self.ids,self.units,a.net.network,
                              max_ticks=int(os.environ.get("AIF_TRACE_TICKS","4096")))
        self.trace_error=None
        self.dec=Decoder().bind(a,self.ids)      # read-only instrument, see --verify
        self.graded=[AG.MLp,AG.MLr,AG.MRp,AG.MRr]      # cells with r=1e9: they never spike, S is the output
        self.step=0; self.ticks=0; self.t0=time.time()
        self.cam=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(self.cam)
        self.cam.lookat[:]=[0,0,0.3]; self.cam.distance=cfg["arena"]*2.1
        self.cam.elevation=-42; self.cam.azimuth=120
        # rendered larger than they are displayed, so the click-to-expand panels are sharp rather
        # than an upscaled thumbnail; the page shows them at 224px until you enlarge one
        self.R3=mujoco.Renderer(self.w.model,height=390,width=520)
        self.RP=mujoco.Renderer(self.w.model,height=300,width=520)
        self.eye=mujoco.mj_name2id(self.w.model,mujoco.mjtObj.mjOBJ_CAMERA,"eye")
        self.publish(force=True)
        self._regen_topology()

    def _regen_topology(self):
        """Rebuild the 3D wiring payload to match this build. Off-thread: it spawns a subprocess and
        the sim must stay responsive. The page polls topo_ver and swaps the diagram when it changes."""
        snapshot=dict(self.params)
        def work():
            self.topo_busy=True; self.topo_err=None
            try:
                payload,info=topo_live.regenerate(snapshot,iters=200,version=self.version)
                self.topo=TopologyDisplayAdapter().adapt(payload); self.topo_ver+=1
                print(f"[topology] {info}",flush=True)
            except Exception as e:
                self.topo_err=f"{type(e).__name__}: {e}"
                print(f"[topology] FAILED {self.topo_err}",flush=True)
            finally:
                self.topo_busy=False
        threading.Thread(target=work,daemon=True).start()

    # ---- the worker ---------------------------------------------------------------------------
    def loop(self):
        while True:
            with self.cmd_lock:
                pend=self.pending; self.pending=None
            if pend:
                if self.built:
                    try: self.R3.close(); self.RP.close()
                    except Exception: pass
                    self.running=False; self.budget=0
                try:
                    self.build(pend); self.built=True
                except Exception as e:
                    import traceback; traceback.print_exc()
                    self.build_err=f"{type(e).__name__}: {e}"
                    self.built=False
                continue
            if not self.built: time.sleep(0.02); continue
            if not self.running or self.budget==0:
                self.running=False; time.sleep(0.04); continue
            # EVERY NEURAL TICK is sampled through run_episode's observer hook, not once per agent
            # step. Sampling one tick in sixteen was throwing away 15/16 of the spike train, which is
            # why sparse populations (arbiter modes, relays, ladders) read as silent.
            _,mt=AG.run_episode(self.a,steps=1,log_every=10**9,render_every=8,
                                tick_hook=self._on_tick)
            # The strict V3 topology is FORAGE/EXPLORE/SLEEP (HOME is not
            # built).  Using the legacy three-mode names here shifted
            # EXPLORE activity into HOME and SLEEP into EXPLORE in the live
            # decoder.  Read the names from the composed agent so the live
            # status, trace and neural network share one protocol contract.
            mode_names=list(getattr(self.a, "_mode_names", ()))
            if self.a._metabolic_sleep and "SLEEP" not in mode_names:
                mode_names.append("SLEEP")
            m=[float(mt.get(nm,0)) for nm in mode_names]
            # ``run_episode`` has already summed every neural tick in this
            # agent-step.  Keep that exact window rather than a max/decay
            # latch: the old smoothing left a stale FORAGE/HOME label visible
            # after V3 had switched to SLEEP or EXPLORE.
            self.mode=m
            self.step+=1
            if self.budget>0:
                self.budget-=self.sub
                if self.budget<=0: self.running=False
            self.publish()

    def _on_tick(self, ag):
        """One NEURAL tick has just completed. Every tick is captured and queued -- none are merged or
        dropped -- so the page can render neural time one tick at a time."""
        self.ticks+=1
        o=np.fromiter((1.0 if u.O>0 else 0.0 for u in self.units),dtype=np.float32,count=len(self.units))
        self.act =np.maximum(self.act*0.80,o)     # fast trace, only used by the HTTP fallback
        self.rate=np.maximum(self.rate*0.985,o)   # slow trace: rate estimate for the status decoder
        self.ring_tr=self.ring_tr*(1-1/12.)+o     # 12-tick window for the heading bump
        # The bitmask is ceil(n/8) bytes for the currently built network (1766 for V1/V2,
        # 1788 for V3). At ~26 ticks/s it fits on the wire with room to spare, so every tick can be
        # sent without coalescing.
        bits=np.packbits((o>0).astype(np.uint8))
        gr=np.array([self.a.nb[n].S for n in self.graded],dtype='<f4')
        self.frames.append((self.ticks, bits.tobytes()+gr.tobytes()))
        try:
            self.trace.capture(tick=self.ticks,step=self.step,ag=ag,units=self.units)
            self.trace_error=self.trace.error
        except Exception as exc:  # the observer can never stop the embodied loop
            self.trace_error=f"{type(exc).__name__}: {exc}"
        self.frame_evt.set()
        if self.ticks%self.pub_every==0: self.publish(cams=False)

    def publish(self, force=False, cams=True):
        w=self.w; a=self.a
        img3=img1=None
        # the cameras are expensive; refresh them on the agent-step boundary, not every neural tick
        if force or (cams and self.step%2==0):
            self.R3.update_scene(w.data,camera=self.cam); img3=jpg(self.R3.render())
            self.RP.update_scene(w.data,camera=self.eye); img1=jpg(self.RP.render())
        x,y,yaw=w.pose()
        act8=np.clip(self.act*255,0,255).astype(np.uint8)
        # graded cells never fire, so a spike raster shows them as permanently dark. Ship their
        # membrane potential separately and let the page draw them from that instead.
        grad={str(n):round(float(self.a.nb[n].S),4) for n in self.graded}
        snap={"ready":True,"world":self.world_name,"step":self.step,"ticks":self.ticks,
              "running":self.running,"budget":self.budget,
              "eaten":int(w.eaten),"tox":int(w.tox_hits),"home":round(float(w.dist_home()),2),
              "pose":[round(x,2),round(y,2),round(yaw,3)],
              "arena":self.cfg["arena"],
              "barrier":getattr(w,"barrier_name",None),
              "barriers":[{k:round(float(v),3) if isinstance(v,(int,float)) else v for k,v in b.items()
                           if k in ("name","x","y","hx","hy")} for b in getattr(w,"barriers",())],
              "obstacle":dict(getattr(a,"last_obstacle_afferents",{})),
              "version":self.version_spec.manifest(),
              "food":[[round(f[0],1),round(f[1],1)] for f in w.foods],
              "toxin":[[round(t[0],1),round(t[1],1)] for t in w.toxins],
              "config":a.effective_config_manifest(),
              "trace":{"first":self.trace.range_public()["first"],
                       "last":self.trace.range_public()["last"],
                       "count":self.trace.range_public()["count"],
                       "total_captured":self.trace.range_public()["total_captured"]},
              "rate":round(self.step/max(1e-6,time.time()-self.t0),2),
              "act":base64.b64encode(act8.tobytes()).decode(),
              "grad":grad,
              "dec":self.dec.read(getattr(self,"rate",self.act),getattr(self,"mode",None),
                                  getattr(self,"ring_tr",None))}
        with self.lock:
            if img3 is not None: self.last3=img3; self.last1=img1
            snap["third"]=getattr(self,"last3",None); snap["pov"]=getattr(self,"last1",None)
            self.snap=snap

    # ---- commands from the page ---------------------------------------------------------------
    def command(self, q):
        c=q.get("c",[""])[0]
        if c=="run":
            n=int(q.get("n",["-1"])[0]); self.budget=-1 if n<0 else n; self.running=True
        elif c=="pause": self.running=False
        elif c=="step":  self.budget=self.sub; self.running=True
        elif c=="world":
            nm=q.get("w",["meadow"])[0]
            if nm in WORLDS:
                with self.cmd_lock: self.pending=nm
        elif c=="params":
            # ?c=params&p=<json>  -- merge overrides, then REBUILD so they take effect. A parameter
            # cannot be changed on a live network: weights and thresholds are baked in at build time.
            try: upd=json.loads(q.get("p",["{}"])[0])
            except Exception as e: return {"ok":False,"err":f"bad json: {e}"}
            for k2,v in upd.items():
                if k2 not in PARAM_BY_NAME: return {"ok":False,"err":f"unknown param {k2}"}
                if v is None: self.params.pop(k2,None)
                else: self.params[k2]=float(v)
            self.running=False; self.budget=0
            with self.cmd_lock: self.pending=self.world_name
            return {"ok":True,"params":self.params}
        elif c=="reset":
            self.params={}
            self.running=False; self.budget=0
            with self.cmd_lock: self.pending=self.world_name
            return {"ok":True,"params":{}}
        elif c=="rebuild":
            self.running=False; self.budget=0
            with self.cmd_lock: self.pending=self.world_name
        return {"ok":True}

    def neuron_detail(self, neuron_id: int, tick: int | None = None) -> dict:
        """Return an immutable, JSON-safe intracellular/synaptic inspection view."""
        nid=int(neuron_id)
        if not self.built or nid not in self.a.nb:
            raise KeyError(nid)
        unit=self.a.nb[nid]
        incoming=[]
        for sid, point in list(getattr(unit, "postsynaptic_points", {}).items()):
            incoming.append({
                "source": int(getattr(unit, "synapse_sources", {}).get(sid, (0, 0))[0]),
                "synapse": int(sid),
                "terminal": int(getattr(unit, "synapse_sources", {}).get(sid, (0, 0))[1]),
                "weight": float(getattr(getattr(point, "u_i", None), "info", 0.0)),
                "plast": float(getattr(getattr(point, "u_i", None), "plast", 0.0)),
                "potential": float(getattr(point, "potential", 0.0)),
                "distance": int(getattr(unit, "distances", {}).get(sid, 1)),
            })
        outgoing=[]
        for (src, terminal), targets in self.a.net.network.connection_cache.items():
            if int(src)!=nid:
                continue
            for target, sid in targets:
                if target not in self.a.nb:
                    continue
                target_unit=self.a.nb[target]
                point=target_unit.postsynaptic_points.get(sid)
                outgoing.append({
                    "source": int(src),
                    "target": int(target), "synapse": int(sid),
                    "terminal": int(terminal),
                    "weight": float(getattr(getattr(point, "u_i", None), "info", 0.0)),
                    "plast": float(getattr(getattr(point, "u_i", None), "plast", 0.0)),
                    "potential": float(getattr(point, "potential", 0.0)),
                    "distance": int(getattr(target_unit, "distances", {}).get(sid, 1)),
                })
        adapter=NeuronDisplayAdapter()
        result=adapter.adapt(nid,unit,incoming=incoming[:256],outgoing=outgoing[:256])
        # These are the live intracellular-facing buffers behind the compact display fields.  They
        # are read only and intentionally kept out of the per-tick websocket frame; a selected-cell
        # request can inspect them without paying the cost for every neuron on every tick.
        result["intracellular"]["input_buffer"] = np.asarray(getattr(unit,"input_buffer",[]),dtype=float).tolist()
        result["intracellular"]["synapse_sources"] = {
            str(sid): [int(source), int(terminal)]
            for sid,(source,terminal) in getattr(unit,"synapse_sources",{}).items()
        }
        result["presynaptic"] = {
            str(terminal): {
                "info": float(getattr(getattr(point,"u_o",None),"info",0.0)),
                "mod": np.asarray(getattr(getattr(point,"u_o",None),"mod",[]),dtype=float).tolist(),
                "retro": float(getattr(point,"u_i_retro",0.0)),
            }
            for terminal,point in getattr(unit,"presynaptic_points",{}).items()
        }
        result["limits"]={"incoming_total":len(incoming),"outgoing_total":len(outgoing),"truncated":len(incoming)>256 or len(outgoing)>256}
        # Historical state comes from the read-only trace ring.  ``tick=None`` means the newest
        # completed neural tick; a missing historical tick is reported by the HTTP route as 404.
        if getattr(self, "trace", None) is not None:
            result["trace"] = self.trace.neuron_public(nid, tick)
        return result

SIM=None
HTTP_PORT=8770
WS_PORT=8771
# Optional static evidence bundle for the compass boundary-replay page.  It
# is deliberately separate from SIM state: replay viewing must never mutate,
# decode into, or otherwise perturb the live brain.
COMPASS_REPLAY_PATH=None
class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    def _send(self,body,ctype="application/json",code=200):
        if isinstance(body,str): body=body.encode()
        self.send_response(code); self.send_header("Content-Type",ctype)
        self.send_header("Content-Length",str(len(body)))
        # The microscope host runs on the harness-lab port while the selected brain runs on its own
        # port.  This read-only local protocol is intentionally embeddable across those origins.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control","no-store"); self.end_headers()
        try: self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError): pass
    def _json(self, value, code=200):
        self._send(json.dumps(value,separators=(",",":")), "application/json; charset=utf-8", code)
    def _command_query(self, payload):
        from urllib.parse import parse_qs
        if isinstance(payload, dict):
            q={k:[str(v)] for k,v in payload.items()}
        else:
            q=payload
        return SIM.command(q)
    def do_GET(self):
        from urllib.parse import urlparse,parse_qs
        u=urlparse(self.path); p=u.path
        if p in ("/","/index.html"):
            self._send((Path(_HERE) / "brain_live.html").read_bytes(),"text/html; charset=utf-8")
        elif p=="/lab":
            from urllib.parse import quote
            lab_url=os.environ.get("AIF_LAB_URL", "http://127.0.0.1:8850/lab")
            separator="&" if "?" in lab_url else "?"
            self.send_response(302)
            self.send_header("Location", lab_url+separator+"source="+
                             quote(f"http://127.0.0.1:{HTTP_PORT}", safe=""))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
        elif p=="/api/health" or p=="/healthz":
            self._json({"ok":bool(SIM and SIM.built and not SIM.build_err),"protocol":PROTOCOL_VERSION,
                        "version":SIM.version if SIM else None,"build_error":SIM.build_err if SIM else None})
        elif p=="/api/schema":
            self._json(protocol_schema())
        elif p=="/api/session":
            msg=protocol_session(version=SIM.version_spec.manifest(),ids=SIM.ids,graded=SIM.graded,
                                 http_port=HTTP_PORT,ws_port=WS_PORT)
            msg["introspection"]={"protocol":"aif-introspection/1",
                                   "trace":SIM.trace.range_public() if SIM.trace is not None else None}
            self._json(msg)
        elif p in ("/api/introspection", "/api/introspection/"):
            if not SIM.built or SIM.trace is None:
                self._json({"error": "brain is still building", "trace": None}, 503)
            else:
                manifest=SIM.trace.manifest()
                manifest.update({"version": SIM.version_spec.manifest(), "world": SIM.world_name,
                                 "trace_error": SIM.trace_error})
                self._json(manifest)
        elif p in ("/api/trace", "/api/trace/"):
            q=parse_qs(u.query)
            if SIM.trace is None:
                self._json({"error": "brain is still building"}, 503)
                return
            def _optional_int(name):
                try: return int(q[name][0]) if q.get(name) else None
                except (TypeError,ValueError): return None
            try:
                self._json(SIM.trace.timeline(start=_optional_int("from"), end=_optional_int("to"),
                                              stride=int(q.get("stride", ["1"])[0]),
                                              limit=int(q.get("limit", ["512"])[0])))
            except (TypeError,ValueError):
                self._json({"error": "from/to/stride/limit must be integers"}, 400)
        elif p=="/api/tick" or p.startswith("/api/tick/"):
            q=parse_qs(u.query)
            if SIM.trace is None:
                self._json({"error": "brain is still building"}, 503)
                return
            raw=p.rsplit("/",1)[-1] if p.startswith("/api/tick/") else (q.get("tick", [""])[0])
            try: tick=int(raw)
            except (TypeError,ValueError):
                self._json({"error": "tick must be an integer"}, 400)
            else:
                detail=str(q.get("detail", ["neurons"])[0]).lower()
                try:
                    self._json(SIM.trace.tick_public(tick,
                                                     include_neurons=detail in ("neurons", "all"),
                                                     include_synapses=detail in ("synapses", "all")))
                except KeyError:
                    self._json({"error": "tick is outside the retained trace window",
                                "trace": SIM.trace.range_public()}, 404)
        elif p in ("/api/synapse", "/api/synapse/"):
            q=parse_qs(u.query)
            if SIM.trace is None:
                self._json({"error": "brain is still building"}, 503)
                return
            def _required_int(name):
                return int(q[name][0])
            try:
                value=SIM.trace.synapse_public(source=_required_int("source"), target=_required_int("target"),
                                               synapse=_required_int("synapse"),
                                               tick=int(q["tick"][0]) if q.get("tick") else None)
            except (KeyError,TypeError,ValueError):
                self._json({"error": "source, target, and synapse are required; tick is optional"}, 400)
            else:
                self._json(value)
        elif p in ("/compass-replay","/compass-replay.html"):
            self._send((Path(_HERE) / "compass_replay.html").read_bytes(),"text/html; charset=utf-8")
        elif p=="/compass-replay.json":
            if COMPASS_REPLAY_PATH is None:
                self._send(json.dumps({"error":"no replay bundle configured; start with --compass-replay PATH"}),code=404)
            elif not COMPASS_REPLAY_PATH.is_file():
                self._send(json.dumps({"error":f"configured replay bundle is missing: {COMPASS_REPLAY_PATH}"}),code=404)
            else:
                self._send(COMPASS_REPLAY_PATH.read_bytes())
        elif p=="/topology.json":
            self._send((Path(_HERE) / "brain_topology.json").read_bytes())
        elif p in ("/payload.json","/api/topology"):
            # the LIVE wiring payload, matching the network currently built. 204 while it is being
            # regenerated so the page keeps the diagram it already has instead of flashing empty.
            if SIM.topo is None:
                self._send(json.dumps({"pending":True,"err":SIM.topo_err}),code=200)
            else:
                self._json(SIM.topo)
        elif p in ("/state","/api/state"):
            with SIM.lock: s=SIM.snap
            self._json(s)
        elif p.startswith("/api/neuron/") and p.endswith("/history"):
            q=parse_qs(u.query)
            if SIM.trace is None:
                self._json({"error": "brain is still building"},503)
                return
            try:
                nid=int(p.rstrip("/").split("/")[-2])
                optional=lambda name: int(q[name][0]) if q.get(name) else None
                value=SIM.trace.neuron_history(nid,start=optional("from"),end=optional("to"),
                                               stride=int(q.get("stride", ["1"])[0]),
                                               limit=int(q.get("limit", ["4096"])[0]))
            except (KeyError,TypeError,ValueError):
                self._json({"error": "unknown neuron or invalid history range"},404)
            else:
                self._json(value)
        elif p=="/api/neuron/" or p.startswith("/api/neuron/"):
            try:
                nid=int(p.rsplit("/",1)[-1]); q=parse_qs(u.query)
                tick=int(q["tick"][0]) if q.get("tick") else None
                self._json(SIM.neuron_detail(nid,tick=tick))
            except (ValueError,KeyError): self._json({"error":"unknown neuron"},404)
        elif p in ("/params","/api/params"):
            self._json({
                "spec":[dict({"name":n,"kind":k,"ship":sh,"fix":fx,"lo":lo,"hi":hi,
                         "step":st,"group":g,"help":h}, **({"struct":STRUCT_GATES[n]}
                                                           if n in STRUCT_GATES else {}))
                        for (n,k,sh,fx,lo,hi,st,g,h) in PARAM_SPEC],
                "current":SIM.params,
                "built":SIM.built,
                # neuron count is the ONLY honest confirmation that a STRUCTURAL gate took effect --
                # a weight change leaves it identical, a population gate must move it.
                "neurons":(len(SIM.ids) if SIM.built else 0),
                "topo_ver":SIM.topo_ver,"topo_busy":SIM.topo_busy,"topo_err":SIM.topo_err,
                "err":SIM.build_err})
        elif p in ("/cmd","/api/command"):
            self._json(SIM.command(parse_qs(u.query)))
        else:
            self._send("not found","text/plain",404)

    def do_POST(self):
        from urllib.parse import parse_qs
        length=int(self.headers.get("Content-Length","0") or 0)
        raw=self.rfile.read(length) if length else b""
        try:
            payload=json.loads(raw.decode() or "{}") if raw else {}
        except json.JSONDecodeError:
            payload=parse_qs(raw.decode())
        if self.path in ("/api/command","/cmd"):
            self._json(self._command_query(payload)); return
        if self.path=="/api/params":
            payload=dict(payload); payload.setdefault("c","params")
            if isinstance(payload.get("p"),dict): payload["p"]=json.dumps(payload["p"])
            self._json(self._command_query(payload)); return
        self._json({"error":"not found"},404)

def _hash_state(ag):
    import hashlib
    h=hashlib.sha256()
    for nid in sorted(ag.net.network.neurons.keys()):
        u=ag.net.network.neurons[nid]
        h.update(f"{nid}:{u.S:.10f}:{u.O:.6f}:{u.t_last_fire}".encode())
        for sid in sorted(u.postsynaptic_points.keys()):
            h.update(f"{sid}:{u.postsynaptic_points[sid].u_i.info:.10f}".encode())
    return h.hexdigest()[:16]

def _verify(steps=120):
    """Run the same seed twice — decoding every step, and not decoding at all — and compare the
    full network state. Identical hashes are the proof that the decoder is outside the loop."""
    res={}
    for label,decoding in (("with decoder",True),("without decoder",False)):
        np.random.seed(11)
        a=AG.AIFAgent3D(seed=11)
        a.world=AG.w3.World3D(n_food=9,n_tox=3,arena=11.0,seed=11)
        a.img=a.world.retina(); a.birth()
        ids=sorted(a.net.network.neurons.keys()); units=[a.nb[i] for i in ids]
        dec=Decoder().bind(a,ids); trace=np.zeros(len(ids),dtype=np.float32)
        for _ in range(steps):
            AG.run_episode(a,steps=1,log_every=10**9,render_every=8)
            o=np.fromiter((1.0 if u.O>0 else 0.0 for u in units),dtype=np.float32,count=len(units))
            trace=np.maximum(trace*0.55,o)
            if decoding: dec.read(trace)
        res[label]=(_hash_state(a),a.world.eaten,a.world.tox_hits,round(float(a.world.dist_home()),3))
        print(f"  {label:16s} state={res[label][0]}  ate={res[label][1]}  "
              f"toxin={res[label][2]}  d_home={res[label][3]}",flush=True)
    ok=res["with decoder"]==res["without decoder"]
    print(("PASS - decoder is purely external, network state identical" if ok else
           "FAIL - decoding changed the network"),flush=True)
    return ok



# ================================ WEBSOCKET: EVERY TICK =========================================
# One binary frame per NEURAL tick, in order, nothing merged and nothing skipped. Each frame is
#   uint32 tick | ceil(n/8)-byte spike bitmask | 4 x float32 graded muscle S
# and the heavier JSON (decoded state, cameras, telemetry) goes out separately on the agent-step
# boundary as a text frame. A client that stalls builds a backlog rather than losing ticks; if it
# falls more than the ring buffer behind it is dropped, because at that point it is not watching.
import struct
WS_CLIENTS=set()

async def _ws_pump(ws):
    """Drain the sim's frame queue to one client, oldest first, never skipping.
    A client joins at the LIVE EDGE rather than replaying the ring buffer: history it never saw is
    not 'skipped', and replaying 4096 stale ticks would leave it minutes behind the brain."""
    cursor=SIM.ticks
    while True:
        if not SIM.frames:
            SIM.frame_evt.clear()
            await asyncio.sleep(0.004); continue
        # take everything at or after our cursor
        pending=[f for f in list(SIM.frames) if f[0]>cursor]
        if not pending:
            await asyncio.sleep(0.004); continue
        for tick,payload in pending:
            try:
                await ws.send(struct.pack("<I",tick)+payload)
            except Exception:
                return
            cursor=tick
        await asyncio.sleep(0)

async def _ws_state(ws):
    """The heavy snapshot: decoded status, telemetry and camera JPEGs, at agent-step rate."""
    last=-1
    while True:
        with SIM.lock: snap=SIM.snap
        if snap.get("ready") and snap.get("step")!=last:
            last=snap.get("step")
            try:
                await ws.send(json.dumps({k:v for k,v in snap.items() if k!="act"}))
            except Exception:
                return
        await asyncio.sleep(0.05)

async def _ws_handler(ws):
    WS_CLIENTS.add(ws)
    try:
        hello_msg=protocol_hello(SIM.version_spec.manifest(), SIM.ids, SIM.graded)
        hello_msg["introspection"]={"protocol":"aif-introspection/1",
                                     "trace":SIM.trace.range_public() if SIM.trace is not None else None}
        await ws.send(json.dumps(hello_msg))
        pump=asyncio.create_task(_ws_pump(ws)); state=asyncio.create_task(_ws_state(ws))
        recv=asyncio.create_task(_ws_recv(ws))
        done,pend=await asyncio.wait([pump,state,recv],return_when=asyncio.FIRST_COMPLETED)
        for t in pend: t.cancel()
    except Exception: pass
    finally: WS_CLIENTS.discard(ws)

async def _ws_recv(ws):
    from urllib.parse import parse_qs
    try:
        async for msg in ws:
            try: SIM.command(parse_qs(str(msg)))
            except Exception: pass
    except Exception:
        # A browser tab closing is normal; do not leave an un-retrieved task traceback in the live
        # process log (the reader is an observer, not part of the simulation worker).
        return

async def ws_main(port):
    from websockets.asyncio.server import serve
    async with serve(_ws_handler,"127.0.0.1",port,max_queue=None):
        await asyncio.Future()

if __name__=="__main__":
    if "--verify" in sys.argv:
        sys.exit(0 if _verify(int(sys.argv[sys.argv.index("--verify")+1])
                              if len(sys.argv)>sys.argv.index("--verify")+1
                              and sys.argv[sys.argv.index("--verify")+1].isdigit() else 120) else 1)
    ap=argparse.ArgumentParser()
    ap.add_argument("--port",type=int,default=8770)
    ap.add_argument("--version",default="v1",choices=["v1","v2","v3","v4"],
                    help="PAULA composition to run (V1 reactive, V2 memory, V3 interoceptive, V4 tactile detour)")
    ap.add_argument("--world",default="meadow",choices=list(WORLDS))
    ap.add_argument("--ticks",type=int,default=0,help="start running immediately for N ticks; -1 = forever")
    ap.add_argument("--compass-replay",type=Path,
                    help="per-tick compass replay bundle; serves it at /compass-replay")
    A=ap.parse_args()
    if not os.path.exists("brain_live.html"):
        sys.exit("brain_live.html missing -- run: python build_brain_page.py --live")
    # This module changes to its own directory during import so its bundled
    # assets work.  Resolve a user-supplied relative evidence path against the
    # shell directory instead, which is the least surprising CLI contract.
    COMPASS_REPLAY_PATH=((A.compass_replay if A.compass_replay.is_absolute() else _LAUNCH_CWD / A.compass_replay).resolve()
                         if A.compass_replay else None)
    if COMPASS_REPLAY_PATH is not None and not COMPASS_REPLAY_PATH.is_file():
        sys.exit(f"--compass-replay does not exist: {COMPASS_REPLAY_PATH}")
    HTTP_PORT=A.port; WS_PORT=A.port+1
    SIM=Sim(A.world,version=A.version)
    # bounded wait: a failed build must not hang the process forever (LAB_RULES rule 10)
    _t0=time.time()
    while not SIM.built:
        if SIM.build_err: sys.exit(f"initial build failed: {SIM.build_err}")
        if time.time()-_t0 > 600: sys.exit("initial build timed out after 600s")
        time.sleep(0.1)
    if A.ticks: SIM.budget=-1 if A.ticks<0 else A.ticks; SIM.running=True
    print(f"live brain on http://localhost:{A.port}/  version={A.version} world={A.world} "
          f"{len(SIM.ids)} neurons",flush=True)
    if COMPASS_REPLAY_PATH is not None:
        print(f"compass replay on http://localhost:{A.port}/compass-replay  bundle={COMPASS_REPLAY_PATH}",flush=True)
    if importlib.util.find_spec("websockets") is not None:
        print(f"live WebSocket on :{A.port+1}", flush=True)
        threading.Thread(target=lambda: asyncio.run(ws_main(A.port+1)),daemon=True).start()
    else:
        # The static replay page is fully functional over HTTP.  Avoid a
        # background-thread traceback in minimal environments where the
        # optional live-streaming dependency was not installed.
        print("live WebSocket unavailable (install the web extra); HTTP/replay remains available", flush=True)
    ThreadingHTTPServer(("127.0.0.1",A.port),H).serve_forever()

"""EXTERNAL DECODER — neural state translated into human-readable states and actions.

This is an INSTRUMENT, not part of the brain. It only ever READS:
  * a leaky spike-rate trace supplied by the caller (one value per neuron, 0..1), and
  * the membrane potential S of the graded cells, which never spike.
It never calls set_external_input, never touches a weight, never steps the network. Deleting this
file changes no behaviour — `python decode.py --verify` proves that by running the agent twice, with
and without decoding every step, and comparing a hash of the full network state.

Every number below is a readout of activity that is already there. Where a decode involves a
convention (which way a bump angle runs, the sign of the home vector) it is named in the label rather
than hidden, so nothing here can be mistaken for the agent "knowing" something it does not.
"""
import sys, importlib.util
import numpy as np
sys.path.insert(0,'/Users/arterialist/Projects/agi-research/neuron-model')

def _pop_vector(rates, angles):
    """circular mean of a population laid out around a circle -> (angle_deg, resultant_length)"""
    r=np.asarray(rates,float); a=np.asarray(angles,float)
    if r.sum()<=0: return 0.0, 0.0
    x=float((r*np.cos(a)).sum()); y=float((r*np.sin(a)).sum())
    return float(np.degrees(np.arctan2(y,x))%360.0), float(np.hypot(x,y)/max(r.sum(),1e-9))

class Decoder:
    def __init__(self, AG):
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

    def read(self, tr, mode_hint=None):
        """mode_hint: the arbiter's per-mode activity accumulated ACROSS the 16-tick sub-loop, which
        run_episode already computes and returns. The rest of this decoder samples the last tick of the
        step, which is fine for ladders, sensors and graded muscles but silently misses the arbiter's
        sparse firing -- reading one tick in sixteen showed every mode at 0%."""
        AG,vc,cc,nv,ar=self.AG,self.vc,self.cc,self.nv,self.ar
        out=[]
        def item(sec,label,text,level=None,note=None):
            out.append({"s":sec,"k":label,"v":text,
                        "b":None if level is None else max(0.0,min(1.0,float(level))),
                        "n":note})

        # ---------- INTEROCEPTION AND DRIVE ----------
        hun=self._r(tr,ar.HUNGER)
        item("drive","hunger",
             ("sated" if hun<.12 else "peckish" if hun<.35 else "hungry" if hun<.65 else "starving")
             +f"  {hun*100:.0f}%", hun, "interoceptive ladder, 12 cells; eating drains it")
        # Versioned compositions own their mode topology.  The historical
        # decoder used the three full-brain names unconditionally, which
        # relabelled strict V3's ``EXPLORE`` as ``HOME`` and ``SLEEP`` as
        # ``EXPLORE``.  Fall back to the legacy groups only for an unprofiled
        # full brain so this standalone instrument and the live decoder share
        # the same read-only protocol contract.
        mode_names=list(getattr(self.ag, "_mode_names", ar.MODES)) if getattr(self, "ag", None) is not None else list(ar.MODES)
        if mode_names and getattr(self, "ag", None) is not None and getattr(self.ag, "_metabolic_sleep", False):
            if "SLEEP" not in mode_names:
                mode_names.append("SLEEP")
        if mode_hint is None:
            groups=list(getattr(self.ag, "_mode_groups", ar.MODE)) if getattr(self, "ag", None) is not None else list(ar.MODE)
            if "SLEEP" in mode_names and len(groups) == len(mode_names)-1:
                groups.append(ar.SLEEP_MODE)
            mrates=[self._r(tr,group) for group in groups]
        else:
            mrates=list(mode_hint)
            mrates.extend([0.0] * max(0, len(mode_names)-len(mrates)))
            mrates=mrates[:len(mode_names)]
        win=int(np.argmax(mrates)); tot=sum(mrates)
        marg=(mrates[win]-sorted(mrates)[-2])/max(mrates[win],1e-9) if tot>0 else 0.0
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
        ring=self._each(tr,cc.RING)
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
        item("motor","left muscles (graded S)",
             f"protract {s[AG.MLp]:+.3f}  retract {s[AG.MLr]:+.3f}", min(1.0,left/2.0),
             "membrane potential read straight out as actuator force; these cells never spike")
        item("motor","right muscles (graded S)",
             f"protract {s[AG.MRp]:+.3f}  retract {s[AG.MRr]:+.3f}", min(1.0,right/2.0))
        item("motor","thrust", f"{(left+right)/2:.3f}", min(1.0,(left+right)/4.0))
        turn=left-right
        item("motor","turn",
             ("left" if turn>0.02 else "right" if turn<-0.02 else "straight")+f"  {turn:+.3f}",
             min(1.0,abs(turn)), "steering is inhibition of one side's muscles, not a yaw command")
        rl={m:sum(1 for n in AG.RLY[m] if self._r(tr,[n])>0.05) for m in AG.RLY}
        item("motor","relay gating",
             "  ".join(f"{'LR'[m in (AG.MRp,AG.MRr)]}{'pr'[m in (AG.MLr,AG.MRr)]}:{v}/{AG.NRLY}"
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

# ---------------------------------------------------------------------------------------------
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
    spec=importlib.util.spec_from_file_location('ag','aif_agent3d.py')
    AG=importlib.util.module_from_spec(spec); spec.loader.exec_module(AG)
    res={}
    for label,decoding in (("with decoder",True),("without decoder",False)):
        np.random.seed(11)
        a=AG.AIFAgent3D(seed=11)
        a.world=AG.w3.World3D(n_food=9,n_tox=3,arena=11.0,seed=11)
        a.img=a.world.retina(); a.birth()
        ids=sorted(a.net.network.neurons.keys()); units=[a.nb[i] for i in ids]
        dec=Decoder(AG).bind(a,ids); trace=np.zeros(len(ids),dtype=np.float32)
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

if __name__=="__main__":
    sys.exit(0 if _verify(int(sys.argv[2]) if len(sys.argv)>2 else 120) else 1)

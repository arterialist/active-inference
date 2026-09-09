"""Active inference ENTIRELY on PAULA spiking neurons (ckit) — the T-maze controller.

Brain (all PAULA neurons, no numpy computation in the loop):
  BL,BR  belief accumulators   -- integrate cue/reward evidence (integrate-to-bound = Bayesian SPRT),
                                  mutual inhibition (normalisation) + slow leak (memory across steps)
  U      uncertainty / epistemic value  -- tonic drive, inhibited by BL+BR -> high when belief unresolved
  DL,DR  confidence / pragmatic value   -- DL=BL gated against BR (expected preference for each arm)
  Acue,Aleft,Aright,Acenter  action WTA -- driven by U (epistemic) and DL/DR (pragmatic); winner = action

The sensorimotor loop (position -> observation -> evidence current; action-spikes -> movement) is the
body/world interface, exactly like the embodied organism. Belief-update, uncertainty and action-selection
all happen in the neurons.
"""
import sys, pathlib, numpy as np
sys.path.insert(0, "/Users/arterialist/Projects/agi-research/neuron-model")
from paula_agent import ckit as k

# neuron ids
BL, BR = 10, 11
U = 20
DL, DR = 30, 31
ACUE, ALEFT, ARIGHT, ACENTER = 40, 41, 42, 43
ACT = [ACUE, ALEFT, ARIGHT, ACENTER]
def T(nid): return 900 + nid   # unique terminal id per neuron


def build_brain(w_self=6.0, w_inh=-12.0, w_ev=4.0, d_self=3, lam_b=10, r_b=0.5,
                w_tonic=1.4, w_uinh=-9.0, w_bd=3.2, w_dinh=-9.0,
                w_epi=4.0, w_prag=4.5, w_base=0.5, w_wta=-5.0):
    ne = []; sy = []; conns = []; exts = []

    # ---- belief accumulators BL, BR : evidence integrators, mutual inhibition, self-excitation memory ----
    for nid, other in [(BL, BR), (BR, BL)]:
        ne.append(k.neuron(nid, r=r_b, c=2, lam=lam_b))     # rate-codes accumulated evidence
        sy.append(k.syn(nid, 0, w_ev, 1))                   # syn0: evidence (external)
        sy.append(k.syn(nid, 1, w_inh, 1))                  # syn1: inhibition from the other
        sy.append(k.syn(nid, 2, w_self, d_self))            # syn2: self-excitation (delay clears refractory)
        sy.append(k.term(nid, T(nid)))
        exts.append(k.ext(nid, 0))
    conns += [k.conn(BR, BL, 1, T(BR)), k.conn(BL, BR, 1, T(BL))]
    conns += [k.conn(BL, BL, 2, T(BL)), k.conn(BR, BR, 2, T(BR))]

    # ---- uncertainty U : tonic, inhibited by both beliefs -> high only when belief is unresolved ----
    ne.append(k.neuron(U, r=0.6, c=2, lam=5))
    sy.append(k.syn(U, 0, w_tonic, 1)); exts.append(k.ext(U, 0))   # syn0: tonic epistemic baseline
    sy.append(k.syn(U, 1, w_uinh, 1)); sy.append(k.syn(U, 2, w_uinh, 1))
    sy.append(k.term(U, T(U)))
    conns += [k.conn(BL, U, 1, T(BL)), k.conn(BR, U, 2, T(BR))]

    # ---- confidence DL, DR : belief for one arm, suppressed by the other (pragmatic value) ----
    for nid, pos, neg in [(DL, BL, BR), (DR, BR, BL)]:
        ne.append(k.neuron(nid, r=0.6, c=2, lam=5))
        sy.append(k.syn(nid, 0, w_bd, 1)); sy.append(k.syn(nid, 1, w_dinh, 1))
        sy.append(k.term(nid, T(nid)))
        conns += [k.conn(pos, nid, 0, T(pos)), k.conn(neg, nid, 1, T(neg))]

    # ---- action WTA : Acue<-U (epistemic), Aleft<-DL, Aright<-DR (pragmatic), Acenter<-tonic ----
    drive_src = {ACUE: (U, w_epi), ALEFT: (DL, w_prag), ARIGHT: (DR, w_prag), ACENTER: (None, w_base)}
    for a in ACT:
        ne.append(k.neuron(a, r=0.6, c=2, lam=5))
        src, w = drive_src[a]
        sy.append(k.syn(a, 0, w, 1))                        # syn0: EFE drive
        if src is None: exts.append(k.ext(a, 0))            # Acenter: tonic baseline
        else: conns.append(k.conn(src, a, 0, T(src)))
        for j, o in enumerate([x for x in ACT if x != a]):
            sy.append(k.syn(a, 1 + j, w_wta, 1))            # mutual inhibition
        sy.append(k.term(a, T(a)))
    for a in ACT:
        for j, o in enumerate([x for x in ACT if x != a]):
            conns.append(k.conn(o, a, 1 + j, T(o)))

    return k.build(ne, sy, conns, exts)


class Brain:
    """Persistent PAULA brain. Feed observation-evidence each tick; read the action WTA winner."""
    def __init__(self, **kw):
        self.path = build_brain(**kw); self.net, self.core = k.load(self.path)
        self.nb = {n: u for n, u in self.net.network.neurons.items()}
    def reset(self):
        self.net.reset_simulation(); self.core.state.current_tick = 0; self.net.current_tick = 0
    def run(self, ticks, evidence=(0.0, 0.0), tonicU=1.0, base=1.0, probe=None, tick_trace=None):
        """Run `ticks` ticks with constant evidence current (ev_left, ev_right) to BL/BR; return
        spike counts for probed neurons.  When ``tick_trace`` is supplied, append the raw neural
        driver/output record once per PAULA tick; it is instrumentation only and never feeds back."""
        probe = probe or (ACT + [BL, BR, U, DL, DR])
        cnt = {i: 0 for i in probe}
        for tick in range(ticks):
            self.net.set_external_input(BL, 0, evidence[0])
            self.net.set_external_input(BR, 0, evidence[1])
            self.net.set_external_input(U, 0, tonicU)
            self.net.set_external_input(ACENTER, 0, base)
            self.core.do_tick()
            for i in probe:
                if self.nb[i].O > 0: cnt[i] += 1
            if tick_trace is not None:
                tick_trace.append(dict(
                    neural_tick=tick,
                    evidence_left=float(evidence[0]), evidence_right=float(evidence[1]),
                    tonic_uncertainty=float(tonicU), center_baseline=float(base),
                    spikes={str(i): int(self.nb[i].O > 0) for i in probe},
                ))
        return cnt


# ---- the T-maze world + sensorimotor loop (body/world interface; the BRAIN is the PAULA net) ----
CENTER, CUE, LEFT, RIGHT = 0, 1, 2, 3
RL, RR = 0, 1   # hidden context: reward in Left / Right arm
EV = 3.0        # evidence current magnitude

def run_episode(ctx, brain, W=18, maxsteps=4, log=None, tick_log=None):
    """One episode. At each location the brain runs W ticks with the observation's evidence current;
    the winning action neuron moves the agent. Belief persists across steps (not reset mid-episode)."""
    loc = CENTER; brain.reset(); visited_cue = False; reward = None
    for step in range(maxsteps):
        if loc == CUE:            ev = (EV, 0.0) if ctx == RL else (0.0, EV)   # cue reveals context
        else:                     ev = (0.0, 0.0)
        step_ticks = [] if tick_log is not None else None
        cnt = brain.run(W, evidence=ev, tick_trace=step_ticks)
        acts = {ACUE: cnt[ACUE], ALEFT: cnt[ALEFT], ARIGHT: cnt[ARIGHT], ACENTER: cnt[ACENTER]}
        win = max(acts, key=acts.get)
        # Absence of a neural action is physical immobility, not an instruction for Python to select
        # ACUE because it happens to occur first in this dict.  This makes a pathway ablation a real
        # causal control while keeping the world interface read-only with respect to choice.
        if acts[win] == 0:
            win = ACENTER
        if tick_log is not None:
            for row in step_ticks:
                row.update(
                    step=step, location=loc, context=ctx,
                    selected_action=win,
                    action_counts={str(action): count for action, count in acts.items()},
                )
                tick_log.append(row)
        if log is not None:
            b2 = 'left' if cnt[BL] > cnt[BR] else ('right' if cnt[BR] > cnt[BL] else 'unsure')
            log.append(dict(step=step, loc=loc, belief=b2, BL=cnt[BL], BR=cnt[BR], U=cnt[U],
                            DL=cnt[DL], DR=cnt[DR], acts=dict(cue=cnt[ACUE], left=cnt[ALEFT],
                            right=cnt[ARIGHT], center=cnt[ACENTER]), chose={ACUE:'cue',ALEFT:'left',ARIGHT:'right',ACENTER:'stay'}[win]))
        if loc in (LEFT, RIGHT): break
        if   win == ACUE:   loc = CUE; visited_cue = True
        elif win == ALEFT:  loc = LEFT
        elif win == ARIGHT: loc = RIGHT
        if loc in (LEFT, RIGHT):
            reward = (ctx == RL and loc == LEFT) or (ctx == RR and loc == RIGHT)
    return dict(visited_cue=visited_cue, reward=bool(reward))


def summarise(tag, n=20, **kw):
    b = Brain(**kw); cue = rew = 0
    for i in range(n):
        r = run_episode(i % 2, b); cue += r['visited_cue']; rew += r['reward']
    print(f"  {tag:34s} visits-cue {100*cue//n:3d}%   reaches-reward {100*rew//n:3d}%")


if __name__ == "__main__":
    b = Brain()
    print("STAGE 1 — belief accumulation & persistence (drive cue=LEFT evidence, then remove it):")
    b.reset()
    c1 = b.run(20, evidence=(3.0, 0.0))                       # 20 ticks of left-evidence
    c2 = b.run(20, evidence=(0.0, 0.0))                       # 20 ticks of NO evidence (does belief persist?)
    print(f"  during evidence : BL={c1[BL]:2d} BR={c1[BR]:2d}  U={c1[U]:2d} DL={c1[DL]:2d} DR={c1[DR]:2d}  "
          f"action spikes cue/left/right/ctr = {c1[ACUE]}/{c1[ALEFT]}/{c1[ARIGHT]}/{c1[ACENTER]}")
    print(f"  after (no ev)   : BL={c2[BL]:2d} BR={c2[BR]:2d}  U={c2[U]:2d} DL={c2[DL]:2d} DR={c2[DR]:2d}  "
          f"action spikes cue/left/right/ctr = {c2[ACUE]}/{c2[ALEFT]}/{c2[ARIGHT]}/{c2[ACENTER]}")
    print("STAGE 2 — uncertain belief (no evidence from reset): expect U high -> Acue wins:")
    b.reset()
    c0 = b.run(24, evidence=(0.0, 0.0))
    print(f"  uncertain       : BL={c0[BL]:2d} BR={c0[BR]:2d}  U={c0[U]:2d} DL={c0[DL]:2d} DR={c0[DR]:2d}  "
          f"action cue/left/right/ctr = {c0[ACUE]}/{c0[ALEFT]}/{c0[ARIGHT]}/{c0[ACENTER]}")
    print("-" * 66)
    print("FULL EPISODES (sensorimotor loop) — one per context, with per-step trace:")
    for ctx in (RL, RR):
        log = []; res = run_episode(ctx, Brain(), log=log)
        print(f" hidden reward = {'LEFT' if ctx==RL else 'RIGHT'}:")
        for e in log:
            print(f"   step{e['step']} at {['CENTER','CUE','LEFT','RIGHT'][e['loc']]:6s} "
                  f"belief={e['belief']:6s} (U={e['U']} DL={e['DL']} DR={e['DR']}) -> chose {e['chose']}")
        print(f"   => visited_cue={res['visited_cue']} reward={res['reward']}")
    print("-" * 66)
    print("Behaviour over 20 episodes (both contexts):")
    summarise("FULL PAULA active inference", n=20)
    summarise("ABLATE epistemic (w_epi=0)", n=20, w_epi=0.0)
    print("@@@PAULA-AIF STAGE DONE@@@")

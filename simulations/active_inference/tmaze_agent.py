"""TRUE ACTIVE INFERENCE — epistemic-foraging T-maze (sophisticated inference / EFE tree search).

The canonical demonstration that separates active inference from reward-maximisation:
  - A reward is hidden in the LEFT or RIGHT arm (a hidden CONTEXT the agent cannot observe).
  - A CUE location reveals which arm is rewarded (information, but no reward itself).
  - A reward-greedy agent that cannot reach the cue's value must GUESS an arm -> ~50%.
  - A true active-inference planner walks to the CUE to RESOLVE its uncertainty (epistemic value),
    then walks to the now-certain reward (pragmatic value) -> ~100%.

The agent has an explicit generative model (A,B,C,D), infers the hidden context by Bayesian
belief-updating (variational free-energy minimisation; exact for this discrete model), and PLANS by
minimising EXPECTED FREE ENERGY over a tree of future action/observation branches (sophisticated
inference). For each candidate action:

    G(a) = - (pragmatic value) - (epistemic value)
    pragmatic value = E_Q(o|a)[ log C(o) ]                 # expected preference satisfaction
    epistemic value = I(states ; observations | a)          # expected information gain (salience)

and the value of an action includes the Bellman backup of acting optimally after each anticipated
observation, so information is valued instrumentally (it enables better future action) as well as
intrinsically (the epistemic term).

Run standalone:  python -m simulations.active_inference.tmaze_agent
"""
import numpy as np

# ----- state space: joint (location, context) -----
C_, CUE, LEFT, RIGHT = 0, 1, 2, 3       # locations
RL, RR = 0, 1                           # hidden context: reward is in Left / Right arm
LOCN = ["CENTER", "CUE", "LEFT", "RIGHT"]
NS = 8
def sid(loc, ctx): return loc * 2 + ctx
def s_loc(s): return s // 2
def s_ctx(s): return s % 2

# observation modalities: location[4], reward[3], cue[3]
NEUTRAL, REWARD, PUNISH = 0, 1, 2
NOCUE, CUEL, CUER = 0, 1, 2
ACTIONS = [C_, CUE, LEFT, RIGHT]        # action == target location


def build_model(pref=4.0):
    """Return the generative model dict with A (likelihoods), B (transitions), C (log-prefs), D (prior)."""
    A_loc = np.zeros((4, NS)); A_rew = np.zeros((3, NS)); A_cue = np.zeros((3, NS))
    for s in range(NS):
        loc, ctx = s_loc(s), s_ctx(s)
        A_loc[loc, s] = 1.0
        if loc in (C_, CUE):   A_rew[NEUTRAL, s] = 1.0
        elif loc == LEFT:      A_rew[REWARD if ctx == RL else PUNISH, s] = 1.0
        elif loc == RIGHT:     A_rew[REWARD if ctx == RR else PUNISH, s] = 1.0
        A_cue[(CUEL if ctx == RL else CUER) if loc == CUE else NOCUE, s] = 1.0
    A = [A_loc, A_rew, A_cue]
    B = []
    for a in ACTIONS:
        Ba = np.zeros((NS, NS))
        for s in range(NS):
            loc, ctx = s_loc(s), s_ctx(s)
            nloc = loc if loc in (LEFT, RIGHT) else a       # arms are absorbing (a commitment)
            Ba[sid(nloc, ctx), s] = 1.0
        B.append(Ba)
    C = [np.zeros(4), np.array([0.0, pref, -pref]), np.zeros(3)]
    D = np.zeros(NS); D[sid(C_, RL)] = 0.5; D[sid(C_, RR)] = 0.5   # start CENTER, context unknown
    return dict(A=A, B=B, C=C, D=D)


def _norm(v):
    v = np.clip(v, 1e-16, None); return v / v.sum()


def infer_state(belief, a, obs, M):
    """Bayesian posterior after taking action a and observing obs=(o_loc,o_rew,o_cue).
    This is the exact minimiser of variational free energy for a fully-factorised discrete model."""
    pred = M['B'][a] @ belief
    lik = np.ones(NS)
    for m, o in enumerate(obs):
        lik = lik * M['A'][m][o, :]
    return _norm(lik * pred)


def efe_terms(belief, a, M):
    """(pragmatic value, epistemic value) of taking action a from belief."""
    pred = M['B'][a] @ belief
    prag = 0.0; epi = 0.0
    for m in range(3):
        Am = M['A'][m]; qo = _norm(Am @ pred)
        prag += float(qo @ M['C'][m])                       # expected log-preference
        Ho = -float(np.sum(qo * np.log(np.clip(qo, 1e-16, None))))
        Hcond = 0.0
        for s in range(NS):
            if pred[s] < 1e-12: continue
            col = np.clip(Am[:, s], 1e-16, None)
            Hcond += pred[s] * (-float(np.sum(col * np.log(col))))
        epi += (Ho - Hcond)                                 # mutual information I(s;o_m)
    return prag, epi


def _expected_free_energy(belief, a, M, w_epi, w_prag):
    prag, epi = efe_terms(belief, a, M)
    return -(w_prag * prag + w_epi * epi)


def _joint_obs_dist(belief, a, M):
    pred = M['B'][a] @ belief; out = []
    for oL in range(4):
        for oR in range(3):
            for oC in range(3):
                p = float((M['A'][0][oL, :] * M['A'][1][oR, :] * M['A'][2][oC, :]) @ pred)
                if p > 1e-9: out.append(((oL, oR, oC), p))
    return out


def plan(belief, M, depth, w_epi=1.0, w_prag=1.0, return_all=False):
    """Sophisticated inference: return (best_action, G) minimising expected free energy over a tree
    that branches on ANTICIPATED observations (genuine multi-step planning with a Bellman backup).
    With return_all=True also return {action: total planned G} for every first action."""
    Gs = {}
    for a in ACTIONS:
        g = _expected_free_energy(belief, a, M, w_epi, w_prag)
        if depth > 1:
            for obs, p in _joint_obs_dist(belief, a, M):
                _, g2 = plan(infer_state(belief, a, obs, M), M, depth - 1, w_epi, w_prag)
                g += p * g2
        Gs[a] = g
    best_a = min(Gs, key=Gs.get)
    if return_all:
        return best_a, Gs[best_a], Gs
    return best_a, Gs[best_a]


class Env:
    """The real T-maze. Hidden context fixed per episode; emits observations of the true state."""
    def __init__(self, ctx, M): self.loc = C_; self.ctx = ctx; self.M = M
    def observe(self):
        s = sid(self.loc, self.ctx)
        return tuple(int(np.argmax(self.M['A'][m][:, s])) for m in range(3))
    def act(self, a):
        if self.loc not in (LEFT, RIGHT): self.loc = a
        return self.observe()


def run_episode(ctx, M, depth=3, steps=3, w_epi=1.0, w_prag=1.0, policy="aif", rng=None, log=None):
    """Run one episode. policy='aif' uses the planner; policy='guess' picks a random arm (the best a
    reward-seeker can do without reaching the cue's value). If `log` is a list, append a per-step trace."""
    if rng is None: rng = np.random.RandomState()
    env = Env(ctx, M); belief = infer_state(M['D'].copy(), C_, env.observe(), M)
    visited_cue = False; got = NEUTRAL
    if log is not None:
        b2 = belief.reshape(4, 2).sum(0)
        log.append(dict(t=0, loc=env.loc, obs=env.observe(), belief=[float(b2[0]), float(b2[1])], efe=None))
    for t in range(steps):
        if policy == "guess":
            a = rng.choice([LEFT, RIGHT])
            efe = None
        elif log is not None:
            a, _, Gs = plan(belief, M, depth, w_epi, w_prag, return_all=True)
            efe = {}
            for aa in ACTIONS:
                prag, ep = efe_terms(belief, aa, M)
                efe[int(aa)] = [round(prag, 3), round(ep, 3), round(-float(Gs[aa]), 3)]  # [prag, epi, planned value = -G]
        else:
            a, _ = plan(belief, M, depth, w_epi, w_prag)
            efe = None
        obs = env.act(a); belief = infer_state(belief, a, obs, M)
        if env.loc == CUE: visited_cue = True
        if obs[1] != NEUTRAL: got = obs[1]
        if log is not None:
            b2 = belief.reshape(4, 2).sum(0)
            log.append(dict(t=t + 1, loc=env.loc, action=int(a), obs=[int(o) for o in obs],
                            belief=[float(b2[0]), float(b2[1])], efe=efe))
        if env.loc in (LEFT, RIGHT): break
    return dict(visited_cue=visited_cue, reward=(got == REWARD), punished=(got == PUNISH))


def summarise(M, n=40, **kw):
    rng = np.random.RandomState(0); cue = rew = 0
    for i in range(n):
        r = run_episode(i % 2, M, rng=rng, **kw)
        cue += r['visited_cue']; rew += r['reward']
    return 100 * cue // n, 100 * rew // n


if __name__ == "__main__":
    M = build_model()
    print("=" * 72)
    print("FULL ACTIVE INFERENCE agent — one episode per context (belief = P(reward in LEFT)):")
    for ctx in (RL, RR):
        print(f" hidden context = reward in {'LEFT' if ctx == RL else 'RIGHT'}:")
        log = []; run_episode(ctx, M, log=log)
        for e in log:
            print(f"   t{e['t']}: at {LOCN[e['loc']]:6s}  belief[reward-left]={e['belief'][0]:.2f}")
    print("-" * 72)
    print("Behaviour over 40 episodes (both contexts):")
    for tag, kw in [("FULL active inference (plan+epistemic)", dict(depth=3, w_epi=1, w_prag=1)),
                    ("myopic greedy (no plan, no epistemic)", dict(depth=1, w_epi=0, w_prag=1)),
                    ("greedy PLANNER (depth 3, no epistemic)", dict(depth=3, w_epi=0, w_prag=1)),
                    ("pure curiosity (epistemic only)",       dict(depth=3, w_epi=1, w_prag=0)),
                    ("reward-greedy GUESSER (random arm)",     dict(policy="guess"))]:
        c, r = summarise(M, **kw)
        print(f"  {tag:40s} visits-cue {c:3d}%   reaches-reward {r:3d}%")
    print("=" * 72)
    print("@@@AIF DONE@@@")

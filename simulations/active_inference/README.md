# PAULA embodied agents + active inference

Two lines of work in one place:
1. **Fully-neural embodied agents** — spiking/graded PAULA networks that sense, decide, and move an
   articulated MuJoCo body. Behaviour lives in the wiring, not Python. Built with the `ckit` construction
   kit (`paula_agent/ckit.py` in the `neuron-model` repo); **`neuron.py` is never edited**. The only
   non-neural steps are physics transducers — odour-field-value → neuron current (chemoreceptor) and
   graded-muscle-membrane-S → actuator force (NMJ) — plus a one-time CPG "birth" seed.
2. **Active inference** — a numpy planning agent with a world model (T-maze), and a spiking version.

For a versioned hand-off that distinguishes current reruns, historical measurements, retractions,
recovered artifacts, live-brain operation, and the next falsifiable experiments, start with
[`RESEARCH_RESUMPTION.md`](RESEARCH_RESUMPTION.md).  The longer architecture and research-log files
remain the primary chronological laboratory records.

Run any embodied agent with:
```
cd <this dir>
PYTHONPATH=/Users/arterialist/Projects/agi-research/neuron-model \
  /Users/arterialist/Projects/agi-research/neuron-model/.venv/bin/python <file>.py
```

## Stable working agents (the reusable base — keep these working)

| file | what it is | status |
|------|-----------|--------|
| `neural_agent.py` | Fully-neural 360° navigator. Odour sensor populations → opponent tropotaxis → relay-gated graded muscles → body. Engine (coprime pacemakers) + RISE-population klinokinesis. | **12/12 random-angle navigation.** Canonical single-goal navigator. |
| `dual_chemotaxis.py` | Two chemoreceptor channels (FOOD attractant + TOXIN repellent) summing at one motor — approach food, avoid toxin, no mode switch. Toxin short-range (`sigma_tox=6`) vs food (`sigma=14`). | **Works:** eats food, 0 toxin contacts. Latest / most complete embodied agent. |
| `gated_agent.py` | Mode-gated arbiter: the SAME food sensor drives approach OR avoid by which mode population is active, via **veto relays** (mode inhibits the wrong-mode klinokinesis relay). | Attractor-reversal solved. Behind-food approach is the known weak spot. |
| `nav_smooth.py` | Earlier robust navigator: tropotaxis + smooth klinokinesis; Python-side turn scalar. | Works (6/6). Superseded by `neural_agent.py` (fully-neural motor). |
| `neural_rower.py`, `rower.py`, `nmrower2.py` | Neuromuscular motor plants: two-paddle rower on a PAULA CPG; graded muscles → force. | Motor primitives. `nmrower2` is the tuned one. |
| `paula_aif.py` | Spiking active inference on a T-maze (belief accumulators, epistemic/uncertainty, action WTA). | 100% reward. |
| `forager.py`, `mini_brain.py` | Earlier WTA forager brains (approach/avoid/flee/home selection). | Working selection; older motor. |
| `tmaze_agent.py`, `record.py` | numpy discrete active-inference planner (see §"Active inference" below). | Working. |

## Recordings (self-contained canvas HTML; generators read `*_traj.json`)
- Multi-food foraging (`neural_agent`, 5/5): `multi_food_anim.html` — https://claude.ai/code/artifact/c098d4b1-060d-4e4d-8728-4d09e329283f
- Food + toxin, bigger world (`dual_chemotaxis`, 4/9 eaten, 0 toxin contacts): `dual_forage_anim.html` — https://claude.ai/code/artifact/235939d2-4cfa-4115-b048-59283ac3eaca

Recorders: `multi_food_forage.py`, `dual_forage_record.py`, `big_exploration.py`.
Animation generators: `make_forage_anim.py`, `make_dual_anim.py`.

## Shared design primitives (reused across agents)
- **Engine**: one tonic drive → coprime-period pacemaker population (`[2,3,5,7,11]`) → smooth power train; drive level = global gain. (PAULA neurons can't self-fire; a pacemaker IS a tonically-driven neuron.)
- **Relay-gated muscles**: steering gates the *phasic* CPG drive through a relay population (not tonic membrane inhibition, which kills the stroke). Read physics at the neural tick rate.
- **Opponent tropotaxis**: turn neuron = (left sensors) − (right sensors); fires on the DIFFERENCE, not absolute odour.
- **Population per function**: sensors, engine, turn, RISE-trend, relays are all populations — denser/smoother than single neurons; distributed thresholds give graded, range-invariant coding.
- **Klinokinesis**: a RISE population (POOL now − POOL delayed over staggered windows) = smooth "odour rising"; suppresses the search spiral when climbing a gradient, drives it when fleeing.

## Development
`brain_dev.py` is the active iteration file — a copy of the latest stable agent (`dual_chemotaxis.py`)
where new capabilities are built: **mushroom body** (associative memory), **spatial memory**, and
**planning**. The stable agents above are frozen references; iterate in `brain_dev.py` and promote to a
new named agent when a capability is proven.

---

## Active inference — a planning agent with a world model (`tmaze_agent.py`)

A genuine active-inference agent: it holds an explicit **generative model**, **infers** a hidden fact it
cannot observe by Bayesian belief-updating (variational free-energy minimisation), and **plans** by
minimising **expected free energy** over a tree of futures. Pure `numpy` — no PAULA / MuJoCo.

Task: the canonical **epistemic-foraging T-maze**. A reward hides LEFT or RIGHT (hidden context); a CUE
reveals which. A reward-greedy agent guesses (≈50%); the planner walks to the cue to **resolve uncertainty
first**, then collects the now-certain reward (≈100%).

Run from the repo root (`active-inference/`):
```bash
.venv/bin/python -m simulations.active_inference.tmaze_agent   # agent + ablation table
.venv/bin/python -m simulations.active_inference.record        # build tmaze_demo.html
```

Generative model: **A** likelihood P(obs|state); **B** transitions; **C** log-preferences (reward +4,
punishment −4); **D** initial prior (CENTER, reward 50/50). Perception = exact Bayesian posterior; action =
minimise `G(a) = −pragmatic − epistemic` via EFE tree search.

The current maintained run reaches **100%** reward for full EFE planning and for deep reward-only
planning, while curiosity-only planning reaches 0% reward and a random-arm guesser reaches roughly
chance. Its displayed depth-one/no-epistemic result is 0% because a tie deterministically chooses
`CENTER` repeatedly; it is *not* a fair reward-only baseline. The historical cue-blind fixed-arm
control reached **50%** over balanced contexts; its temporary recovery-corpus script was removed
from the cleaned workspace, and the interpretation is retained in
[`RESEARCH_RESUMPTION.md`](RESEARCH_RESUMPTION.md).

The spiking counterpart is `paula_aif.py` (belief accumulators as firing-rate attractors, uncertainty
neuron, and action WTA). It reaches 100% on the current T-maze loop, but its present `w_epi=0` ablation
also reaches 100%, so this implementation does not yet establish causal necessity of the epistemic route.

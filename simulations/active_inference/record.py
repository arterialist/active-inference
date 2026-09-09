"""Run the active-inference T-maze and render a self-contained HTML visualization.

Produces (next to this module):
  - tmaze_viz.json   : episode traces (belief, observations, expected-free-energy breakdown) + the
                       ablation comparison table.
  - tmaze_demo.html  : viz_template.html with the data injected — open this in a browser.

Usage:  python -m simulations.active_inference.record
"""
import json, pathlib
import numpy as np
from simulations.active_inference import tmaze_agent as T

HERE = pathlib.Path(__file__).parent


def build():
    M = T.build_model()
    episodes = []
    for ctx in (T.RL, T.RR):
        log = []
        T.run_episode(ctx, M, depth=3, log=log)
        episodes.append(dict(ctx=int(ctx), ctx_name=("LEFT" if ctx == T.RL else "RIGHT"), steps=log))

    conditions = []
    for name, kw in [("Full active inference", dict(depth=3, w_epi=1, w_prag=1)),
                     ("Greedy planner (no epistemic term)", dict(depth=3, w_epi=0, w_prag=1)),
                     ("Myopic greedy (no planning)", dict(depth=1, w_epi=0, w_prag=1)),
                     ("Pure curiosity (epistemic only)", dict(depth=3, w_epi=1, w_prag=0)),
                     ("Reward-greedy guesser (random arm)", dict(policy="guess"))]:
        cue, rew = T.summarise(M, n=100, **kw)
        conditions.append(dict(name=name, cue=cue, reward=rew))

    out = dict(
        episodes=episodes,
        conditions=conditions,
        maze=dict(  # normalized [0,1] canvas coordinates (start at bottom, arms at top)
            CENTER=[0.5, 0.82], CUE=[0.5, 0.5], LEFT=[0.16, 0.16], RIGHT=[0.84, 0.16]),
        actions=["CENTER", "CUE", "LEFT", "RIGHT"],
        summary=dict(pref=4.0, states=T.NS, note="sophisticated inference, planning depth 3"),
    )
    viz = HERE / "tmaze_viz.json"; viz.write_text(json.dumps(out))
    tpl = (HERE / "viz_template.html").read_text()
    demo = HERE / "tmaze_demo.html"
    demo.write_text(tpl.replace("/*__DATA__*/{}", json.dumps(out), 1))
    print(f"saved {viz.name} and {demo.name}")
    for c in conditions:
        print(f"  {c['name']:38s} cue {c['cue']:3d}%  reward {c['reward']:3d}%")
    print("open in a browser:", demo)


if __name__ == "__main__":
    build()
    print("@@@RECORD DONE@@@")

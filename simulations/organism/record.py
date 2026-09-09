"""Run a simulation of the Organism and render a self-contained HTML visualization.

Produces (next to this module):
  - organism_viz.json  : per-step log (bug/food/predator positions + brain state) + a strip of real
                         MuJoCo frames + summary
  - organism_demo.html : the viz_template.html with the data injected — open this in a browser.

Usage:
  python -m simulations.organism.record [seed] [steps]
  python -m simulations.organism.record -              # auto-select the richest of 8 seeds
  python -m simulations.organism.record 2 2200         # force seed 2, 2200 steps (the shipped demo)
"""
import sys, json, base64, io, pathlib
import numpy as np, mujoco
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from simulations.organism import organism as ORG
from simulations.organism import mushroom_body as MB

HERE = pathlib.Path(__file__).parent

def richness(o, steps):
    m=o.modes; present=sum(1 for k in m if m[k]>steps*0.03)  # behaviours occupying >3% of the run
    return (present*10 + min(o.good,15) + o.trips*4 + min(o.avoid_events,80)/10 + o.tox*2
            + (5 if o.energy>-5 else 0) - max(0,-o.energy)/10)

def run(force=None, steps=2200, film_n=8, Wpx=300, Hpx=225):
    best=force
    if force is None:
        bs=-1e9
        for sd in range(8):
            o=ORG.Organism(seed=sd)
            for _ in range(steps): o.step()
            r=richness(o, steps)
            print(f"seed{sd}: rich={r:.1f} good={o.good} tox={o.tox} trips={o.trips} avoid={o.avoid_events} modes={o.modes} E={o.energy:.0f}")
            if r>bs: bs=r;best=sd
        print("best seed",best,"rich",round(bs,1))
    else:
        print("forced seed",best)

    o=ORG.Organism(seed=best)
    ren=mujoco.Renderer(o.world.model, Hpx, Wpx)
    cam=mujoco.MjvCamera(); cam.distance=13.5; cam.elevation=-72; cam.azimuth=90
    every=max(1, steps//film_n)
    log=[]; film=[]
    for i in range(steps):
        o.step(); Lg=o.log[-1]; hx,hy=o.cx.home_vector()
        log.append(dict(t=Lg['t'],x=round(Lg['x'],2),y=round(Lg['y'],2),mode=Lg['mode'],val=Lg['valence'],
                        smell=Lg['smell'],energy=Lg['energy'],dpred=Lg['dpred'],dnest=Lg['dnest'],
                        tasted=Lg['tasted'],crop=Lg['crop'],hx=round(float(hx),2),hy=round(float(hy),2),
                        px=round(float(o.px),2),py=round(float(o.py),2),
                        foods=[[round(f[0],1),round(f[1],1),int(bool(f[3]))] for f in o.foods]))
        if i%every==0:
            cam.lookat[:]=[0,0,0]
            if ren.model is not o.world.model: ren=mujoco.Renderer(o.world.model, Hpx, Wpx)
            ren.update_scene(o.world.data, cam)
            buf=io.BytesIO(); Image.fromarray(ren.render()).save(buf,format='JPEG',quality=58)
            film.append({'t':Lg['t'],'img':base64.b64encode(buf.getvalue()).decode()})
    _,ag,vg=MB.present(o.net,o.core,o.nb,ORG.P_GOOD); _,ab,vb=MB.present(o.net,o.core,o.nb,ORG.P_BAD)
    neurons=MB.N_PN+MB.N_KC+3+4+4
    summary=dict(steps=o.t,good=o.good,tox=o.tox,hurt=o.hurt,trips=o.trips,avoid=o.avoid_events,modes=o.modes,
                 mb_good=[int(ag),int(vg)],mb_bad=[int(ab),int(vb)],neurons=neurons,arena=ORG.ARENA,seed=best)
    out={'log':log,'film':film,'summary':summary}

    viz=HERE/"organism_viz.json"; json.dump(out, open(viz,"w"))
    # inject into the template -> a self-contained, openable HTML
    tpl=(HERE/"viz_template.html").read_text()
    html=tpl.replace("/*__DATA__*/{}", json.dumps(out), 1)
    demo=HERE/"organism_demo.html"; demo.write_text(html)
    kb=viz.stat().st_size/1024; mb=demo.stat().st_size/1e6
    print(f"saved {viz.name} ({kb:.0f}KB) and {demo.name} ({mb:.2f}MB) | "
          f"good={o.good} tox={o.tox} trips={o.trips} avoid={o.avoid_events} modes={o.modes}")
    print("open in a browser:", demo)
    return best

if __name__=="__main__":
    force=None
    if len(sys.argv)>1 and sys.argv[1]!='-': force=int(sys.argv[1])
    steps=int(sys.argv[2]) if len(sys.argv)>2 else 2200
    run(force=force, steps=steps)
    print("@@@RECORD DONE@@@")

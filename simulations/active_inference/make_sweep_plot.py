"""Reads sweep_*.json (one per environment scenario) and writes a self-contained HTML grid comparing the
mushroom-body learning across environments: LEARNED AVERSION (toxic vs food odorant) and toxin crossings
per window over the horizon. Shows how the plasticity mechanism responds to contact frequency."""
import json, pathlib
D=pathlib.Path(__file__).parent
ORDER=["food_only","sparse_toxin","minefield","minefield_sharp","toxin_removed"]
TITLES={"food_only":"Food only (control)","sparse_toxin":"Sparse toxin (rare contact)",
        "minefield":"Minefield (broad identity — cross-contamination)","minefield_sharp":"Minefield (sharp identity)",
        "toxin_removed":"Minefield → toxin removed"}
data={}
for name in ORDER:
    f=D/f"sweep_{name}.json"
    if f.exists():
        try: data[name]=json.load(open(f))
        except Exception: pass

PW,PH=430,240; ML,MB,MT,MR=44,30,30,14
gw,gh=PW-ML-MR, PH-MT-MB
def panel(name,x0,y0):
    d=data.get(name);
    if not d or not d.get("curve"):
        return f'<text x="{x0+ML}" y="{y0+PH/2}" class="ax">{TITLES.get(name,name)}: (starting…)</text>'
    c=d["curve"]; xs=[p["ticks"] for p in c]
    ag=[p["aversion_good"] for p in c]; at=[p["aversion_toxic"] for p in c]; tox=[p["toxin_win"] for p in c]
    X1=max(xs+[1]); vmax=max(max(ag+at+[1]),8); cmax=max(max(tox+[1]),3)
    def sx(v): return x0+ML+(v/max(X1,1))*gw
    def sy(v,mx): return y0+MT+gh-(v/max(mx,1e-9))*gh
    def line(vals):
        return "M"+" L".join(f"{sx(t):.1f},{sy(v,vmax):.1f}" for v,t in zip(vals,xs))
    bars=""
    bw=max(1.0,gw/max(len(tox),1)*0.8)
    for v,t in zip(tox,xs):
        if v<=0: continue
        ph=(v/cmax)*gh*0.5; bars+=f'<rect x="{sx(t)-bw/2:.1f}" y="{y0+MT+gh-ph:.1f}" width="{bw:.1f}" height="{ph:.1f}" fill="var(--tox)" opacity="0.28"/>'
    # removal marker
    rem=""
    for p in c:
        if p.get("removed"):
            rx=sx(p["ticks"]); rem=f'<line x1="{rx:.0f}" y1="{y0+MT}" x2="{rx:.0f}" y2="{y0+MT+gh}" class="rem"/><text x="{rx:.0f}" y="{y0+MT-4}" class="remt" text-anchor="middle">danger gone</text>'
            break
    fin=d.get("final",{})
    grid=""
    for fr in (0,.5,1):
        gy=y0+MT+gh-fr*gh; grid+=f'<line x1="{x0+ML}" y1="{gy:.0f}" x2="{x0+ML+gw}" y2="{gy:.0f}" class="grid"/><text x="{x0+ML-6}" y="{gy+3:.0f}" class="ax" text-anchor="end">{vmax*fr:.0f}</text>'
    return f'''<text x="{x0+ML}" y="{y0+18}" class="ttl">{TITLES.get(name,name)}</text>
    {grid}{bars}{rem}
    <path d="{line(at)}" fill="none" stroke="var(--tox)" stroke-width="2.2"/>
    <path d="{line(ag)}" fill="none" stroke="var(--good)" stroke-width="2.2"/>
    <text x="{x0+ML+gw}" y="{y0+18}" class="fin" text-anchor="end">final tox={fin.get("toxic","·")} food={fin.get("good","·")}</text>
    <text x="{x0+ML}" y="{y0+PH-6}" class="ax">{xs[-1]/1e6:.2f}M ticks</text>'''

COLS=2; ROWS=3; W=PW*COLS; H=PH*ROWS+30
svg=f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif">'
for i,name in enumerate(ORDER):
    x0=(i%COLS)*PW; y0=(i//COLS)*PH+20
    svg+=panel(name,x0,y0)
svg+="</svg>"

HTML=f'''<title>Mushroom-body learning across environments</title>
<style>
:root{{--bg:#0d1014;--panel:#151a21;--ink:#e8eef4;--muted:#8a96a3;--line:#232b34;--tox:#ff5470;--good:#39d0c8;--rem:#ffc14d;}}
@media (prefers-color-scheme:light){{:root{{--bg:#f5f8fb;--panel:#fff;--ink:#182029;--muted:#5a6675;--line:#e3e8ee;--tox:#d61f43;--good:#0e9c94;--rem:#c98a00;}}}}
:root[data-theme=dark]{{--bg:#0d1014;--panel:#151a21;--ink:#e8eef4;--muted:#8a96a3;--line:#232b34;--tox:#ff5470;--good:#39d0c8;--rem:#ffc14d;}}
:root[data-theme=light]{{--bg:#f5f8fb;--panel:#fff;--ink:#182029;--muted:#5a6675;--line:#e3e8ee;--tox:#d61f43;--good:#0e9c94;--rem:#c98a00;}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.5}}
.wrap{{max-width:940px;margin:0 auto;padding:26px 18px 40px}}
h1{{font-size:20px;font-weight:650;margin:0 0 2px}} .sub{{color:var(--muted);font-size:13.5px;margin:0 0 16px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:12px}}
.ttl{{fill:var(--ink);font-size:12.5px;font-weight:600}} .ax{{fill:var(--muted);font-size:10px}}
.fin{{fill:var(--muted);font-size:10px}} .grid{{stroke:var(--line);stroke-width:1}}
.rem{{stroke:var(--rem);stroke-width:1.4;stroke-dasharray:3 3}} .remt{{fill:var(--rem);font-size:9.5px}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;margin-top:12px;font-size:12.5px;color:var(--muted)}}
.sw{{display:inline-block;width:16px;height:4px;border-radius:2px;margin-right:5px;vertical-align:middle}}
</style>
<div class="wrap">
<h1>Mushroom-body valence learning across environments</h1>
<p class="sub">Same fully-neural agent + learning rule, six worlds spanning the toxin-contact-frequency space.
Each panel: learned aversion to the toxic odorant (red) vs the food odorant (teal), with toxin crossings per
window (faint red bars). The test: forget danger when it is <em>gone</em>, but hold it under constant exposure.</p>
<div class="card">{svg}
<div class="legend">
  <span><span class="sw" style="background:var(--tox)"></span>aversion — toxic odorant</span>
  <span><span class="sw" style="background:var(--good)"></span>aversion — food odorant</span>
  <span><span class="sw" style="background:var(--tox);opacity:.4"></span>toxin crossings / window</span>
</div></div></div>'''
p=D/"mb_sweep_plot.html"; open(p,"w").write(HTML)
print("wrote",p,"| scenarios with data:",list(data.keys()))

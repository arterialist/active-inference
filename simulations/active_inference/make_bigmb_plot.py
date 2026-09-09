"""Reads big_mushroom_traj.json (or big_mushroom_progress.json if the run is still going) and writes a
self-contained HTML learning-curve plot: LEARNED AVERSION (toxic vs food odorant) and TOXIN CROSSINGS
per window over the 2,000,000-tick horizon. Static inline SVG, theme-aware."""
import json, pathlib
D=pathlib.Path(__file__).parent
src = D/"big_mushroom_traj.json"
if not src.exists(): src = D/"big_mushroom_progress.json"
data=json.load(open(src))
curve=data["curve"]
xs=[c["ticks"] for c in curve]
ag=[c["aversion_good"] for c in curve]; at=[c["aversion_toxic"] for c in curve]
tox=[c["toxin_win"] for c in curve]; eat=[c["eaten_win"] for c in curve]
X0,X1=0,(max(xs) if xs else 1)
def sxx(v,x,w): return x + (v-X0)/max(X1-X0,1)*w
def path(vals, x,y,w,h, vmax):
    pts=[]
    for v,t in zip(vals,xs):
        px=sxx(t,x,w); py=y+h-(v/max(vmax,1e-9))*h; pts.append(f"{px:.1f},{py:.1f}")
    return "M"+" L".join(pts) if pts else ""
def bars(vals,x,y,w,h,vmax,color):
    n=len(vals); bw=max(1.0,w/max(n,1)*0.7); out=[]
    for i,(v,t) in enumerate(zip(vals,xs)):
        px=sxx(t,x,w)-bw/2; ph=(v/max(vmax,1e-9))*h
        out.append(f'<rect x="{px:.1f}" y="{y+h-ph:.1f}" width="{bw:.1f}" height="{ph:.1f}" fill="{color}" opacity="0.55"/>')
    return "".join(out)

W,H=920,620; ML,MR,MT=64,24,16
pw=W-ML-MR
# panel geometry
p1y,p1h=44,230      # aversion
p2y,p2h=360,210     # toxin crossings + eaten
avmax=max(max(ag+at+[1]),4)
crmax=max(max(tox+[1]),2); emax=max(max(eat+[1]),2)
def ticks_x(y,h):
    out=[]
    for frac in (0,.25,.5,.75,1):
        tv=X0+(X1-X0)*frac; px=ML+pw*frac
        out.append(f'<line x1="{px}" y1="{y}" x2="{px}" y2="{y+h}" class="grid"/>')
        out.append(f'<text x="{px}" y="{y+h+16}" class="ax" text-anchor="middle">{tv/1e6:.2f}M</text>')
    return "".join(out)
def yaxis(y,h,vmax,label):
    out=[f'<text x="14" y="{y+h/2}" class="ax" transform="rotate(-90 14 {y+h/2})" text-anchor="middle">{label}</text>']
    for frac in (0,.5,1):
        vv=vmax*frac; py=y+h-frac*h
        out.append(f'<line x1="{ML}" y1="{py}" x2="{ML+pw}" y2="{py}" class="grid"/>')
        out.append(f'<text x="{ML-8}" y="{py+4}" class="ax" text-anchor="end">{vv:.0f}</text>')
    return "".join(out)

fa=data.get("final_aversion",{}); ne=data.get("n_eaten","?"); tc=data.get("toxin_crossings","?")
svg=f'''<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif">
<text x="{ML}" y="24" class="ttl">Learned aversion</text>
{yaxis(p1y,p1h,avmax,"MBON spikes / 40")}{ticks_x(p1y,p1h)}
<path d="{path(at,ML,p1y,pw,p1h,avmax)}" fill="none" stroke="var(--tox)" stroke-width="2.5"/>
<path d="{path(ag,ML,p1y,pw,p1h,avmax)}" fill="none" stroke="var(--good)" stroke-width="2.5"/>
<text x="{ML}" y="{p2y-14}" class="ttl">Toxin crossings per window &amp; food eaten per window</text>
{yaxis(p2y,p2h,max(crmax,emax),"per window")}{ticks_x(p2y,p2h)}
{bars(eat,ML,p2y,pw,p2h,max(crmax,emax),"var(--food)")}
{bars(tox,ML,p2y,pw,p2h,max(crmax,emax),"var(--tox)")}
<text x="{ML}" y="{H-6}" class="ax">neural ticks &#8594;</text>
</svg>'''

HTML=f'''<title>Mushroom-body learning over 2,000,000 ticks</title>
<style>
:root{{--bg:#0d1014;--panel:#151a21;--ink:#e8eef4;--muted:#8a96a3;--line:#2a333d;--tox:#ff5470;--good:#39d0c8;--food:#ffc14d;}}
@media (prefers-color-scheme:light){{:root{{--bg:#f5f8fb;--panel:#fff;--ink:#182029;--muted:#5a6675;--line:#e3e8ee;--tox:#d61f43;--good:#0e9c94;--food:#d98a00;}}}}
:root[data-theme=dark]{{--bg:#0d1014;--panel:#151a21;--ink:#e8eef4;--muted:#8a96a3;--line:#2a333d;--tox:#ff5470;--good:#39d0c8;--food:#ffc14d;}}
:root[data-theme=light]{{--bg:#f5f8fb;--panel:#fff;--ink:#182029;--muted:#5a6675;--line:#e3e8ee;--tox:#d61f43;--good:#0e9c94;--food:#d98a00;}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.55}}
.wrap{{max-width:940px;margin:0 auto;padding:28px 20px 40px}}
h1{{font-size:20px;font-weight:650;margin:0 0 2px}}
.sub{{color:var(--muted);font-size:13.5px;margin:0 0 18px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px}}
.ttl{{fill:var(--ink);font-size:14px;font-weight:600}}
.ax{{fill:var(--muted);font-size:11px}}
.grid{{stroke:var(--line);stroke-width:1}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;margin-top:14px;font-size:12.5px;color:var(--muted)}}
.sw{{display:inline-block;width:14px;height:4px;border-radius:2px;margin-right:5px;vertical-align:middle}}
.stat{{display:flex;gap:20px;flex-wrap:wrap;margin-top:12px;font-size:13px}}
.stat .v{{font-weight:650;font-variant-numeric:tabular-nums}}.stat .k{{color:var(--muted)}}
</style>
<div class="wrap">
<h1>Mushroom-body valence learning over 2,000,000 ticks</h1>
<p class="sub">A fully-neural agent forages a dense world of food (odorant 0) and toxin (odorant 1). The
mushroom body learns each odour's valence from experience — cortisol on toxin contact potentiates aversion,
dopamine on food suppresses it, with slow-decay consolidation.</p>
<div class="card">{svg}
<div class="legend">
  <span><span class="sw" style="background:var(--tox)"></span>aversion to toxic odorant</span>
  <span><span class="sw" style="background:var(--good)"></span>aversion to food odorant</span>
  <span><span class="sw" style="background:var(--food)"></span>food eaten / window</span>
  <span><span class="sw" style="background:var(--tox)"></span>toxin crossings / window</span>
</div>
<div class="stat">
  <span><span class="k">final aversion — toxic </span><span class="v">{fa.get("toxic","?")}</span><span class="k"> · food </span><span class="v">{fa.get("good","?")}</span></span>
  <span><span class="k">total food eaten </span><span class="v">{ne}</span></span>
  <span><span class="k">total toxin crossings </span><span class="v">{tc}</span></span>
</div>
</div></div>'''

p=D/"big_mushroom_plot.html"; open(p,"w").write(HTML)
print("wrote",p,"points",len(curve))

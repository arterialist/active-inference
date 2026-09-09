"""Reads sweep_<scenario>_traj.json for all scenarios and writes ONE self-contained HTML that plays every
run in a synchronized grid: the agent foraging each world (food = amber, respawns; toxins = red ✕, vanish
after removal), its path, and a live learned-aversion readout (toxic vs food) that tracks the mushroom body
learning as it happens. No external assets."""
import json, pathlib
D=pathlib.Path(__file__).parent
ORDER=["food_only","sparse_toxin","minefield","minefield_sharp","toxin_removed"]
TITLES={"food_only":"Food only (control)","sparse_toxin":"Sparse toxin","minefield":"Minefield (broad identity)",
        "minefield_sharp":"Minefield (sharp identity)","toxin_removed":"Minefield → toxin removed"}
data={}
for name in ORDER:
    f=D/f"sweep_{name}_traj.json"
    if f.exists():
        try:
            d=json.load(open(f))
            if d.get("frames"): data[name]=d
        except Exception: pass
payload=json.dumps({"order":[n for n in ORDER if n in data],"titles":TITLES,
                    "scen":{n:{"frames":data[n]["frames"],"toxins":data[n]["toxins"],
                               "curve":[[c["ticks"],c["aversion_good"],c["aversion_toxic"]] for c in data[n]["curve"]],
                               "framestride":data[n]["framestride"],"sub":data[n]["sub"]} for n in data}},
                   separators=(",",":"))

HTML=r"""<title>Mushroom-body learning — all environments</title>
<style>
:root{--bg:#0d1014;--panel:#151a21;--ink:#e8eef4;--muted:#8a96a3;--line:#232b34;--food:#ffc14d;--tox:#ff5470;--good:#39d0c8;--body:#f2f6fa;--accent:#4cc2ff;}
@media (prefers-color-scheme:light){:root{--bg:#f5f8fb;--panel:#fff;--ink:#182029;--muted:#5a6675;--line:#e3e8ee;--food:#d98a00;--tox:#d61f43;--good:#0e9c94;--body:#182029;--accent:#0a7ea4;}}
:root[data-theme=dark]{--bg:#0d1014;--panel:#151a21;--ink:#e8eef4;--muted:#8a96a3;--line:#232b34;--food:#ffc14d;--tox:#ff5470;--good:#39d0c8;--body:#f2f6fa;--accent:#4cc2ff;}
:root[data-theme=light]{--bg:#f5f8fb;--panel:#fff;--ink:#182029;--muted:#5a6675;--line:#e3e8ee;--food:#d98a00;--tox:#d61f43;--good:#0e9c94;--body:#182029;--accent:#0a7ea4;}
body{margin:0;background:var(--bg);color:var(--ink);font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.5}
.wrap{max-width:960px;margin:0 auto;padding:24px 16px 40px}
h1{font-size:20px;font-weight:650;margin:0 0 2px}.sub{color:var(--muted);font-size:13px;margin:0 0 16px}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
.cell{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:8px}
.cell h2{font-size:12.5px;font-weight:600;margin:2px 4px 6px}
canvas{width:100%;height:auto;display:block;border-radius:7px}
.hud{display:flex;gap:12px;align-items:center;margin:6px 4px 2px;font-size:11px;color:var(--muted)}
.hud .v{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}
.bars{display:flex;gap:8px;margin:4px 4px 2px;font-size:10.5px}
.bar{flex:1;height:8px;background:var(--line);border-radius:4px;overflow:hidden;position:relative}
.bar>i{position:absolute;left:0;top:0;bottom:0;display:block}
.ctl{display:flex;gap:12px;align-items:center;margin:16px 4px 0}
button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:7px 16px;font-size:13px;font-weight:600;cursor:pointer}
button:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
input[type=range]{flex:1;accent-color:var(--accent)}
.lg{display:flex;gap:16px;flex-wrap:wrap;margin-top:12px;font-size:12px;color:var(--muted)}
.sw{display:inline-block;width:13px;height:4px;border-radius:2px;margin-right:5px;vertical-align:middle}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;vertical-align:middle}
</style>
<div class="wrap">
<h1>Mushroom-body valence learning — every environment</h1>
<p class="sub">The same fully-neural agent + learning rule in five worlds, playing in sync. Watch it forage
(food = amber), cross toxins (red ✕), and the aversion bars below each world track what it has learned:
<b style="color:var(--good)">food</b> vs <b style="color:var(--tox)">toxin</b>.</p>
<div class="grid" id="grid"></div>
<div class="ctl">
  <button id="play">Pause</button><input id="scrub" type="range" min="0" max="1000" value="0"><button id="speed">1×</button>
</div>
<div class="lg">
  <span><span class="dot" style="background:var(--food)"></span>food</span>
  <span><span style="color:var(--tox);font-weight:800">✕</span> toxin</span>
  <span><span class="dot" style="background:var(--body)"></span>agent</span>
  <span><span class="sw" style="background:var(--good)"></span>learned aversion: food</span>
  <span><span class="sw" style="background:var(--tox)"></span>learned aversion: toxin</span>
</div></div>
<script>
const DATA=__PAYLOAD__;
const AVMAX=90;
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const grid=document.getElementById('grid');
const panels={};
for(const name of DATA.order){
  const cell=document.createElement('div'); cell.className='cell';
  cell.innerHTML=`<h2>${DATA.titles[name]}</h2><canvas width="760" height="560"></canvas>
    <div class="hud"><span>eaten <span class="v" nm="eat">0</span></span><span>toxin contacts <span class="v" nm="hit">0</span></span><span nm="ticks"></span></div>
    <div class="bars"><span style="color:var(--good)">food</span><span class="bar"><i nm="bg" style="background:var(--good)"></i></span>
      <span style="color:var(--tox)">toxin</span><span class="bar"><i nm="bt" style="background:var(--tox)"></i></span></div>`;
  grid.appendChild(cell);
  const s=DATA.scen[name];
  let pts=s.frames.map(f=>[f[0],f[1]]).concat(s.toxins);
  let lo=Math.min(...pts.flat())-1.2, hi=Math.max(...pts.flat())+1.2; const span=Math.max(hi-lo,8);
  panels[name]={s,cv:cell.querySelector('canvas'),lo,span,
    el:{eat:cell.querySelector('[nm=eat]'),hit:cell.querySelector('[nm=hit]'),ticks:cell.querySelector('[nm=ticks]'),
        bg:cell.querySelector('[nm=bg]'),bt:cell.querySelector('[nm=bt]')}};
}
function aversionAt(s,tick){ // last curve point <= tick
  let g=0,t=0;
  for(const c of s.curve){ if(c[0]<=tick){g=c[1];t=c[2];} else break; }
  return [g,t];
}
function drawPanel(name,frac){
  const P=panels[name], s=P.s, g=P.cv.getContext('2d'), W=P.cv.width, H=P.cv.height;
  const nf=s.frames.length, fi=Math.min(nf-1, Math.floor(frac*(nf-1)));
  const sx=x=>(x-P.lo)/P.span*W, sy=y=>H-(y-P.lo)/P.span*H;
  g.clearRect(0,0,W,H);
  const F=s.frames[fi], txlive=F[6];
  // toxins (vanish after removal)
  if(txlive){ g.strokeStyle=css('--tox');
    for(const t of s.toxins){const cx=sx(t[0]),cy=sy(t[1]);
      g.globalAlpha=.35;g.lineWidth=1.5;g.beginPath();g.arc(cx,cy,9,0,7);g.stroke();g.globalAlpha=1;
      g.lineWidth=2.4;g.beginPath();g.moveTo(cx-4,cy-4);g.lineTo(cx+4,cy+4);g.moveTo(cx+4,cy-4);g.lineTo(cx-4,cy+4);g.stroke();}
  }
  // path (faint, last ~120 frames)
  g.lineWidth=1.6;g.strokeStyle=css('--accent');g.globalAlpha=.5;g.beginPath();
  for(let k=Math.max(0,fi-120);k<=fi;k++){const p=s.frames[k];const cx=sx(p[0]),cy=sy(p[1]);k===Math.max(0,fi-120)?g.moveTo(cx,cy):g.lineTo(cx,cy);}
  g.stroke();g.globalAlpha=1;
  // food
  g.fillStyle=css('--food');
  for(const fp of F[3]){g.beginPath();g.arc(sx(fp[0]),sy(fp[1]),4,0,7);g.fill();}
  // agent
  const cx=sx(F[0]),cy=sy(F[1]),fwd=F[2]+Math.PI,ang=Math.atan2(-Math.sin(fwd),Math.cos(fwd));
  g.save();g.translate(cx,cy);g.rotate(ang);g.fillStyle=css('--body');
  g.beginPath();g.moveTo(11,0);g.lineTo(-8,-7);g.lineTo(-4,0);g.lineTo(-8,7);g.closePath();g.fill();g.restore();
  // HUD
  const tick=fi*s.framestride*s.sub, av=aversionAt(s,tick);
  P.el.eat.textContent=F[4]; P.el.hit.textContent=F[5];
  P.el.ticks.textContent=(tick/1000).toFixed(0)+'k ticks';
  P.el.bg.style.width=Math.min(100,av[0]/AVMAX*100)+'%';
  P.el.bt.style.width=Math.min(100,av[1]/AVMAX*100)+'%';
}
let playing=true,speed=1,frac=0,last=performance.now();
const scrub=document.getElementById('scrub');
function loop(now){const dt=now-last;last=now;
  if(playing){frac+=dt*speed/45000; if(frac>=1){frac=1;playing=false;document.getElementById('play').textContent='Replay';} scrub.value=frac*1000;}
  for(const n of DATA.order) drawPanel(n,frac); requestAnimationFrame(loop);}
document.getElementById('play').onclick=()=>{const b=document.getElementById('play');if(frac>=1)frac=0;playing=!playing;b.textContent=playing?'Pause':(frac>=1?'Replay':'Play');};
document.getElementById('speed').onclick=()=>{speed=speed===1?2:(speed===2?4:1);document.getElementById('speed').textContent=speed+'×';};
scrub.oninput=()=>{frac=+scrub.value/1000;playing=false;document.getElementById('play').textContent='Play';for(const n of DATA.order)drawPanel(n,frac);};
requestAnimationFrame(loop);
</script>"""
out=HTML.replace("__PAYLOAD__",payload)
p=D/"mb_sweep_anim.html"; open(p,"w").write(out)
print("wrote",p,"| scenarios:",list(data.keys()),"| bytes",len(out))

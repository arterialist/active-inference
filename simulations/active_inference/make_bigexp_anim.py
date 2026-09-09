"""Reads big_exploration_traj.json and writes a self-contained canvas animation of the full 2,000,000-tick
open-ended forage in a dense 40-food / 30-toxin world (food respawns on eating; toxins persist)."""
import json, pathlib
D=pathlib.Path(__file__).parent
data=json.load(open(D/"big_exploration_traj.json"))
payload=json.dumps(data, separators=(",",":"))

HTML=r"""<title>2,000,000-tick foraging run</title>
<style>
:root{--bg:#0c0f13;--panel:#141922;--ink:#e8eef4;--muted:#8a96a3;--line:#222a34;
  --trail:#4cc2ff;--tox:#ff5470;--food:#ffc14d;--body:#f2f6fa;--accent:#4cc2ff;}
@media (prefers-color-scheme:light){:root{--bg:#f5f8fb;--panel:#fff;--ink:#182029;--muted:#5a6675;--line:#e3e8ee;
  --trail:#0a7ea4;--tox:#d61f43;--food:#d98a00;--body:#182029;--accent:#0a7ea4;}}
:root[data-theme=dark]{--bg:#0c0f13;--panel:#141922;--ink:#e8eef4;--muted:#8a96a3;--line:#222a34;
  --trail:#4cc2ff;--tox:#ff5470;--food:#ffc14d;--body:#f2f6fa;--accent:#4cc2ff;}
:root[data-theme=light]{--bg:#f5f8fb;--panel:#fff;--ink:#182029;--muted:#5a6675;--line:#e3e8ee;
  --trail:#0a7ea4;--tox:#d61f43;--food:#d98a00;--body:#182029;--accent:#0a7ea4;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.5}
.wrap{max-width:880px;margin:0 auto;padding:28px 20px 40px}
h1{font-size:20px;font-weight:650;margin:0 0 2px;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:13.5px;margin:0 0 18px}
.stage{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.2)}
canvas{width:100%;height:auto;display:block;border-radius:8px}
.hud{display:flex;gap:18px;flex-wrap:wrap;align-items:center;margin-top:12px;font-size:13px}
.hud .k{color:var(--muted)}.hud .v{font-variant-numeric:tabular-nums;font-weight:600}
.ctl{display:flex;gap:12px;align-items:center;margin-top:12px}
button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:7px 16px;font-size:13px;font-weight:600;cursor:pointer}
button:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
input[type=range]{flex:1;accent-color:var(--accent)}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:14px;font-size:12.5px;color:var(--muted)}
.sw{display:inline-block;width:16px;height:4px;border-radius:2px;margin-right:5px;vertical-align:middle}
.x{color:var(--tox);font-weight:800;margin-right:4px}
</style>
<div class="wrap">
<h1>2,000,000-tick foraging run</h1>
<p class="sub">A dual-channel PAULA agent foraging an open-ended dense world — 40 food sources (they respawn
when eaten) and 30 persistent toxins. Full trajectory over two million neural ticks.</p>
<div class="stage">
  <canvas id="c" width="1680" height="1180"></canvas>
  <div class="hud">
    <span><span class="k">food eaten </span><span class="v" id="eaten">0</span></span>
    <span><span class="k">toxin crossings </span><span class="v" id="hits">0</span></span>
    <span><span class="k">neural ticks </span><span class="v" id="ticks">0</span></span>
    <span><span class="k">speed </span><span class="v" id="spd">3×</span></span>
  </div>
  <div class="ctl">
    <button id="play">Pause</button><input id="scrub" type="range" min="0" value="0"><button id="speed">3×</button>
  </div>
  <div class="legend">
    <span><span class="sw" style="background:var(--trail)"></span>path</span>
    <span><span class="sw" style="background:var(--food);height:9px;width:9px;border-radius:50%"></span>food</span>
    <span><span class="x">✕</span>toxin</span>
    <span><span class="sw" style="background:var(--body);height:9px;width:9px;border-radius:50%"></span>agent</span>
  </div>
</div></div>
<script>
const DATA=__PAYLOAD__;
const F=DATA.frames, TOX=DATA.toxins, sub=DATA.sub, stride=DATA.stride, sigTox=DATA.sigma_tox;
const cv=document.getElementById('c'),g=cv.getContext('2d'),W=cv.width,H=cv.height;
let pts=F.map(f=>[f[0],f[1]]).concat(TOX);
let lo=Math.min(...pts.flat())-1.5, hi=Math.max(...pts.flat())+1.5; const span=Math.max(hi-lo,10);
const sx=x=>(x-lo)/span*W, sy=y=>H-(y-lo)/span*H;
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const dark=()=>matchMedia('(prefers-color-scheme:dark)').matches||document.documentElement.dataset.theme==='dark';
const GN=90; let tg=new Float32Array(GN*GN),tmax=1e-6;
for(let j=0;j<GN;j++)for(let i=0;i<GN;i++){const wx=lo+span*(i+.5)/GN,wy=lo+span*(j+.5)/GN;let s=0;
  for(const t of TOX){const dx=wx-t[0],dy=wy-t[1];s+=Math.exp(-(dx*dx+dy*dy)/sigTox);}tg[j*GN+i]=s;if(s>tmax)tmax=s;}
function draw(fi){
  g.clearRect(0,0,W,H); const dk=dark(); const cw=W/GN,ch=H/GN;
  for(let j=0;j<GN;j++)for(let i=0;i<GN;i++){const v=Math.pow(tg[j*GN+i]/tmax,.9);if(v<.03)continue;
    g.fillStyle=dk?`rgba(255,84,112,${.28*v})`:`rgba(214,31,67,${.17*v})`;g.fillRect(i*cw-.5,H-(j+1)*ch-.5,cw+1,ch+1);}
  // full trail (single colour), fading with age
  g.lineWidth=2.0; g.lineJoin='round'; g.strokeStyle=css('--trail'); g.globalAlpha=.7; g.beginPath();
  for(let t=0;t<=fi;t++){const p=F[t];t?g.lineTo(sx(p[0]),sy(p[1])):g.moveTo(sx(p[0]),sy(p[1]));}
  g.stroke(); g.globalAlpha=1;
  const cf=F[fi];
  // food (current frame's active food positions; they respawn as the agent eats)
  const food=cf[5]||[]; g.fillStyle=css('--food');
  for(const fp of food){const cx=sx(fp[0]),cy=sy(fp[1]);g.beginPath();g.arc(cx,cy,4.5,0,7);g.fill();}
  for(const t of TOX){const cx=sx(t[0]),cy=sy(t[1]);g.strokeStyle=css('--tox');
    g.globalAlpha=.4;g.lineWidth=2;g.beginPath();g.arc(cx,cy,16,0,7);g.stroke();g.globalAlpha=1;
    g.lineWidth=3.4;g.beginPath();g.moveTo(cx-7,cy-7);g.lineTo(cx+7,cy+7);g.moveTo(cx+7,cy-7);g.lineTo(cx-7,cy+7);g.stroke();}
  const cx=sx(cf[0]),cy=sy(cf[1]),fwd=cf[2]+Math.PI,ang=Math.atan2(-Math.sin(fwd),Math.cos(fwd));
  g.save();g.translate(cx,cy);g.rotate(ang);g.fillStyle=css('--body');
  g.beginPath();g.moveTo(13,0);g.lineTo(-9,-8);g.lineTo(-5,0);g.lineTo(-9,8);g.closePath();g.fill();g.restore();
  document.getElementById('eaten').textContent=cf[3];
  document.getElementById('hits').textContent=cf[4];
  document.getElementById('ticks').textContent=(fi*stride*sub).toLocaleString();
}
let fi=0,playing=true,speed=3,acc=0,last=performance.now();
const scrub=document.getElementById('scrub');scrub.max=F.length-1;
function loop(now){const dt=now-last;last=now;
  if(playing){acc+=dt*speed/33;while(acc>=1){fi++;acc--;if(fi>=F.length){fi=F.length-1;playing=false;document.getElementById('play').textContent='Replay';break;}}scrub.value=fi;}
  draw(fi);requestAnimationFrame(loop);}
document.getElementById('play').onclick=()=>{const b=document.getElementById('play');if(fi>=F.length-1)fi=0;playing=!playing;b.textContent=playing?'Pause':(fi>=F.length-1?'Replay':'Play');};
document.getElementById('speed').onclick=()=>{speed=speed===3?6:(speed===6?1:3);document.getElementById('speed').textContent=speed+'×';document.getElementById('spd').textContent=speed+'×';};
scrub.oninput=()=>{fi=+scrub.value;playing=false;document.getElementById('play').textContent='Play';draw(fi);};
requestAnimationFrame(loop);
</script>"""
out=HTML.replace("__PAYLOAD__",payload)
p=D/"big_exploration_anim.html"; open(p,"w").write(out)
print("wrote",p,"bytes",len(out))

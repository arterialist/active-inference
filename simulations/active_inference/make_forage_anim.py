"""Reads multi_food_traj.json and writes a self-contained HTML canvas animation of the fully-neural
agent foraging a multi-source odour field. No external assets."""
import json, pathlib
D=pathlib.Path(__file__).parent
data=json.load(open(D/"multi_food_traj.json"))
payload=json.dumps(data, separators=(",",":"))

HTML = """<title>Fully-neural agent — multi-food foraging</title>
<style>
:root{
  --bg:#0e1116; --panel:#161b22; --ink:#e6edf3; --muted:#8b949e; --line:#232a33;
  --accent:#4cc2ff; --trail:#4cc2ff; --food:#ffb454; --eaten:#3d4653; --body:#e6edf3;
}
@media (prefers-color-scheme:light){
  :root{ --bg:#f6f8fa; --panel:#ffffff; --ink:#1a1f26; --muted:#5a636e; --line:#e2e6ea;
         --accent:#0a7ea4; --trail:#0a7ea4; --food:#e07b00; --eaten:#c4ccd4; --body:#1a1f26; }
}
:root[data-theme=dark]{ --bg:#0e1116; --panel:#161b22; --ink:#e6edf3; --muted:#8b949e; --line:#232a33;
  --accent:#4cc2ff; --trail:#4cc2ff; --food:#ffb454; --eaten:#3d4653; --body:#e6edf3; }
:root[data-theme=light]{ --bg:#f6f8fa; --panel:#ffffff; --ink:#1a1f26; --muted:#5a636e; --line:#e2e6ea;
  --accent:#0a7ea4; --trail:#0a7ea4; --food:#e07b00; --eaten:#c4ccd4; --body:#1a1f26; }
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.5}
.wrap{max-width:820px;margin:0 auto;padding:28px 20px 40px}
h1{font-size:20px;font-weight:650;margin:0 0 2px;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:13.5px;margin:0 0 20px}
.stage{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px;
  box-shadow:0 1px 3px rgba(0,0,0,.18)}
canvas{width:100%;height:auto;display:block;border-radius:8px;background:transparent}
.hud{display:flex;gap:18px;flex-wrap:wrap;align-items:center;margin-top:12px;font-size:13px}
.hud .k{color:var(--muted)}.hud .v{font-variant-numeric:tabular-nums;font-weight:600}
.ctl{display:flex;gap:12px;align-items:center;margin-top:12px}
button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:7px 16px;
  font-size:13px;font-weight:600;cursor:pointer}
button:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
input[type=range]{flex:1;accent-color:var(--accent)}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:14px;font-size:12.5px;color:var(--muted)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:middle}
</style>
<div class="wrap">
  <h1>Fully-neural agent — multi-food foraging</h1>
  <p class="sub">One PAULA spiking network from odour to muscle. It senses the summed odour field, swims to a
  source, consumes it (removing it from the field), and re-navigates to the next — behaviour lives in the wiring.</p>
  <div class="stage">
    <canvas id="c" width="1580" height="1120"></canvas>
    <div class="hud">
      <span><span class="k">foraged </span><span class="v" id="eaten">0</span><span class="k"> / </span><span class="v" id="total">0</span></span>
      <span><span class="k">neural ticks </span><span class="v" id="ticks">0</span></span>
      <span><span class="k">speed </span><span class="v" id="spd">2×</span></span>
    </div>
    <div class="ctl">
      <button id="play">Pause</button>
      <input id="scrub" type="range" min="0" max="100" value="0">
      <button id="speed">2×</button>
    </div>
    <div class="legend">
      <span><span class="dot" style="background:var(--food)"></span>odour source</span>
      <span><span class="dot" style="background:var(--eaten)"></span>consumed</span>
      <span><span class="dot" style="background:var(--body)"></span>agent</span>
      <span><span class="dot" style="background:var(--trail)"></span>path</span>
    </div>
  </div>
</div>
<script>
const DATA=__PAYLOAD__;
const foods=DATA.foods, frames=DATA.frames, sub=DATA.sub, stride=DATA.stride, sigma=14.0;
const cv=document.getElementById('c'), g=cv.getContext('2d');
const W=cv.width, H=cv.height;
// world bounds from trajectory + foods, padded
let xs=frames.map(f=>f[0]).concat(foods.map(f=>f[0]));
let ys=frames.map(f=>f[1]).concat(foods.map(f=>f[1]));
let lo=Math.min(Math.min(...xs),Math.min(...ys))-1.2, hi=Math.max(Math.max(...xs),Math.max(...ys))+1.2;
const span=Math.max(hi-lo, 6);
function sx(x){return (x-lo)/span*W;}
function sy(y){return H-(y-lo)/span*H;}   // flip y so +y is up
// precompute a coarse odour grid per distinct active-set (fields only change on consumption)
const GN=70;
function fieldFor(active){
  const grid=new Float32Array(GN*GN); let mx=1e-6;
  for(let j=0;j<GN;j++)for(let i=0;i<GN;i++){
    const wx=lo+span*(i+0.5)/GN, wy=lo+span*(j+0.5)/GN; let s=0;
    for(let k=0;k<foods.length;k++){ if(!active[k])continue;
      const dx=wx-foods[k][0], dy=wy-foods[k][1]; s+=Math.exp(-(dx*dx+dy*dy)/sigma); }
    grid[j*GN+i]=s; if(s>mx)mx=s;
  }
  return {grid,mx};
}
const cache={};
function keyOf(a){return a.map(b=>b?1:0).join('');}
function getField(a){const k=keyOf(a); if(!cache[k])cache[k]=fieldFor(a); return cache[k];}
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}

let trail=[];
function draw(fi){
  const F=frames[fi], x=F[0], y=F[1], yaw=F[2], active=F[3];
  g.clearRect(0,0,W,H);
  // odour heatmap
  const fld=getField(active), gr=fld.grid, mx=fld.mx;
  const cw=W/GN, ch=H/GN, isDark=matchMedia('(prefers-color-scheme:dark)').matches ||
      document.documentElement.dataset.theme==='dark';
  for(let j=0;j<GN;j++)for(let i=0;i<GN;i++){
    const v=Math.pow(gr[j*GN+i]/mx,0.75); if(v<0.02)continue;
    const px=i*cw, py=H-(j+1)*ch;
    if(isDark) g.fillStyle=`rgba(255,180,84,${0.30*v})`;
    else       g.fillStyle=`rgba(224,123,0,${0.22*v})`;
    g.fillRect(px-0.5,py-0.5,cw+1,ch+1);
  }
  // trail (rebuild up to fi)
  g.lineWidth=3; g.strokeStyle=css('--trail'); g.globalAlpha=0.55; g.beginPath();
  for(let t=0;t<=fi;t++){ const p=frames[t]; const cx=sx(p[0]), cy=sy(p[1]);
    if(t===0)g.moveTo(cx,cy); else g.lineTo(cx,cy); }
  g.stroke(); g.globalAlpha=1;
  // foods
  for(let k=0;k<foods.length;k++){
    const cx=sx(foods[k][0]), cy=sy(foods[k][1]);
    g.beginPath(); g.arc(cx,cy, active[k]?16:11, 0, 7);
    g.fillStyle=active[k]?css('--food'):css('--eaten'); g.fill();
    if(active[k]){ g.globalAlpha=0.25; g.beginPath(); g.arc(cx,cy,26,0,7); g.fillStyle=css('--food'); g.fill(); g.globalAlpha=1; }
  }
  // agent body (forward = -x body axis => fwd angle = yaw+pi)
  const cx=sx(x), cy=sy(y), fwd=yaw+Math.PI;
  // screen: +x right, +y up (we flipped). heading vector in world:
  const hx=Math.cos(fwd), hy=Math.sin(fwd);
  const ang=Math.atan2(-hy, hx);   // canvas y is down
  g.save(); g.translate(cx,cy); g.rotate(ang);
  g.fillStyle=css('--body');
  g.beginPath(); g.moveTo(15,0); g.lineTo(-11,-9); g.lineTo(-6,0); g.lineTo(-11,9); g.closePath(); g.fill();
  g.restore();
  // HUD
  const eaten=active.filter(a=>!a).length;
  document.getElementById('eaten').textContent=eaten;
  document.getElementById('total').textContent=foods.length;
  document.getElementById('ticks').textContent=(fi*stride*sub).toLocaleString();
}
let fi=0, playing=true, speed=2, acc=0, last=performance.now();
const scrub=document.getElementById('scrub'); scrub.max=frames.length-1;
function loop(now){
  const dt=now-last; last=now;
  if(playing){ acc+=dt*speed/33; while(acc>=1){ fi++; acc--; if(fi>=frames.length){fi=frames.length-1;playing=false;document.getElementById('play').textContent='Replay';break;} } scrub.value=fi; }
  draw(fi); requestAnimationFrame(loop);
}
document.getElementById('play').onclick=()=>{ const b=document.getElementById('play');
  if(fi>=frames.length-1){fi=0;} playing=!playing; b.textContent=playing?'Pause':(fi>=frames.length-1?'Replay':'Play'); };
document.getElementById('speed').onclick=()=>{ speed=speed===2?4:(speed===4?1:2);
  document.getElementById('speed').textContent=speed+'×'; document.getElementById('spd').textContent=speed+'×'; };
scrub.oninput=()=>{ fi=+scrub.value; playing=false; document.getElementById('play').textContent='Play'; draw(fi); };
requestAnimationFrame(loop);
</script>"""

out=HTML.replace("__PAYLOAD__", payload)
p=D/"multi_food_anim.html"
open(p,"w").write(out)
print("wrote", p, "bytes", len(out))

"""Reads dual_forage_traj.json and writes a self-contained HTML canvas animation of the dual-channel agent
foraging FOOD (attractant, amber) while avoiding TOXIN (repellent, red hazard zones). No external assets."""
import json, pathlib
D=pathlib.Path(__file__).parent
data=json.load(open(D/"dual_forage_traj.json"))
payload=json.dumps(data, separators=(",",":"))

HTML = r"""<title>Dual-channel agent — food vs toxin</title>
<style>
:root{
  --bg:#0d1014; --panel:#151a21; --ink:#e8eef4; --muted:#8a96a3; --line:#232b34;
  --accent:#4cc2ff; --trail:#67d0ff; --food:#ffc14d; --eaten:#3a434e; --tox:#ff5470; --body:#f2f6fa;
}
@media (prefers-color-scheme:light){
  :root{ --bg:#f4f7fa; --panel:#ffffff; --ink:#182029; --muted:#5a6675; --line:#e3e8ee;
         --accent:#0a7ea4; --trail:#0a7ea4; --food:#d98a00; --eaten:#c2cbd4; --tox:#d61f43; --body:#182029; }
}
:root[data-theme=dark]{ --bg:#0d1014; --panel:#151a21; --ink:#e8eef4; --muted:#8a96a3; --line:#232b34;
  --accent:#4cc2ff; --trail:#67d0ff; --food:#ffc14d; --eaten:#3a434e; --tox:#ff5470; --body:#f2f6fa; }
:root[data-theme=light]{ --bg:#f4f7fa; --panel:#ffffff; --ink:#182029; --muted:#5a6675; --line:#e3e8ee;
  --accent:#0a7ea4; --trail:#0a7ea4; --food:#d98a00; --eaten:#c2cbd4; --tox:#d61f43; --body:#182029; }
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.5}
.wrap{max-width:860px;margin:0 auto;padding:28px 20px 40px}
h1{font-size:20px;font-weight:650;margin:0 0 2px;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:13.5px;margin:0 0 20px}
.stage{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px;
  box-shadow:0 1px 3px rgba(0,0,0,.2)}
canvas{width:100%;height:auto;display:block;border-radius:8px;background:transparent}
.hud{display:flex;gap:18px;flex-wrap:wrap;align-items:center;margin-top:12px;font-size:13px}
.hud .k{color:var(--muted)}.hud .v{font-variant-numeric:tabular-nums;font-weight:600}
.hud .warn{color:var(--tox)}
.ctl{display:flex;gap:12px;align-items:center;margin-top:12px}
button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:7px 16px;
  font-size:13px;font-weight:600;cursor:pointer}
button:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
input[type=range]{flex:1;accent-color:var(--accent)}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:14px;font-size:12.5px;color:var(--muted)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:middle}
.x{display:inline-block;color:var(--tox);font-weight:800;margin-right:4px}
</style>
<div class="wrap">
  <h1>Dual-channel agent — food vs toxin</h1>
  <p class="sub">Two chemoreceptor channels in one PAULA network, summing at the muscles: FOOD (long-range
  attractant) pulls the agent in to eat; TOXIN (short-range repellent) drives an escape turn. No mode switch —
  attraction and avoidance coexist and compete in the wiring.</p>
  <div class="stage">
    <canvas id="c" width="1640" height="1180"></canvas>
    <div class="hud">
      <span><span class="k">eaten </span><span class="v" id="eaten">0</span><span class="k"> / </span><span class="v" id="total">0</span></span>
      <span class="warn"><span class="k" style="color:inherit;opacity:.8">toxin contacts </span><span class="v" id="hits">0</span></span>
      <span><span class="k">neural ticks </span><span class="v" id="ticks">0</span></span>
      <span><span class="k">speed </span><span class="v" id="spd">2×</span></span>
    </div>
    <div class="ctl">
      <button id="play">Pause</button>
      <input id="scrub" type="range" min="0" max="100" value="0">
      <button id="speed">2×</button>
    </div>
    <div class="legend">
      <span><span class="dot" style="background:var(--food)"></span>food (attractant)</span>
      <span><span class="dot" style="background:var(--eaten)"></span>eaten</span>
      <span><span class="x">✕</span>toxin (repellent)</span>
      <span><span class="dot" style="background:var(--body)"></span>agent</span>
      <span><span class="dot" style="background:var(--trail)"></span>path</span>
    </div>
  </div>
</div>
<script>
const DATA=__PAYLOAD__;
const foods=DATA.foods, toxins=DATA.toxins, frames=DATA.frames, sub=DATA.sub, stride=DATA.stride,
      sigma=DATA.sigma, sigmaTox=DATA.sigma_tox, hitsTotal=DATA.toxin_hits;
const cv=document.getElementById('c'), g=cv.getContext('2d'), W=cv.width, H=cv.height;
let pts=frames.map(f=>[f[0],f[1]]).concat(foods).concat(toxins);
let lo=Math.min(...pts.flat())-1.4, hi=Math.max(...pts.flat())+1.4;
const span=Math.max(hi-lo, 8);
function sx(x){return (x-lo)/span*W;}
function sy(y){return H-(y-lo)/span*H;}
const GN=90;
function gridField(items,active,sig){
  const grid=new Float32Array(GN*GN); let mx=1e-6;
  for(let j=0;j<GN;j++)for(let i=0;i<GN;i++){
    const wx=lo+span*(i+0.5)/GN, wy=lo+span*(j+0.5)/GN; let s=0;
    for(let k=0;k<items.length;k++){ if(active&&!active[k])continue;
      const dx=wx-items[k][0], dy=wy-items[k][1]; s+=Math.exp(-(dx*dx+dy*dy)/sig); }
    grid[j*GN+i]=s; if(s>mx)mx=s;
  }
  return {grid,mx};
}
const toxField=gridField(toxins,null,sigmaTox);   // constant
const foodCache={};
function foodField(active){const key=active.map(b=>b?1:0).join('');
  if(!foodCache[key])foodCache[key]=gridField(foods,active,sigma); return foodCache[key];}
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
function isDark(){return matchMedia('(prefers-color-scheme:dark)').matches ||
  document.documentElement.dataset.theme==='dark';}

function heat(fld, colorFn, gamma, amax){
  const gr=fld.grid, mx=fld.mx, cw=W/GN, ch=H/GN;
  for(let j=0;j<GN;j++)for(let i=0;i<GN;i++){
    const v=Math.pow(gr[j*GN+i]/mx,gamma); if(v<0.03)continue;
    g.fillStyle=colorFn(Math.min(v,1)*amax);
    g.fillRect(i*cw-0.5, H-(j+1)*ch-0.5, cw+1, ch+1);
  }
}
function draw(fi){
  const F=frames[fi], x=F[0], y=F[1], yaw=F[2], active=F[3];
  g.clearRect(0,0,W,H);
  const dk=isDark();
  // toxin hazard field (red), then food field (amber)
  heat(toxField, a=>dk?`rgba(255,84,112,${0.34*a})`:`rgba(214,31,67,${0.20*a})`, 0.9, 1);
  heat(foodField(active), a=>dk?`rgba(255,193,77,${0.30*a})`:`rgba(217,138,0,${0.20*a})`, 0.75, 1);
  // trail
  g.lineWidth=3; g.strokeStyle=css('--trail'); g.globalAlpha=0.6; g.beginPath();
  for(let t=0;t<=fi;t++){const p=frames[t]; const cx=sx(p[0]),cy=sy(p[1]); t?g.lineTo(cx,cy):g.moveTo(cx,cy);}
  g.stroke(); g.globalAlpha=1;
  // toxins (persistent hazard X with keep-out ring)
  for(let k=0;k<toxins.length;k++){const cx=sx(toxins[k][0]), cy=sy(toxins[k][1]);
    g.strokeStyle=css('--tox'); g.globalAlpha=0.4; g.lineWidth=2; g.beginPath(); g.arc(cx,cy,20,0,7); g.stroke(); g.globalAlpha=1;
    g.lineWidth=4; g.beginPath(); g.moveTo(cx-9,cy-9); g.lineTo(cx+9,cy+9); g.moveTo(cx+9,cy-9); g.lineTo(cx-9,cy+9); g.stroke();
  }
  // foods
  for(let k=0;k<foods.length;k++){const cx=sx(foods[k][0]), cy=sy(foods[k][1]);
    g.beginPath(); g.arc(cx,cy, active[k]?15:10, 0, 7); g.fillStyle=active[k]?css('--food'):css('--eaten'); g.fill();
    if(active[k]){g.globalAlpha=0.22; g.beginPath(); g.arc(cx,cy,25,0,7); g.fillStyle=css('--food'); g.fill(); g.globalAlpha=1;}
  }
  // agent
  const cx=sx(x), cy=sy(y), fwd=yaw+Math.PI, ang=Math.atan2(-Math.sin(fwd), Math.cos(fwd));
  g.save(); g.translate(cx,cy); g.rotate(ang); g.fillStyle=css('--body');
  g.beginPath(); g.moveTo(15,0); g.lineTo(-11,-9); g.lineTo(-6,0); g.lineTo(-11,9); g.closePath(); g.fill(); g.restore();
  // HUD
  const eaten=active.filter(a=>!a).length;
  document.getElementById('eaten').textContent=eaten;
  document.getElementById('total').textContent=foods.length;
  document.getElementById('hits').textContent=hitsTotal;
  document.getElementById('ticks').textContent=(fi*stride*sub).toLocaleString();
}
let fi=0, playing=true, speed=2, acc=0, last=performance.now();
const scrub=document.getElementById('scrub'); scrub.max=frames.length-1;
function loop(now){const dt=now-last; last=now;
  if(playing){acc+=dt*speed/33; while(acc>=1){fi++; acc--; if(fi>=frames.length){fi=frames.length-1;playing=false;document.getElementById('play').textContent='Replay';break;}} scrub.value=fi;}
  draw(fi); requestAnimationFrame(loop);}
document.getElementById('play').onclick=()=>{const b=document.getElementById('play');
  if(fi>=frames.length-1)fi=0; playing=!playing; b.textContent=playing?'Pause':(fi>=frames.length-1?'Replay':'Play');};
document.getElementById('speed').onclick=()=>{speed=speed===2?4:(speed===4?1:2);
  document.getElementById('speed').textContent=speed+'×'; document.getElementById('spd').textContent=speed+'×';};
scrub.oninput=()=>{fi=+scrub.value; playing=false; document.getElementById('play').textContent='Play'; draw(fi);};
requestAnimationFrame(loop);
</script>"""

out=HTML.replace("__PAYLOAD__", payload)
p=D/"dual_forage_anim.html"; open(p,"w").write(out)
print("wrote", p, "bytes", len(out))

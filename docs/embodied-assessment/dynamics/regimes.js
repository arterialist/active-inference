/* Small CPU-projected 3D reader. It never generates dynamics or drives neurons. */
(() => {
  const $=s=>document.querySelector(s), canvas=$('#scene'),ctx=canvas.getContext('2d');
  const col={ink:'#24251f',muted:'#5e6056',rule:'#bcbeb0',blue:'#245d79',rust:'#984526',paper:'#f7f5ef'};
  let key='feedback-slow',meta=null,chunk=null,tick=11870,yaw=-.32,tilt=.38,zoom=1,active=false,timer,sequence=0;
  const layers=[
    {name:'Sensory activity',axis:'V(C0), V(C1) · pre-reset',a:r=>r.derived['1'].pre_reset,b:r=>r.derived['2'].pre_reset},
    {name:'Predictive activity',axis:'V(P0), V(P1) · pre-reset',a:r=>r.derived['5'].pre_reset,b:r=>r.derived['6'].pre_reset},
    {name:'Local regulation',axis:'M₀(P0), M₀(P1) · recorded',a:r=>r.state.neurons['5'].M[0],b:r=>r.state.neurons['6'].M[0]},
    {name:'Changing memory',axis:'C0→P0, C0→P1 weights · recorded',a:r=>r.state.neurons['5'].synapses['0'][0],b:r=>r.state.neurons['6'].synapses['0'][0]}
  ];
  const project=(x,y,z)=>{
    const X=x*Math.cos(yaw)-y*Math.sin(yaw),Y=x*Math.sin(yaw)+y*Math.cos(yaw);
    const s=Math.min(canvas.clientWidth/1050,canvas.clientHeight/590)*zoom;
    return [canvas.clientWidth*.55+X*s,canvas.clientHeight*.56+(Y*Math.sin(tilt)-z*Math.cos(tilt))*s];
  };
  const line=(points,color,width=1,dash=[])=>{ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();points.forEach((p,i)=>{const q=project(...p);if(i)ctx.lineTo(...q);else ctx.moveTo(...q);});ctx.stroke();ctx.setLineDash([]);};
  function draw(){
    if(!chunk)return;
    const width=canvas.clientWidth,height=canvas.clientHeight,dpr=devicePixelRatio||1;
    canvas.width=width*dpr;canvas.height=height*dpr;ctx.scale(dpr,dpr);
    const rows=chunk.rows,rel=tick-rows[0].executed_tick,current=rows[rel],stateView=$('#view').value==='state';
    const anchor=[];
    layers.forEach((layer,j)=>{
      const z=(j-1.5)*125,vals=rows.flatMap(r=>[layer.a(r),layer.b(r)]),lo=Math.min(0,...vals),hi=Math.max(.00001,...vals),span=hi-lo;
      line([[-290,-110,z],[290,-110,z],[290,110,z],[-290,110,z],[-290,-110,z]],col.rule,.8);
      const labels=project(-320,0,z);
      ctx.textAlign='right';ctx.fillStyle=col.ink;ctx.font=`600 ${width<600?11:14}px "Avenir Next",Verdana,sans-serif`;ctx.fillText(layer.name,labels[0],labels[1]);
      ctx.font=`${width<600?9:11}px "Avenir Next",Verdana,sans-serif`;ctx.fillStyle=col.muted;ctx.fillText(layer.axis,labels[0],labels[1]+17);
      if(stateView){
        const point=r=>[-270+(layer.a(r)-lo)/span*540,-90+(layer.b(r)-lo)/span*180,z];
        line(rows.map(point),col.rule,.8);line(rows.slice(Math.max(0,rel-40),rel+1).map(point),col.blue,1.8);
        anchor.push(point(current));const p=project(...point(current));ctx.fillStyle=col.ink;ctx.beginPath();ctx.arc(...p,4,0,Math.PI*2);ctx.fill();
        ctx.textAlign='left';ctx.fillStyle=col.muted;ctx.font='10px Menlo,monospace';const p0=project(-290,110,z);ctx.fillText(`x: channel 0 · y: channel 1 · range ${lo.toFixed(3)}…${hi.toFixed(3)}`,p0[0],p0[1]+13);
      }else{
        const valuePoint=(r,k)=>[-270+(r.executed_tick-rows[0].executed_tick)/159*540,k===0?-65:50,z+((k===0?layer.a(r):layer.b(r))-lo)/span*45];
        for(const k of [0,1]){
          const points=[];rows.forEach((r,t)=>{const p=valuePoint(r,k);if(t){const old=points[points.length-1];points.push([p[0],p[1],old[2]]);}points.push(p);});
          line(points,k===0?col.blue:col.rust,1.2);
          const p=project(...valuePoint(current,k));ctx.fillStyle=col.ink;ctx.beginPath();ctx.arc(...p,3,0,Math.PI*2);ctx.fill();
        }
        const x=-270+rel/159*540;line([[x,-100,z],[x,100,z]],col.ink,1,[3,4]);anchor.push([x,100,z]);
        ctx.textAlign='left';ctx.fillStyle=col.muted;ctx.font='10px Menlo,monospace';const p=project(-290,110,z);ctx.fillText(`0…159 ticks · amplitude ${lo.toFixed(3)}…${hi.toFixed(3)}`,p[0],p[1]+13);
      }
    });
    line(anchor,col.muted,1,[3,6]);
    ctx.textAlign='left';ctx.font='12px "Avenir Next",Verdana,sans-serif';ctx.fillStyle=col.muted;
    ctx.fillText(stateView?'Projection of a recorded trajectory; a loop is not proof of an attractor.':'Common time runs left to right; amplitude scales differ and are labeled.',16,25);
    const trial=meta.trials[Math.floor(tick/160)];
    $('#regime-phase').textContent=trial.phase.replaceAll('_',' ')+' · trial '+trial.index;
    const result=trial.paired?'Outcome-driven training: do not score this as recall.':trial.category==='correct_only'?'Cue-only recall: correct prediction only.':trial.category==='silent_or_insufficient'?'Cue-only probe: silent or insufficient prediction.':trial.category==='silence'?'Silent interval.':'Cue-only probe: wrong or ambiguous prediction.';
    $('#regime-output').textContent=`${result} Whole-trial output: P0 ${trial.counts['5']||0}, P1 ${trial.counts['6']||0} spikes. Seed ${meta.manifest.seed}. Full trial shown; future samples remain visible.`;
    $('#reading').textContent=layers.map(l=>`${l.name}: ${l.a(current).toPrecision(5)} / ${l.b(current).toPrecision(5)}`).join('\n');$('#reading').style.whiteSpace='pre-line';
    $('#regime-tick').textContent=`${tick.toLocaleString('en-US')} / ${(meta.ticks-1).toLocaleString('en-US')}`;
    $('#regime-seek').value=tick;
  }
  async function seek(value){
    tick=Math.max(0,Math.min(meta.ticks-1,Math.round(Number(value))));const s=++sequence;
    try{const loaded=await PAULA_DATA.get(key+'/'+Math.floor(tick/160));if(s!==sequence)return;chunk=loaded;draw();$('#regime-status').textContent=active?'Playing recorded ticks':'Paused · actual recording';history.replaceState(null,'',`#${key}/${tick}`);}catch(e){stop();$('#regime-status').textContent=e.message;}
  }
  function stop(){active=false;clearTimeout(timer);$('#regime-play').textContent='Play';}
  async function load(){
    stop();key=$('#recording').value;
    try{meta=await PAULA_DATA.get(key+'/meta');$('#regime-seek').max=meta.ticks-1;
      const events=key==='recall'?[['Retained recall',10900],['Slower cues',11880],['Return to learned timing',14760]]:[['Acquisition',990],['Retained recall',10900],['Reversal',11870],['Late reversal',12360],['Recall test',18260]];
      $('#regime-events').innerHTML=events.map(([label,t])=>`<button data-tick="${t}">${label}</button>`).join('');$('#regime-events').querySelectorAll('button').forEach(b=>b.onclick=()=>{stop();seek(b.dataset.tick);});await seek(key==='recall'?11880:11870);
    }catch(e){$('#regime-status').textContent=e.message;}
  }
  $('#recording').onchange=load;$('#view').onchange=draw;$('#regime-seek').oninput=e=>{stop();seek(e.target.value);};$('#regime-step').onclick=()=>{stop();seek(tick+1);};
  $('#regime-play').onclick=()=>{if(active){stop();$('#regime-status').textContent='Paused · actual recording';return;}active=true;$('#regime-play').textContent='Pause';const next=async()=>{if(!active)return;if(tick>=meta.ticks-1){stop();return;}await seek(tick+1);if(active)timer=setTimeout(next,1000/12);};next();};
  $('#regime-orbit').onclick=()=>{yaw+=.3;draw();};$('#regime-reset').onclick=()=>{yaw=-.32;tilt=.38;zoom=1;draw();};
  const pointers=new Map();let lastDistance=null;
  canvas.onpointerdown=e=>{canvas.setPointerCapture(e.pointerId);pointers.set(e.pointerId,[e.clientX,e.clientY]);};
  canvas.onpointermove=e=>{if(!pointers.has(e.pointerId))return;const old=pointers.get(e.pointerId);pointers.set(e.pointerId,[e.clientX,e.clientY]);if(pointers.size===2){const [a,b]=[...pointers.values()],d=Math.hypot(a[0]-b[0],a[1]-b[1]);if(lastDistance)zoom=Math.max(.55,Math.min(1.5,zoom*d/lastDistance));lastDistance=d;}else{yaw+=(e.clientX-old[0])*.006;tilt=Math.max(.15,Math.min(1.25,tilt+(e.clientY-old[1])*.004));}draw();};
  canvas.onpointerup=canvas.onpointercancel=e=>{pointers.delete(e.pointerId);lastDistance=null;};
  canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(.55,Math.min(1.5,zoom*Math.exp(-e.deltaY*.001)));draw();};
  new ResizeObserver(draw).observe(canvas);load();
})();

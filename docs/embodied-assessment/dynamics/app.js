/* Read-only figures. Every visible state belongs to one completed recorded tick. */
(() => {
  'use strict';
  const $ = s => document.querySelector(s);
  const NAMES = {'1':'C0','2':'C1','3':'U0','4':'U1','5':'P0','6':'P1','7':'E0','8':'E1'};
  const ROLES = {'1':'cue 0','2':'cue 1','3':'outcome 0','4':'outcome 1','5':'prediction 0','6':'prediction 1','7':'comparator 0','8':'comparator 1'};
  const POS = {'1':[48,108],'2':[48,224],'3':[238,30],'4':[238,300],'5':[238,108],'6':[238,224],'7':[460,108],'8':[460,224]};
  const SCENARIOS = {
    credit: {
      title:'When correction strengthens the wrong memory',
      description:'Both networks receive contradictory evidence. A recent wrong spike still earns positive credit on the left. On the right, modulation shortens that credit window, allowing the old association to weaken.',
      keys:['credit-wide','credit-narrow'], start:11870,
      events:[['Before learning',20],['First outcome',990],['Retained recall',10900],['Wrong prediction',11865],['Credit diverges',11870],['Reversal test',18260]]
    },
    feedback: {
      title:'Plasticity can damage the pathway that regulates it',
      description:'The same seed and protocol, with different prediction-terminal adaptation rates. On the left, the export gain crosses zero; its negative packets stop entering PAULA’s positive-information path. On the right, slower adaptation preserves transmission.',
      keys:['feedback-fast','feedback-slow'], start:11879,
      events:[['Before reversal',11700],['First reversal',11865],['P0 export crosses zero',11879],['P1 export crosses zero',12037],['Reversal test',18260]]
    },
    recall: {
      title:'A silent output does not mean the memory was erased',
      description:'The left pane holds a retained, two-tick cue probe as a reference. The right follows the same network through slower cues and a return to two-tick spacing. Both show the same position within their trial; absolute ticks differ. No outcomes or retraining occur during these probes.',
      keys:['recall','recall'], start:11880,
      events:[['Trained spacing · 2',10920],['Slower · 3',11880],['Slower · 4',12840],['Slower · 6',13800],['Return · 2',14760]]
    }
  };
  let scenario='credit', tick=11870, selected='5', meta=[], chunks=[], records=[], running=false, generation=0, timer=null;
  const esc = v => String(v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = (v,d=4) => v == null ? 'none' : Math.abs(v)>0 && Math.abs(v)<.0001 ? v.toExponential(2) : Number(v).toFixed(d);
  const number = n => Number(n).toLocaleString('en-US');
  const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const color = id => Number(id)%2 ? css('--blue') : css('--rust');
  const paneTick = i => scenario==='recall' && i===0 ? 10880+(tick%160) : tick;
  const phaseName = s => ({before:'Naïve probe',train_first:'Learn first association',after_first:'First recall test',train_second:'Learn second association',after_second:'Both-association test',silence:'Silent interval',retention:'Retained recall',reversal_train:'Reversed training',reversal_probe:'Reversal recall test',transfer_period2:'Return to 2-tick cues',transfer_period3:'3-tick cues',transfer_period4:'4-tick cues',transfer_period6:'6-tick cues'}[s] || s.replaceAll('_',' '));
  const labelFor = i => scenario==='recall' ? (i===0 ? 'Trained spacing · reference' : 'Timing transfer · same network') : meta[i].label;

  function stop() { running=false; clearTimeout(timer); $('#play').textContent='Play replay'; }
  function showError(error) { stop(); $('#error').hidden=false; $('#error').textContent=error.message; $('#status').textContent='Recording unavailable'; }
  function changeSelected(id) { selected=String(id); $('#cell').value=selected; render(); history.replaceState(null,'',`#${scenario}/${tick}/${selected}`); }
  async function choose(name, wantedTick) {
    stop(); scenario=name; const s=SCENARIOS[name]; const g=++generation;
    $('#question').textContent=s.title; $('#explanation').textContent=s.description;
    document.querySelectorAll('[data-scenario]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.scenario===name)));
    $('#events').innerHTML=s.events.map(([label,t])=>`<button data-event="${t}">${esc(label)}</button>`).join('');
    $('#events').querySelectorAll('button').forEach(b=>b.onclick=()=>{stop();seek(Number(b.dataset.event));});
    $('#status').textContent='Loading recordings'; $('#error').hidden=true;
    try {
      const loaded=await Promise.all(s.keys.map(k=>PAULA_DATA.get(k+'/meta')));
      if(g!==generation)return;
      meta=loaded; createPanes();
      const max=Math.min(...meta.map(m=>m.ticks))-1;
      for(const el of [$('#seek'),$('#tick-number')]){el.max=max;el.disabled=false;}
      ['play','back','forward'].forEach(id=>$('#'+id).disabled=false);
      await seek(wantedTick ?? s.start);
    } catch(error){if(g===generation)showError(error);}
  }
  async function seek(value) {
    if(!meta.length)return;
    const candidate=Math.max(0,Math.min(Number($('#seek').max),Math.round(Number(value)||0)));
    tick=candidate; const g=++generation;
    $('#status').textContent='Loading tick '+number(tick);
    try {
      const loaded=await Promise.all(meta.map((m,i)=>PAULA_DATA.get(m.key+'/'+Math.floor(paneTick(i)/160))));
      if(g!==generation)return;
      chunks=loaded;
      records=chunks.map((ch,i)=>ch.rows[paneTick(i)-ch.rows[0].executed_tick]);
      if(records.some((r,i)=>!r || r.executed_tick!==paneTick(i)))throw new Error('Tick alignment failed. Replay stopped.');
      render();
      history.replaceState(null,'',`#${scenario}/${tick}/${selected}`);
    }catch(error){if(g===generation)showError(error);}
  }
  async function play() {
    if(running){stop();return;}
    running=true; $('#play').textContent='Pause replay';
    const advance=async()=>{
      if(!running)return;
      if(tick>=Number($('#seek').max)){stop();render();return;}
      await seek(tick+1);
      if(running)timer=setTimeout(advance,1000/Number($('#speed').value));
    };
    advance();
  }
  function createPanes() {
    $('#panes').innerHTML=meta.map((m,i)=>`<article class="pane" aria-label="${esc(labelFor(i))}">
      <div class="pane-heading"><h3>${esc(labelFor(i))}</h3><small id="pane-tick-${i}"></small></div>
      <p class="trial-context" id="context-${i}"></p><div id="network-${i}" class="network-stage"></div>
      <div id="mini-${i}" class="mini-trace"></div>
      <div class="pane-bottom"><div><div id="mechanism-${i}" class="mechanism"></div><div id="credit-${i}" class="credit-strip"></div></div><div><div class="weight-caption">Cue weights · recorded</div><div id="weights-${i}" class="weights"></div></div></div>
      <div id="output-${i}" class="output-result"></div>
    </article>`).join('');
    $('#plots').innerHTML=meta.map((m,i)=>`<div id="plots-${i}" class="plots-pane" aria-label="${esc(labelFor(i))} traces"></div>`).join('');
    $('#inspectors').innerHTML=meta.map((m,i)=>`<div id="inspector-${i}" class="inspector"></div>`).join('');
    const overviewMeta=scenario==='recall' ? [meta[1]] : meta;
    $('#overview').innerHTML=overviewMeta.map((m,i)=>`<div class="overview-row"><span class="overview-name">${esc(m.label)}</span><canvas class="overview-canvas" id="overview-${i}" aria-label="Trials from ${esc(m.label)}. Use the tick slider to seek."></canvas></div>`).join('');
    document.querySelectorAll('.overview-canvas').forEach(c=>c.onclick=e=>{stop();seek((e.offsetX/c.clientWidth)*(Number($('#seek').max)+1));});
    $('#provenance').innerHTML=[...new Map(meta.map(m=>[m.key,m])).values()].map(m=>`<article><h3>${esc(m.label)}</h3><p>Seed ${m.manifest.seed} · ${number(m.ticks)} completed ticks · independent equation audit passed.<br>Variant: <code>${esc(m.manifest.variant)}</code><br>Source: <code>${esc(m.source)}</code><br>Trace SHA-256: <code>${m.trace_sha256}</code><br>Largest membrane residual: <code>${fmt(m.audit.max_equation_residuals.membrane)}</code>. This is an arithmetic check, not a claim that the behavior succeeds.</p></article>`).join('');
  }

  function edgePath(source,target,mod) {
    const [x,y]=POS[source], [X,Y]=POS[target];
    const dx=X-x,dy=Y-y,len=Math.hypot(dx,dy), r=25;
    const start=[x+dx/len*r,y+dy/len*r],end=[X-dx/len*r,Y-dy/len*r];
    if(mod){const offset=source==='7' ? -50 : 50;return `M${x},${y+Math.sign(offset)*24} C${x},${y+offset} ${X},${Y+offset} ${X},${Y+Math.sign(offset)*24}`;}
    if(source==='1'||source==='2')return `M${start} Q${(x+X)/2},${(y+Y)/2+(source==='1'?-40:40)} ${end}`;
    if((source==='3'||source==='4') && target!==String(Number(source)+2))return `M${start} Q${Math.max(x,X)+55},${(y+Y)/2} ${end}`;
    return `M${start} L${end}`;
  }
  function network(i) {
    const m=meta[i], row=records[i], state=row.state;
    const stage=$('#network-'+i);
    const edges=m.config.connections.map(c=>{
      const s=String(c.source_neuron),t=String(c.target_neuron),sid=String(c.target_synapse);
      const terminal=state.neurons[s].terminals[String(c.source_terminal)];
      const w=state.neurons[t].synapses[sid][0];
      const modulation=s==='7'||s==='8';
      const related=s===selected||t===selected;
      const width=modulation?1.5:Math.max(.7,Math.min(2.6,Math.abs(w)*.6));
      return `<path class="edge ${modulation?'modulation':''} ${related?'related':''}" d="${edgePath(s,t,modulation)}" style="stroke-width:${width}" marker-end="url(#${modulation?'mod':w<0?'bar':'arrow'}-${i})"><title>${NAMES[s]} → ${NAMES[t]}.${sid}; ${modulation?'pure modulator':`target weight ${w}; source export ${terminal[0]}`}</title></path>`;
    }).join('');
    stage.innerHTML=`<svg viewBox="0 0 520 322" aria-label="Saved eight-neuron connectivity at tick ${row.executed_tick}">
      <defs><marker id="arrow-${i}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8" fill="none" stroke="currentColor"/></marker><marker id="bar-${i}" viewBox="0 0 8 10" refX="7" refY="5" markerWidth="6" markerHeight="9" orient="auto"><path d="M7 0 L7 10" stroke="currentColor" stroke-width="1.5"/></marker><marker id="mod-${i}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 4 L4 0 L8 4 L4 8 Z" fill="${css('--green')}"/></marker></defs>${edges}
    </svg>`;
    Object.keys(NAMES).forEach(id=>{
      const b=document.createElement('button'),n=state.neurons[id];
      b.textContent=NAMES[id]; b.style.left=POS[id][0]/520*100+'%';b.style.top=POS[id][1]/322*100+'%';b.style.setProperty('--node-color',color(id));
      b.classList.toggle('spike',n.O>0);b.setAttribute('aria-pressed',String(id===selected));
      b.setAttribute('aria-label',`${NAMES[id]}, ${ROLES[id]}, ${n.O>0?'spiking':'not spiking'}, inspect cell`);
      b.dataset.neuron=id;b.onclick=()=>{changeSelected(id);$('#network-'+i).querySelector(`[data-neuron="${id}"]`).focus({preventScroll:true});};stage.append(b);
    });
  }
  function currentMechanism(i) {
    const row=records[i],n=row.state.neurons['5'],d=row.derived['5'];
    const rate=row.state.plasticity_rates['5'].used_multiplier;
    if(scenario==='credit') {
      const delta=d.weight_delta['0'];
      $('#mechanism-'+i).innerHTML=`<strong>P0 · credit for C0</strong><br>Last spike <b>${d.age===null?'none':d.age+' ticks ago'}</b> · window <b>${fmt(n.t_ref,2)}</b><br>Weight change this tick <b class="${delta>0?'bad':'good'}">${delta>0?'+':''}${fmt(delta,6)}</b><br>Plasticity rate <b>×${fmt(rate,1)}</b>`;
      const width=300, x=v=>20+v/12*260,age=d.age;
      $('#credit-'+i).innerHTML=`<svg viewBox="0 0 ${width} 54" role="img" aria-label="P0 temporal credit window ${n.t_ref} ticks; spike age ${age}"><rect x="20" y="19" width="${x(Math.min(12,n.t_ref))-20}" height="12" fill="${css('--green')}" opacity=".18"/><path d="M20 26 H280" stroke="${css('--rule')}"/>${age!==null&&age<=12?`<path d="M${x(age)} 14 V36" stroke="${css('--ink')}" stroke-width="2"/>`:''}<text x="20" y="50">now</text><text x="280" y="50" text-anchor="end">12 ticks into the past</text></svg>`;
    } else if(scenario==='feedback') {
      const q=n.terminals['900'][0];
      $('#mechanism-'+i).innerHTML=`<strong>P0 · prediction export</strong><br>Terminal gain <b class="${q<=0?'bad':''}">${fmt(q,6)}</b><br>${q<=0?'Nonpositive information is not processed.':'Positive export can reach the comparator.'}<br>Basal terminal rate <b>${meta[i].manifest.resolved['5'].parameters.eta_retro.toExponential(0)}</b>`;
      $('#credit-'+i).innerHTML='';
    } else {
      const tr=meta[i].trials[row.trial];const margin=meta[i].audit.probe_response_margins.find(p=>p.trial===tr.index&&p.neuron==='5');
      $('#mechanism-'+i).innerHTML=`<strong>P0 · integration margin</strong><br>Cue pulse spacing <b>${tr.cue_period} ticks</b><br>Trial peak before reset <b>${margin?fmt(margin.maximum_membrane_before_reset):'not a probe'}</b><br>Resting threshold <b>${fmt(n.r)}</b>`;
      $('#credit-'+i).innerHTML='';
    }
    $('#weights-'+i).innerHTML='<span></span><span>P0</span><span>P1</span>'+['0','1'].map(sid=>`<span>C${sid}</span>`+['5','6'].map(nid=>{const w=row.state.neurons[nid].synapses[sid][0];return `<span class="weight-value" style="background:color-mix(in srgb, ${color(nid)} ${Math.min(35,Math.abs(w)*40)}%, transparent)">${fmt(w,3)}</span>`;}).join('')).join('');
  }
  function output(i) {
    const row=records[i],trial=meta[i].trials[row.trial];
    const sofar={'5':0,'6':0};
    for(const r of chunks[i].rows){if(r.executed_tick>row.executed_tick)break;for(const n of ['5','6'])sofar[n]+=r.state.neurons[n].O>0?1:0;}
    const category=trial.category;
    const label=trial.cue===null ? 'No cue presented' : trial.paired ? 'Outcome presented · not a recall test' : ({correct_only:'Correct-only recall',wrong_only:'Wrong-only recall',ambiguous:'Both predictions fired',silent_or_insufficient:'Silent / insufficient recall'}[category]);
    $('#output-'+i).innerHTML=`<div><strong class="${trial.paired?'':category==='correct_only'?'good':category==='silent_or_insufficient'?'':'bad'}">${label}</strong><small>${trial.paired?'Training output may be driven by the outcome.':'Classification uses the entire trial, including future ticks.'}</small></div><div class="counts">P0 ${sofar['5']} · P1 ${sofar['6']}<small>spikes so far</small>P0 ${trial.counts['5']||0} · P1 ${trial.counts['6']||0}<small>whole trial</small></div>`;
  }
  function drawOverview() {
    const list=scenario==='recall'?[meta[1]]:meta;
    list.forEach((m,i)=>{
      const c=$('#overview-'+i);if(!c)return;
      const w=c.clientWidth,h=c.clientHeight,dpr=window.devicePixelRatio||1;c.width=w*dpr;c.height=h*dpr;
      const ctx=c.getContext('2d');ctx.scale(dpr,dpr);
      const X=t=>t/m.ticks*w;
      for(const tr of m.trials){
        const x=X(tr.start),right=X(tr.stop),wide=Math.max(1,right-x-1);
        ctx.fillStyle=tr.paired?css('--wash'):tr.category==='correct_only'?css('--green'):['wrong_only','ambiguous'].includes(tr.category)?css('--rust'):css('--rule');
        ctx.globalAlpha=tr.paired?.7:tr.cue===null?.25:1;ctx.fillRect(x,22,wide,14);ctx.globalAlpha=1;
        if(tr.paired){ctx.save();ctx.beginPath();ctx.rect(x,22,wide,14);ctx.clip();ctx.strokeStyle=css('--muted');ctx.lineWidth=.7;for(let a=x-14;a<right;a+=5){ctx.beginPath();ctx.moveTo(a,36);ctx.lineTo(a+14,22);ctx.stroke();}ctx.restore();}
        if(!tr.paired&&tr.cue!==null){ctx.fillStyle=css('--paper');ctx.font='10px Menlo,monospace';ctx.textAlign='center';if(wide>=8)ctx.fillText(tr.category==='correct_only'?'·':tr.category==='silent_or_insufficient'?'○':'×',x+wide/2,33);}
      }
      const phases=m.trials.filter((tr,j)=>j===0||tr.phase!==m.trials[j-1].phase);
      ctx.font='10px Avenir Next,Verdana,sans-serif';ctx.fillStyle=css('--muted');ctx.textAlign='left';let last=-100;
      for(const tr of phases){const x=X(tr.start);const name=phaseName(tr.phase);const len=ctx.measureText(name).width;if(x>=last+12&&x+len<w){ctx.fillText(name,x,12);last=x+len;}}
      const x=X(tick);ctx.strokeStyle=css('--ink');ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x,17);ctx.lineTo(x,43);ctx.stroke();
    });
  }

  function plot(i, title, series, extent, opts={}) {
    const container=$(opts.target||'#plots-'+i), width=Math.max(290,container.clientWidth||500), height=opts.height||126;
    const left=48,right=10,top=8,bottom=25;
    const row=records[i], tr=meta[i].trials[row.trial], rel=row.executed_tick-tr.start;
    const small=Number($('#zoom').value)===48;
    const lo=small?Math.max(0,Math.min(112,rel-24)):0,hi=small?lo+47:159;
    const data=chunks[i].rows.slice(lo,hi+1);
    const X=t=>left+(t-tr.start-lo)/(hi-lo)*(width-left-right);
    const [min,max]=extent,Y=v=>top+(max-v)/(max-min)*(height-top-bottom);
    let content='';
    for(const v of [min,(min+max)/2,max])content+=`<path class="axis" d="M${left},${Y(v)} H${width-right}"/><text x="${left-7}" y="${Y(v)+4}" text-anchor="end">${opts.yformat?opts.yformat(v):fmt(v,2)}</text>`;
    for(const v of [lo,Math.round((lo+hi)/2),hi])content+=`<text x="${X(v+tr.start)}" y="${height-7}" text-anchor="${v===lo?'start':v===hi?'end':'middle'}">+${v}</text>`;
    for(const s of series){
      const pts=data.map(r=>[X(r.executed_tick),Y(s.value(r))]);
      content+=`<path class="signal ${s.dashed?'dashed':''}" stroke="${s.color||color(selected)}" d="${pts.map(([x,y],j)=>`${j?'H':'M'}${x.toFixed(2)}${j?' V':','}${y.toFixed(2)}`).join(' ')}"/>`;
    }
    if(opts.spikes)for(const r of data)if(r.state.neurons[selected].O>0)content+=`<path d="M${X(r.executed_tick)} ${top} v7" stroke="${color(selected)}" stroke-width="2"/>`;
    content+=`<path class="cursor" d="M${X(row.executed_tick)} ${top} V${height-bottom}"/>`;
    for(const s of series)content+=`<circle cx="${X(row.executed_tick)}" cy="${Y(s.value(row))}" r="2.5" fill="${s.color||color(selected)}"/>`;
    const fig=document.createElement('figure');fig.className='plot';
    fig.innerHTML=`<figcaption class="plot-caption"><span>${title}</span><output>${opts.readout||''}</output></figcaption><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(title)} across ticks ${tr.start+lo} through ${tr.start+hi}">${content}</svg>`;
    fig.querySelector('svg').onclick=e=>{stop();const rect=e.currentTarget.getBoundingClientRect();const v=lo+Math.round(((e.clientX-rect.left)*width/rect.width-left)/(width-left-right)*(hi-lo));seek(tick-tick%160+Math.max(lo,Math.min(hi,v)));};
    container.append(fig);
  }
  function raster(i) {
    const container=$('#plots-'+i),width=Math.max(290,container.clientWidth||500),left=48,right=10,height=142;
    const tr=meta[i].trials[records[i].trial], x=t=>left+(t-tr.start)/159*(width-left-right);
    let marks='';
    for(const id of Object.keys(NAMES)){const y=14+(Number(id)-1)*14;marks+=`<text x="34" y="${y+4}" text-anchor="end">${NAMES[id]}</text><path d="M${left} ${y} H${width-right}" stroke="${css('--rule')}" opacity=".4"/>`;for(const r of chunks[i].rows)if(r.state.neurons[id].O>0)marks+=`<path d="M${x(r.executed_tick)} ${y-4} v8" stroke="${color(id)}" stroke-width="1.4"/>`;}
    marks+=`<path class="cursor" d="M${x(records[i].executed_tick)} 6 V119"/><text x="${left}" y="137">+0</text><text x="${width-right}" y="137" text-anchor="end">+159 ticks from trial start</text>`;
    const fig=document.createElement('figure');fig.className='plot';fig.innerHTML=`<figcaption class="plot-caption"><span>All eight cells · spike raster</span><output>${esc(labelFor(i))}</output></figcaption><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="All recorded spikes this trial">${marks}</svg>`;
    fig.querySelector('svg').onclick=e=>{stop();const rect=e.currentTarget.getBoundingClientRect();seek(tick-tick%160+Math.max(0,Math.min(159,Math.round(((e.clientX-rect.left)*width/rect.width-left)/(width-left-right)*159))));};container.append(fig);
  }
  function drawPlots() {
    if(!records.length)return;
    const all=chunks.flatMap(ch=>ch.rows);
    const bounds=fn=>{const values=all.map(fn);let lo=Math.min(0,...values),hi=Math.max(.01,...values);const pad=(hi-lo)*.08;return [lo-pad,hi+pad];};
    const vBounds=bounds(r=>r.derived[selected].pre_reset);
    vBounds[1]=Math.max(vBounds[1],...all.map(r=>r.derived[selected].threshold*1.04));
    const wBounds=bounds(r=>r.state.neurons[selected].synapses['0'][0]);
    const w1=all.map(r=>r.state.neurons[selected].synapses['1'][0]);wBounds[0]=Math.min(wBounds[0],...w1);wBounds[1]=Math.max(wBounds[1],...w1)*1.02;
    const qBounds=bounds(r=>r.state.neurons[selected].terminals['900'][0]);
    const modBounds=bounds(r=>r.state.neurons[selected].M[0]);
    meta.forEach((m,i)=>{
      $('#plots-'+i).innerHTML='';raster(i);
      const n=records[i].state.neurons[selected],d=records[i].derived[selected];
      $('#mini-'+i).innerHTML='';
      plot(i,`${NAMES[selected]} voltage · before reset (derived)`,[{value:r=>r.derived[selected].pre_reset},{value:r=>r.derived[selected].threshold,dashed:true,color:css('--muted')}],vBounds,{spikes:true,height:100,target:'#mini-'+i,readout:`V ${fmt(d.pre_reset)} / θ ${fmt(d.threshold)}`});
      plot(i,`${NAMES[selected]} modulator · M₀`,[{value:r=>r.state.neurons[selected].M[0],color:css('--green')}],modBounds,{readout:`${fmt(n.M[0])} · rate ×${fmt(records[i].state.plasticity_rates[selected].used_multiplier,1)}`});
      plot(i,`${NAMES[selected]} input weights · 0 solid / 1 dashed`,[{value:r=>r.state.neurons[selected].synapses['0'][0],color:css('--blue')},{value:r=>r.state.neurons[selected].synapses['1'][0],color:css('--rust'),dashed:true}],wBounds,{readout:`${fmt(n.synapses['0'][0])} / ${fmt(n.synapses['1'][0])}`});
      plot(i,`${NAMES[selected]} terminal 900 · export gain`,[{value:r=>r.state.neurons[selected].terminals['900'][0],color:css('--ink')}],qBounds,{readout:fmt(n.terminals['900'][0],6)});
    });
  }
  function inspector(i) {
    const el=$('#inspector-'+i),open=[...el.querySelectorAll('details')].map(d=>d.open);
    const row=records[i],n=row.state.neurons[selected],d=row.derived[selected];
    const values=[['Recorded S (after reset)',n.S],['Reconstructed V',d.pre_reset],['Spike output O',n.O],['Resting threshold r',n.r],['Refractory threshold b',n.b],['Credit window t_ref',n.t_ref],['Spike age (ticks)',d.age],['Firing average',n.F_avg],['Modulator M₀',n.M[0]],['Rate used this tick',row.state.plasticity_rates[selected].used_multiplier],['Dendritic arrivals',d.arrivals.length],['Queued dendritic events',n.dendritic_queue.length]];
    el.innerHTML=`<h3>${NAMES[selected]} · ${esc(labelFor(i))}</h3><dl class="values">${values.map(([k,v])=>`<div><dt>${k}</dt><dd>${fmt(v,6)}</dd></div>`).join('')}</dl>
      <details ${open[0]?'open':''}><summary>Synapses &amp; this tick’s input</summary><table class="synapse-table"><thead><tr><th>Port</th><th>Weight</th><th>Δ weight</th><th>Input info</th><th>Input M₀</th></tr></thead><tbody>${Object.entries(n.synapses).map(([sid,v])=>`<tr><td>${sid}</td><td>${fmt(v[0],6)}</td><td>${fmt(d.weight_delta[sid],6)}</td><td>${fmt(row.delivered_inputs[selected][Number(sid)][0],6)}</td><td>${fmt(row.delivered_inputs[selected][Number(sid)][2],6)}</td></tr>`).join('')}</tbody></table></details>
      <details ${open[1]?'open':''}><summary>Exact cell record &amp; parameters</summary><pre>${esc(JSON.stringify({executed_tick:row.executed_tick,recorded:n,derived:d,plasticity_rate:row.state.plasticity_rates[selected],parameters:meta[i].manifest.resolved[selected]},null,2))}</pre></details>
      <details ${open[2]?'open':''}><summary>Full tick · all cells, inputs &amp; queues</summary><pre>${esc(JSON.stringify(row,null,2))}</pre><button class="download-tick">Download this tick as JSON</button></details>`;
    el.querySelector('.download-tick').onclick=()=>{
      const blob=new Blob([JSON.stringify({source:meta[i].source,trace_sha256:meta[i].trace_sha256,...row},null,2)],{type:'application/json'});
      const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`paula-${meta[i].key}-tick-${row.executed_tick}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
    };
  }
  function render() {
    if(!records.length)return;
    $('#seek').value=tick;$('#tick-number').value=tick;$('#seek-label').textContent=`${number(tick)} / ${number(Number($('#seek').max))}`;
    $('#status').textContent=(running?'Playing':'Paused')+' · recorded ticks';$('#selected-name').textContent=NAMES[selected];
    document.querySelectorAll('[data-event]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.event)===tick)));
    meta.forEach((m,i)=>{
      const row=records[i],tr=m.trials[row.trial];
      $('#pane-tick-'+i).textContent=`seed ${m.manifest.seed} · tick ${number(row.executed_tick)}`;
      $('#context-'+i).textContent=`${phaseName(tr.phase)} · ${tr.cue===null?'no cue':`C${tr.cue} → expected U${tr.outcome}`} · ${tr.paired?'outcome presented':'cue only'} · trial ${tr.index}`;
      network(i);currentMechanism(i);output(i);inspector(i);
    });
    drawOverview();drawPlots();
    $('#back').disabled=tick===0;$('#forward').disabled=tick>=Number($('#seek').max);
  }
  document.querySelectorAll('[data-scenario]').forEach(b=>b.onclick=()=>choose(b.dataset.scenario));
  $('#play').onclick=play;$('#back').onclick=()=>{stop();seek(tick-1);};$('#forward').onclick=()=>{stop();seek(tick+1);};
  $('#tick-number').onchange=e=>{stop();seek(e.target.value);};$('#seek').oninput=e=>{stop();seek(e.target.value);};
  $('#cell').onchange=e=>changeSelected(e.target.value);$('#zoom').onchange=drawPlots;
  document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA','BUTTON','SUMMARY'].includes(e.target.tagName))return;if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();stop();seek(tick+(e.key==='ArrowLeft'?-1:1));}if(e.code==='Space'){e.preventDefault();play();}});
  let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{drawOverview();drawPlots();},100);});
  const hash=location.hash.slice(1).split('/');if(NAMES[hash[2]]){selected=hash[2];$('#cell').value=selected;}
  choose(SCENARIOS[hash[0]]?hash[0]:'credit',/^\d+$/.test(hash[1]||'')?Number(hash[1]):undefined);
  // Diagnostic projection only. It cannot inject state into PAULA.
  window.PAULA_REPLAY={snapshot:()=>({scenario,tick,selected,running,recordedTicks:records.map(r=>r.executed_tick),cache:PAULA_DATA.cacheSize()})};
})();

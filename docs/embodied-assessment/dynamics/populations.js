"use strict";
(() => {
  const $ = id => document.getElementById(id);
  let meta, cells, tick = 1968, playing = false, angle = -.16, tilt = -.12, zoom = 1;
  let selected = 193, points = [], projected = [], lastFrame = 0, accumulator = 0, loadId = 0, soundEnabled = false;
  const centers = {
    vision: [-2.4, 0, 0], touch: [2.4, 0, 0],
    visual_core: [-2.4, 1.9, 0], tactile_core: [2.4, 1.9, 0],
    visual_inhibition: [-3.8, 1.9, .3], tactile_inhibition: [3.8, 1.9, .3],
    connector: [0, 3.5, 0], upper_core: [0, 5.2, 0], upper_inhibition: [1.8, 5.2, .3],
    mismatch_candidate: [4.5, 4.2, 0], activity_regulator: [-4.5, 4.2, 0]
  };
  const short = {vision:"Visual input", touch:"Tactile input", visual_core:"Visual population", tactile_core:"Tactile population", connector:"Neural interface", upper_core:"Upper population", mismatch_candidate:"Mismatch candidate", activity_regulator:"Activity regulator"};
  const waveGroups = ["vision", "touch", "visual_core", "tactile_core", "connector", "upper_core", "mismatch_candidate", "activity_regulator"];
  const css = getComputedStyle(document.documentElement);
  const ink = css.getPropertyValue("--ink").trim(), blue = css.getPropertyValue("--blue").trim(), rust = css.getPropertyValue("--rust").trim();
  const val = (t, nid, field) => cells[(t * meta.size + nid-1)*5 + field];
  const phaseNames = {before_learning:"Before learning: a visual fragment", before_familiar_pair:"Before learning: the future familiar pair", before_conflicting_pair:"Before learning: the alternative pairing", paired_experience:"Experience: both inputs together", silent_interval:"Silence: no external input", partial_recall:"Recall test: touch is absent", familiar_pair:"Test the familiar pairing", conflicting_pair:"Test a conflicting pairing"};
  function fit(canvas) {
    const r = canvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
    const w = Math.max(1, r.width), h = Math.max(1, r.height);
    if (canvas.width !== Math.round(w*dpr) || canvas.height !== Math.round(h*dpr)) {canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
    const ctx = canvas.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
    return {ctx,w,h};
  }
  function project(p,w,h) {
    const x = p[0]*Math.cos(angle) + p[2]*Math.sin(angle);
    const z = -p[0]*Math.sin(angle) + p[2]*Math.cos(angle);
    const y = (p[1]-2.65)*Math.cos(tilt)-z*Math.sin(tilt);
    const depth = z*Math.cos(tilt)+(p[1]-2.65)*Math.sin(tilt);
    const scale = Math.min(w/12.2,h/7.7)*zoom*(12/(12+depth));
    return [w/2+x*scale,h/2-y*scale,scale,depth];
  }
  function line(ctx,a,b,color,width=1,dashed=false) {
    ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dashed?[4,4]:[]);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();ctx.setLineDash([]);
  }
  function makePoints() {
    points=[];
    for(const [g,ids] of Object.entries(meta.groups)) {
      const c=centers[g];
      ids.forEach((nid,i)=>{
        let offset;
        if(g==="vision"||g==="touch") {offset=[(i%12-5.5)*.105,(Math.floor(i/12)-3.5)*.09,0];}
        else {
          const fraction=(i+.5)/ids.length, phi=i*2.3999632297, y=1-2*fraction;
          const radius=(g.includes("inhibition")?.39:g.includes("core")?.82:.6);
          const shell=Math.sqrt(Math.max(0,1-y*y));
          offset=[Math.cos(phi)*shell*radius,y*radius*.64,Math.sin(phi)*shell*radius*.64];
        }
        points.push({nid,g,p:c.map((v,j)=>v+offset[j])});
      });
    }
  }
  function drawScene() {
    const {ctx,w,h}=fit($("population-scene")); if(!meta)return;
    const links=[["vision","visual_core"],["touch","tactile_core"],["visual_core","tactile_core"],["visual_core","connector"],["tactile_core","connector"],["connector","upper_core"],["upper_core","visual_core"],["upper_core","tactile_core"],["mismatch_candidate","upper_core"],["activity_regulator","visual_core"]];
    for(const [a,b] of links){
      const cut=meta.mode==="ascending_cut"&&a==="connector"&&b==="upper_core" || meta.mode==="recurrence_cut"&&a==="visual_core"&&b==="tactile_core";
      line(ctx,project(centers[a],w,h),project(centers[b],w,h),cut?rust:"#b9b9ac",cut?1.7:1,a.includes("candidate")||a.includes("regulator")||cut);
      if(cut){const c=project(centers[a].map((v,i)=>(v+centers[b][i])/2),w,h);ctx.font="12px Verdana";ctx.fillStyle=rust;ctx.fillText("CUT",c[0]+8,c[1]);}
    }
    projected=points.map(p=>({...p,screen:project(p.p,w,h)})).sort((a,b)=>b.screen[3]-a.screen[3]);
    if(selected){
      const byId=new Map(projected.map(p=>[p.nid,p]));
      for(const [src,tgt,,,present] of meta.edges){if(present&&(src===selected||tgt===selected))line(ctx,byId.get(src).screen,byId.get(tgt).screen,src===selected?"#98452665":"#245d7965",.8);}
    }
    for(const p of projected){
      const s=val(tick,p.nid,1)>0,f=val(tick,p.nid,2),mod=Math.max(val(tick,p.nid,3),val(tick,p.nid,4));
      const radius=(s?3.4:2.1)*Math.max(.7,p.screen[2]/60);
      ctx.globalAlpha=s?1:.25+Math.min(.7,f*3.0+mod*2);
      ctx.fillStyle=s?ink:mod>.015?rust:f>.005?blue:"#99998c";
      ctx.beginPath();ctx.arc(p.screen[0],p.screen[1],radius,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;
      if(p.nid===selected){ctx.strokeStyle=ink;ctx.lineWidth=1.3;ctx.beginPath();ctx.arc(p.screen[0],p.screen[1],radius+4,0,Math.PI*2);ctx.stroke();}
    }
    ctx.textAlign="center";
    for(const [g,c]of Object.entries(centers)){
      const center=project(c,w,h),inhib=g.includes("inhibition");
      const y=center[1]+center[2]*(inhib?.58:g.includes("core")?.78:.57);
      ctx.fillStyle=ink;ctx.font=`${inhib?10:12}px Verdana`;
      ctx.fillText(inhib?"inhibition":short[g],center[0],y);
      if(!inhib){const count=meta.groups[g].filter(id=>val(tick,id,1)>0).length;ctx.font="10px Menlo,monospace";ctx.fillStyle=count?blue:"#66665b";ctx.fillText(`${count} / ${meta.groups[g].length} firing`,center[0],y+16);}
    }
    ctx.textAlign="left";
  }
  function trialNow(){return meta.trials.find(tr=>tick>=tr.start&&tick<tr.stop);}
  function drawStimulus(role,tr){
    const {ctx,w,h}=fit($("stimulus-"+role));
    if(meta.media){
      const m=meta.media,rel=tick-tr.start,enabled=role==="vision"?tr.visual_enabled:tr.audio_enabled;
      const i=role==="vision"?(tr.still?Math.min(120,m.clip_ticks-1):rel%m.clip_ticks):(rel+tr.audio_shift)%m.clip_ticks;
      const values=role==="vision"?m.visual_features[i]:m.auditory_features[i];
      const side=Math.min(w/12,h/8);
      values.forEach((v,j)=>{ctx.fillStyle=enabled?`rgb(${Math.round(245-v*210)},${Math.round(245-v*210)},${Math.round(237-v*202)})`:"#e4e3d8";ctx.fillRect(j%12*side,Math.floor(j/12)*side,side-1,side-1);});
      $("stimulus-"+role+"-label").textContent=enabled?(role==="vision"?"12 × 8 darkness receptors":"32 bands × 3 sensitivities"):"WITHHELD from the network";
      return;
    }
    const ids=meta.groups[role],active=new Set(tr.active_ids),rel=tick-tr.start,pulse=rel>=8&&rel<56&&rel%4===0;
    const cols=12,side=Math.min(w/cols,h/Math.ceil(ids.length/cols));
    ids.forEach((id,i)=>{ctx.fillStyle=active.has(id)&&pulse?ink:"#e4e3d8";ctx.fillRect((i%cols)*side,Math.floor(i/cols)*side,side-2,side-2);if(active.has(id)){ctx.strokeStyle=blue;ctx.strokeRect((i%cols)*side+.5,Math.floor(i/cols)*side+.5,side-3,side-3);}});
    const n=ids.filter(id=>active.has(id)).length;
    $("stimulus-"+role+"-label").textContent=n?`${n} features · ${rel<8?"not yet on":rel>=56?"withdrawn":pulse?"pulse now":"between pulses"}`:"No input";
  }
  function updateReading(){
    const tr=trialNow(),rel=tick-tr.start;
    if(meta.media){updateMediaReading(tr,rel);return;}
    $("population-phase").textContent=phaseNames[tr.phase];
    const text={before_learning:"Half a visual pattern is presented before paired experience. This response is the baseline, not learned recall.",before_familiar_pair:"Both patterns are presented before repeated pairing. The supervisor must eventually distinguish a learned expectation from an initial wiring preference.",before_conflicting_pair:"This pairing will not be the repeated pairing. Its input count and pulse timing match the other pair.",paired_experience:"Visual and tactile patterns arrive together. Their neural populations interact, and native local plasticity changes incoming weights. The host does not supply an association label.",silent_interval:"Both external inputs have ended. This tests whether activity continues without sensory forcing. Quiet populations are a result, not a hidden pause.",partial_recall:"Only half of the visual pattern is present. Tactile sensors receive nothing. A learned cross-sensory completion would have to activate the tactile population through the network.",familiar_pair:"The previously paired patterns arrive together again. Compare the mismatch population with the equally strong conflicting pair.",conflicting_pair:"The visual pattern is paired with the other tactile pattern. Same feature count, same pulse timing. More supervisor activity alone would not establish learned surprise without the before-learning comparison."};
    $("population-explanation").textContent=text[tr.phase];
    const response=meta.summary.trial_responses[tr.index];
    const rate=response.rates.tactile_core;
    $("population-outcome").textContent=tr.phase==="partial_recall"?`Measured tactile response: ${(rate*100).toFixed(2)}% firing/tick. ${rate===0?"No tactile completion in this trial.":"Activity present; learned specificity requires the controls."}`:rel>=56?"Input withdrawn. Watch the remaining activity.":`Trial ${tr.index+1} / ${meta.trials.length} · ${rel} ticks into trial`;
    for(const role of ["vision","touch"])drawStimulus(role,tr);
    $("population-seek").value=tick;$("population-tick").textContent=`${tick.toLocaleString()} / ${(meta.shape[0]-1).toLocaleString()}`;
    const p=points.find(p=>p.nid===selected);
    $("population-details").textContent=p?`${meta.labels[p.g]}\nS       ${val(tick,selected,0).toFixed(6)}\nspike   ${val(tick,selected,1).toFixed(0)}\nF_avg   ${val(tick,selected,2).toFixed(6)}\nM0      ${val(tick,selected,3).toFixed(6)}\nM1      ${val(tick,selected,4).toFixed(6)}`:"Select a neuron";
  }
  function updateMediaReading(tr,rel){
    const names={silent_video_before:"Silent video, before pairing",audio_reference_before:"Sound alone, before pairing",audiovisual_experience:"Learning exposure: video and sound",media_silence:"No image. No sound.",silent_video_after:"Silent video, after pairing",silent_image_after:"Still image, no soundtrack",audio_reference_after:"Sound alone: reference response"};
    const explanations={silent_video_before:"The untrained population sees the movie without its soundtrack. Auditory activity here is an initial-wiring baseline, not memory.",audio_reference_before:"The soundtrack drives the auditory receptors while visual input is absent. This reveals the population response to the actual recording.",audiovisual_experience:"Pixels and sound measurements enter separate sensory neurons. Their populations share recurrent connections, interfaces and regulation. Local adaptation remains active.",media_silence:"All sensory drive is absent. This is an actual neural simulation interval, not a paused display.",silent_video_after:"The same movie returns, with every audio input withheld. Compare the auditory activity with the pre-pairing baseline and sound-alone reference.",silent_image_after:"A frame from two seconds into the movie is held still. No audio reaches the network. This tests a still image separately from the movie’s temporal cues.",audio_reference_after:"Only the actual soundtrack is supplied. The resulting auditory pattern is a reference for comparison, not a decoder that generates a bark."};
    $("population-phase").textContent=names[tr.phase];$("population-explanation").textContent=explanations[tr.phase];
    const response=meta.summary.trial_responses[tr.index],n=response.spikes.tactile_core;
    $("population-outcome").textContent=`Auditory population: ${n.toLocaleString()} spikes across this trial. ${tr.audio_enabled?"Sound was supplied.":"Sound was withheld. Activity alone is not learned recall."}`;
    for(const role of["vision","touch"])drawStimulus(role,tr);
    $("population-seek").value=tick;$("population-tick").textContent=`${tick.toLocaleString()} / ${(meta.shape[0]-1).toLocaleString()}`;
    const p=points.find(p=>p.nid===selected);$("population-details").textContent=p?`${meta.labels[p.g]}\nS       ${val(tick,selected,0).toFixed(6)}\nspike   ${val(tick,selected,1).toFixed(0)}\nF_avg   ${val(tick,selected,2).toFixed(6)}\nM0      ${val(tick,selected,3).toFixed(6)}\nM1      ${val(tick,selected,4).toFixed(6)}`:"Select a neuron";
    syncMedia(tr,rel);
  }
  function syncMedia(tr,rel){
    const m=meta.media,v=$("source-video"),a=$("source-audio"),rate=Number($("population-speed").value)/m.ticks_per_second;
    const vt=(tr.still?Math.min(120,m.clip_ticks-1):rel%m.clip_ticks)/m.ticks_per_second;
    const at=((rel+tr.audio_shift)%m.clip_ticks)/m.ticks_per_second;
    const vtime=Math.min(vt,m.clip_ticks/m.ticks_per_second-.02),atime=Math.min(at,m.clip_ticks/m.ticks_per_second-.02);
    const scrub=(element,time,shouldPlay)=>{element.playbackRate=rate;if(Number.isFinite(element.duration)&&Math.abs(element.currentTime-time)>(playing?.12:.001))element.currentTime=time;if(shouldPlay){if(element.paused)element.play().catch(()=>{});}else element.pause();};
    scrub(v,vtime,playing&&tr.visual_enabled&&!tr.still);scrub(a,atime,playing&&tr.audio_enabled&&soundEnabled);
    v.style.opacity=tr.visual_enabled?"1":".25";
    $("source-visibility").textContent=tr.visual_enabled?(tr.still?"Still source frame presented to visual receptors":"Video is supplied to visual receptors, with 3 ticks of encoder latency"):"Picture shown for context only; visual input is WITHHELD";
    $("source-condition").textContent=tr.audio_enabled?(tr.audio_shift?"Soundtrack shifted against the picture":"Original soundtrack supplied"):"Audio withheld from the network";
    $("source-state").textContent=`Movie ${vtime.toFixed(2)} s · audio ${atime.toFixed(2)} s. ${tr.audio_enabled?"Auditory responses can be directly driven by sound.":"Any auditory-population firing must come through internal pathways."}`;
    const {ctx,w,h}=fit($("source-spectrum"));
    const step=Math.max(1,Math.floor(m.clip_ticks/w));
    for(let t=0;t<m.clip_ticks;t+=step){for(let band=0;band<32;band++){const level=m.auditory_features[t][band*3];ctx.fillStyle=`rgba(36,93,121,${.08+.82*level})`;ctx.fillRect(t/m.clip_ticks*w,h-(band+1)*h/32,Math.ceil(step/m.clip_ticks*w),h/32+1);}}
    line(ctx,[atime*m.ticks_per_second/m.clip_ticks*w,0],[atime*m.ticks_per_second/m.clip_ticks*w,h],tr.audio_enabled?ink:rust,2,!tr.audio_enabled);
  }
  function drawWaves(){
    const start=Math.max(0,tick-191),stop=Math.max(192,tick+1);
    for(const g of waveGroups){
      const {ctx,w,h}=fit($("wave-"+g));ctx.strokeStyle="#c4c4b6";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,h-4);ctx.lineTo(w,h-4);ctx.stroke();
      ctx.strokeStyle=g.includes("candidate")||g.includes("regulator")?rust:blue;ctx.lineWidth=1.5;ctx.beginPath();
      for(let t=start;t<=tick;t++){const x=(t-start)/(stop-start-1)*w,y=h-4-meta.rates[g][t]*(h-8);t===start?ctx.moveTo(x,y):ctx.lineTo(x,y);}ctx.stroke();
      const x=(tick-start)/(stop-start-1)*w;line(ctx,[x,0],[x,h],ink);$("rate-"+g).textContent=(meta.rates[g][tick]*100).toFixed(1)+"%";
    }
  }
  function render(){if(!meta||!cells)return;drawScene();updateReading();drawWaves();$("population-status").textContent=`${meta.size.toLocaleString()} neurons · ${playing?"recorded playback":"paused"}`;for(const button of $("population-events").children)button.setAttribute("aria-pressed",String(button.dataset.phase===trialNow().phase));}
  function pause(){playing=false;$("population-play").textContent="Play";$("source-video").pause();$("source-audio").pause();if(meta)$("population-status").textContent=`${meta.size.toLocaleString()} neurons · paused`;}
  function seek(t){if(!meta)return;tick=Math.max(0,Math.min(meta.shape[0]-1,Math.round(t)));render();}
  function buildGuide(){
    $("population-events").replaceChildren();
    const wanted=meta.media?[["silent_video_before","Before pairing"],["audiovisual_experience","Video + sound"],["media_silence","Withdraw both"],["silent_video_after","Silent video"],["silent_image_after","Still image"],["audio_reference_after","Sound alone"]]:[["before_learning","Before pairing"],["paired_experience","Paired experience"],["silent_interval","Withdraw input"],["partial_recall","Partial-cue recall"],["familiar_pair","Familiar pair"],["conflicting_pair","Conflicting pair"]];
    for(const[phase,label]of wanted){const tr=meta.trials.find(tr=>tr.phase===phase),b=document.createElement("button");b.textContent=label;b.dataset.phase=phase;b.onclick=()=>{pause();seek(tr.start+40);};$("population-events").append(b);}
    $("population-waves").replaceChildren();
    for(const g of waveGroups){const row=document.createElement("div");row.className="activity-row";const label=document.createElement("label");label.textContent=short[g];const c=document.createElement("canvas");c.id="wave-"+g;c.setAttribute("aria-label",short[g]+", fraction firing each tick");const o=document.createElement("output");o.id="rate-"+g;row.append(label,c,o);$("population-waves").append(row);}
  }
  function findings(){
    const s=meta.summary,p=document.createElement("p");
    p.textContent=`${s.changed_information_weights.toLocaleString()} incoming information weights changed. This alone does not establish memory. ${s.nonpositive_terminal_info} ordinary output terminals ended nonpositive.`;
    const table=document.createElement("table");table.className="population-results";
    const head=document.createElement("tr");for(const title of["Probe","Observation"]){const th=document.createElement("th");th.textContent=title;head.append(th);}table.append(head);
    if(meta.media){
      for(const match of s.auditory_reinstatement){const tr=document.createElement("tr");for(const value of[match.phase.replaceAll("_"," "),`Centered auditory-pattern similarity: ${match.centered_auditory_trace_similarity===null?"undefined (no temporal variation)":match.centered_auditory_trace_similarity.toFixed(4)}; shifted reference: ${match.half_clip_shift_similarity===null?"undefined":match.half_clip_shift_similarity.toFixed(4)}`]){const td=document.createElement("td");td.textContent=value;tr.append(td);}table.append(tr);}
      $("population-findings").replaceChildren(p,table);$("population-provenance").textContent=`${meta.raw_directory}. Seed ${meta.seed}, ${meta.size} neurons, ${meta.shape[0]} ticks at ${meta.media.ticks_per_second} ticks per source second. ${meta.media.encoders} Source SHA-256: ${meta.media.source_sha256}. ${meta.media.confounds}`;return;
    }
    const silent=s.trial_responses.filter(r=>r.phase==="silent_interval").every(r=>Object.values(r.rates).every(v=>v===0));
    const tactile=s.trial_responses.filter(r=>r.phase==="partial_recall").map(r=>r.rates.tactile_core);
    for(const[a,b]of[["Persistence in silence",silent?"No population firing in the sampled silent windows":"Some activity remains; inspect its duration"],["Missing tactile response",tactile.every(v=>v===0)?"Absent in both partial-cue trials":"Activity present; compare initial-weight control"],["Conflict sensitivity",`${(s.mismatch_conflict_minus_familiar*100).toFixed(4)} percentage points`],["Same contrast before pairing",`${(s.mismatch_contrast_before_learning*100).toFixed(4)} percentage points`]]){const tr=document.createElement("tr");for(const x of[a,b]){const td=document.createElement("td");td.textContent=x;tr.append(td);}table.append(tr);}
    $("population-findings").replaceChildren(p,table);
    $("population-provenance").textContent=`${meta.raw_directory}. Seed ${meta.seed}, ${meta.size} neurons, ${meta.shape[0]} simulator ticks. Condition: ${meta.mode}. The condition selector uses the same input protocol and seed. Source hashes are in manifest.json. Main-run weights are in the raw chunks.`;
  }
  async function load(key){
    const thisLoad=++loadId;pause();meta=null;cells=null;$("population-play").disabled=true;$("population-step").disabled=true;$("population-error").hidden=true;$("population-status").textContent="Loading recorded neurons…";
    try{
      await new Promise((resolve,reject)=>{const script=document.createElement("script");script.src=`data/population-${key}.js`;script.onload=()=>{script.remove();resolve();};script.onerror=()=>{script.remove();reject(new Error("The recording could not be loaded. Reload, or select another condition."));};document.head.append(script);});
      if(thisLoad!==loadId)return;
      const loadedMeta=window.POPULATION_META,packed=window.POPULATION_RECORDING;
      const compressed=Uint8Array.from(atob(packed),c=>c.charCodeAt(0));
      const buffer=await new Response(new Blob([compressed]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
      if(thisLoad!==loadId)return;
      meta=loadedMeta;cells=new Float32Array(buffer);window.POPULATION_RECORDING=null;window.POPULATION_META=null;
      if(cells.length!==meta.shape.reduce((a,b)=>a*b,1))throw new Error("Recording dimensions do not match the data.");
      $("media-source").hidden=!meta.media;
      short.touch=meta.media?"Auditory input":"Tactile input";short.tactile_core=meta.media?"Auditory population":"Tactile population";
      document.querySelectorAll(".stimulus-pair h4")[1].textContent=short.touch;
      document.querySelector(".population-intro h2").textContent=meta.media?"Does the sight bring back the sound?":"Does the missing sensation come back?";
      document.querySelector(".population-intro p").textContent=meta.media?"Real video and sound enter 1,152 PAULA neurons. No pretrained recognition. This is an experiment, not demonstrated bark recall.":"1,152 PAULA neurons in a two-sensory-channel preparation. The point cloud is a circuit layout, not anatomy or EEG.";
      $("stimulus-touch").setAttribute("aria-label",meta.media?"Actual auditory receptor input":"Actual tactile feature input");
      document.querySelector(".stimulus-pair").nextElementSibling.textContent=meta.media?"Actual low-level transduction. No pretrained recognition, semantic labels or embeddings enter the network.":"These are binary feature masks, not recognized shapes. Dark tiles mark this tick’s external pulse; outlines show the presented pattern.";
      if(meta.media){$("source-video").src=meta.media.url;$("source-audio").src=meta.media.url;tick=meta.trials.find(tr=>tr.phase==="silent_video_after").start+60;}
      $("population-seek").max=meta.shape[0]-1;tick=Math.min(tick,meta.shape[0]-1);makePoints();buildGuide();findings();
      $("population-play").disabled=false;$("population-step").disabled=false;$("population-status").textContent=`${meta.size.toLocaleString()} neurons · paused`;
      render();
    }catch(e){meta=null;cells=null;$("population-error").hidden=false;$("population-error").textContent=e.message;$("population-status").textContent="Recording unavailable";}
  }
  $("population-recording").onchange=e=>load(e.target.value);
  $("source-sound").onclick=()=>{soundEnabled=!soundEnabled;$("source-sound").textContent=soundEnabled?"Mute source sound":"Enable source sound";$("source-sound").setAttribute("aria-pressed",String(soundEnabled));if(meta?.media)render();};
  for(const id of["source-video","source-audio"])$(id).addEventListener("loadedmetadata",()=>{if(meta?.media)render();});
  $("population-play").onclick=()=>{playing=!playing;$("population-play").textContent=playing?"Pause":"Play";lastFrame=0;accumulator=0;};
  $("population-step").onclick=()=>{pause();seek(tick+1);};$("population-seek").oninput=e=>{pause();seek(+e.target.value);};
  $("population-neuron").oninput=e=>{selected=Math.max(1,Math.min(meta?.size||1152,Math.round(+e.target.value)||1));render();};
  $("population-reset").onclick=()=>{angle=-.16;tilt=-.12;zoom=1;render();};$("population-rotate").onclick=()=>{angle+=.3;render();};
  const scene=$("population-scene");let drag=null;
  scene.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,moved:false};scene.setPointerCapture(e.pointerId);};
  scene.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.moved=drag.moved||Math.abs(dx)+Math.abs(dy)>2;angle+=dx*.006;tilt=Math.max(-.8,Math.min(.8,tilt+dy*.004));drag.x=e.clientX;drag.y=e.clientY;render();};
  scene.onpointerup=e=>{if(drag&&!drag.moved&&meta){const r=scene.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;const p=projected.map(p=>[Math.hypot(p.screen[0]-x,p.screen[1]-y),p]).sort((a,b)=>a[0]-b[0])[0];if(p&&p[0]<15){selected=p[1].nid;$("population-neuron").value=selected;render();}}drag=null;};scene.onpointercancel=()=>drag=null;
  scene.addEventListener("wheel",e=>{e.preventDefault();zoom=Math.max(.65,Math.min(1.7,zoom*Math.exp(-e.deltaY*.001)));render();},{passive:false});
  window.addEventListener("resize",render);
  function frame(now){if(playing&&meta){if(lastFrame){accumulator+=(now-lastFrame)/1000*Number($("population-speed").value);if(accumulator>=1){const steps=Math.floor(accumulator);accumulator-=steps;seek(tick+steps);if(tick===meta.shape[0]-1)pause();}}lastFrame=now;}else lastFrame=0;requestAnimationFrame(frame);}
  const initial=new URLSearchParams(location.search).get("recording")||"media-aligned";
  $("population-recording").value=[...$("population-recording").options].some(o=>o.value===initial)?initial:"media-aligned";
  requestAnimationFrame(frame);load($("population-recording").value);
})();

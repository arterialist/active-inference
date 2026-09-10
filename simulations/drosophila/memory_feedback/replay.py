"""Render saved causal B probes with the existing MuJoCo hinge and renderer.

No physics or neural steps run. Every displayed pose uses a saved qpos; qvel
and control also come from the recorded state. Camera and geometry scales are
identical between panels. Slow playback changes time only.
"""
import argparse
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from matplotlib.font_manager import findfont

from ..connectome import sha256
from ...active_inference.components.body.loaded_hinge import LoadedHinge,XML,DT


def run(base,output):
    base,output=Path(base),Path(output);output.mkdir(parents=True,exist_ok=True)
    movie=output/"causal-B-replay.mp4"
    if movie.exists(): raise FileExistsError(movie)
    manifest=json.loads((base/"memory-coverage-paired-20260910/manifest.json").read_text())
    with np.load(base/"memory-coverage-paired-20260910/identities.npz") as z:
        row=z["roots"].tolist().index(manifest["roles"]["SMP108"][0])
    cases=(("Dry B expression",("memory-coverage-eight-diagnostics-20260910/sham","memory-coverage-eight-diagnostics-20260910/removed")),
           ("B with the fixed food well available",("memory-coverage-eight-feeding-sham-20260910","memory-coverage-eight-feeding-removed-20260910")))
    font=ImageFont.truetype(findfont("DejaVu Sans"),21);small=ImageFont.truetype(findfont("DejaVu Sans"),16)
    hinge=LoadedHinge();camera=mujoco.MjvCamera();camera.lookat[:]=[.125,0,0]
    camera.distance=.55;camera.azimuth=90;camera.elevation=-90
    width,height=1200,490;fps=1/(4*DT)
    encoder=subprocess.Popen(["ffmpeg","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{width}x{height}",
        "-r",str(fps),"-i","-","-an","-c:v","libx264","-pix_fmt","yuv420p","-crf","18","-movflags","+faststart",str(movie)],stdin=subprocess.PIPE)
    receipt=dict(source_sha256=sha256(Path(__file__)),renderer="mujoco.Renderer with existing LoadedHinge XML",mujoco_version=mujoco.__version__,
        physical_dt=DT,playback_speed=.25,angular_amplification=1,physics_steps_executed=0,neural_steps_executed=0,cases=[])
    import hashlib
    receipt["body_xml_sha256"]=hashlib.sha256(XML.encode()).hexdigest()
    try:
        with mujoco.Renderer(hinge.model,height=320,width=600) as renderer:
            for stage,records in cases:
                data=[];meta=[]
                for record in records:
                    p=base/record;r=json.loads((p/"summary.json").read_text())
                    assert sha256(p/"trace.npz")==r["trace_sha256"]
                    with np.load(p/"trace.npz") as z: data.append(dict(body=z["body"].copy(),spikes=np.cumsum(z["soma"][:,row,1])))
                    meta.append(dict(record=record,trace_sha256=r["trace_sha256"],summary_sha256=sha256(p/"summary.json"),
                                     receiver_checkpoint_sha256=r["input_checkpoint_sha256"]))
                assert meta[0]["receiver_checkpoint_sha256"]==meta[1]["receiver_checkpoint_sha256"]
                for t in range(200):
                    canvas=Image.new("RGB",(width,height),(15,18,22));draw=ImageDraw.Draw(canvas)
                    draw.text((18,10),stage,fill="white",font=font)
                    draw.text((18,40),"Independent probes from the same retained state | 4x slower | angles not amplified",fill=(190,200,210),font=small)
                    for col,(label,color) in enumerate((("B memory retained",(90,190,230)),("B memory replaced",(240,155,100)))):
                        body=data[col]["body"][t]
                        hinge.data.qpos[0]=body[0];hinge.data.qvel[0]=body[1];hinge.data.ctrl[0]=body[3]
                        mujoco.mj_forward(hinge.model,hinge.data)
                        assert hinge.data.qpos[0]==body[0]
                        renderer.update_scene(hinge.data,camera=camera)
                        canvas.paste(Image.fromarray(renderer.render()),(600*col,94))
                        draw.text((18+600*col,70),label,fill=color,font=font)
                        food=float(data[col]["body"][:t+1,5].sum())
                        draw.text((18+600*col,420),f"angle {body[0]:.5f} rad | spikes {int(data[col]['spikes'][t])} | food {food:.3f} J",fill="white",font=small)
                    draw.text((18,456),f"Saved tick {t:03d} | probe time {(t+1)*DT:.3f} s | one-joint simulation body, not fly locomotion",fill=(190,200,210),font=small)
                    encoder.stdin.write(canvas.tobytes())
                # Hold the actual final pose; never interpolate or exaggerate it.
                for _ in range(round(fps)): encoder.stdin.write(canvas.tobytes())
                receipt["cases"].append(dict(label=stage,records=meta,rendered_ticks=200,final_hold_seconds=round(fps)/fps))
        encoder.stdin.close()
        if encoder.wait()!=0: raise RuntimeError("Video encoding failed")
    finally:
        if encoder.poll() is None: encoder.terminate();encoder.wait()
    receipt["movie_sha256"]=sha256(movie)
    receipt["limit"]="Kinematic replay of recorded hinge poses. The well is an angle-gated transducer, not a rendered physical object. No reconstructed fly, new dynamics, spatial amplification or fabricated intermediate poses."
    (output/"causal-B-replay.json").write_text(json.dumps(receipt,indent=2)+"\n")
    canvas.save(output/"causal-B-replay-final.png")
    return receipt


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("base",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();run(a.base,a.output)

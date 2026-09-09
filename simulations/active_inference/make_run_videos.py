"""Render the recorded 400k-tick runs to video: agent point of view, fixed third person, and a
composite that puts both beside a live trajectory map. Reads run_<world>.json written by run_long.py.

    python make_run_videos.py                 # every run_*.json it finds
    python make_run_videos.py meadow sparse
"""
import sys,os,json,base64,io,glob,subprocess,shutil,tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS=15
FF=shutil.which("ffmpeg") or "ffmpeg"

def decode(b64): return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def font(sz):
    for p in ("/System/Library/Fonts/Supplemental/Menlo.ttc",
              "/System/Library/Fonts/Menlo.ttc","/Library/Fonts/Arial.ttf"):
        if os.path.exists(p):
            try: return ImageFont.truetype(p,sz)
            except Exception: pass
    return ImageFont.load_default()

def encode(frames, out, fps=FPS):
    """PNG sequence -> h264. Dimensions are forced even; libx264 refuses odd ones."""
    if not frames: print(f"  {out}: no frames"); return
    w,h=frames[0].size; w-=w%2; h-=h%2
    with tempfile.TemporaryDirectory() as td:
        for i,f in enumerate(frames):
            f.crop((0,0,w,h)).save(os.path.join(td,f"f{i:06d}.png"))
        cmd=[FF,"-y","-loglevel","error","-framerate",str(fps),"-i",os.path.join(td,"f%06d.png"),
             "-c:v","libx264","-pix_fmt","yuv420p","-crf","20","-movflags","+faststart",out]
        subprocess.run(cmd,check=True)
    print(f"  {out}: {len(frames)} frames {w}x{h}")

def traj_panel(run, upto, size=(360,360)):
    """Top-down map of where the agent has been, with food and toxins."""
    W,H=size; A=run["arena"]; im=Image.new("RGB",(W,H),(8,11,17)); d=ImageDraw.Draw(im)
    def P(x,y): return (W/2+x/A*(W/2-14), H/2-y/A*(H/2-14))
    d.ellipse([P(-A,A),P(A,-A)],outline=(30,39,51))   # top-left then bottom-right: +y is UP in world
    for t in run["toxins"]:
        x,y=P(t[0],t[1]); d.ellipse([x-4,y-4,x+4,y+4],fill=(193,18,31))
    pts=run["traj"][:max(1,upto)]
    for f in pts[-1][5]:
        x,y=P(f[0],f[1]); d.ellipse([x-4,y-4,x+4,y+4],fill=(128,237,153))
    path=[P(p[0],p[1]) for p in pts]
    if len(path)>1: d.line(path,fill=(76,194,255),width=2)
    x,y=path[-1]; d.ellipse([x-5,y-5,x+5,y+5],fill=(255,255,255))
    yaw=pts[-1][2]; d.line([(x,y),(x+14*np.cos(yaw),y-14*np.sin(yaw))],fill=(255,158,0),width=2)
    return im

def compose(run, name):
    """POV + third person + map + HUD. Frames were recorded every frame_every*5 agent steps."""
    fe=run["frame_every"]; sub=run["sub"]; every=fe*5
    n=min(len(run["pov"]),len(run["third"]))
    F=font(13); Fb=font(16)
    pov0=decode(run["pov"][0]); th0=decode(run["third"][0])
    PW=560; ph=int(pov0.height*PW/pov0.width); th=int(th0.height*PW/th0.width)
    W=PW+380; H=max(ph+th+8, 380)+34
    out=[]
    for i in range(n):
        im=Image.new("RGB",(W,H),(8,11,17))
        im.paste(decode(run["pov"][i]).resize((PW,ph)),(0,0))
        im.paste(decode(run["third"][i]).resize((PW,th)),(0,ph+8))
        # index into the trajectory: one traj sample every frame_every steps, one image every 5 of those
        im.paste(traj_panel(run,(i*5)+1,(360,360)),(PW+12,0))
        d=ImageDraw.Draw(im)
        d.text((8,ph-20),"POINT OF VIEW",font=F,fill=(200,220,235))
        d.text((8,ph+th-12),"THIRD PERSON",font=F,fill=(200,220,235))
        step=i*every; tk=step*sub
        p=run["traj"][min(len(run["traj"])-1,i*5)]
        d.text((10,H-26),f"{name}  tick {tk:,} / {run['ticks']:,}   eaten {p[3]}   toxin hits {p[4]}"
                         f"   home {np.hypot(p[0],p[1]):.1f}",font=Fb,fill=(232,241,248))
        out.append(im)
    return out

def do(name):
    fn=f"run_{name}.json"
    if not os.path.exists(fn): print(f"  {fn} missing"); return
    run=json.load(open(fn))
    print(f"{name}: {run['ticks']:,} ticks, ate {run['eaten']}, toxin {run['tox']}, "
          f"{len(run['pov'])} camera frames")
    encode([decode(b) for b in run["pov"]],   f"video_{name}_pov.mp4")
    encode([decode(b) for b in run["third"]], f"video_{name}_third.mp4")
    encode(compose(run,name),                 f"video_{name}_full.mp4")

if __name__=="__main__":
    names=sys.argv[1:] or [os.path.basename(p)[4:-5] for p in sorted(glob.glob("run_*.json"))]
    for n in names: do(n)

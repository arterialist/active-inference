"""Replay saved omission and return failures through the existing 3D renderer."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import findfont

from .replay import LoadedHinge, XML, DT
from ..connectome import sha256


def run(source, output):
    output.mkdir(parents=True, exist_ok=True)
    movie = output / "omission-return-replay.mp4"
    if movie.exists():
        raise FileExistsError(movie)
    stages = (
        ("Food removed while A continues", "action-omission", "", 600),
        ("A food return after omission + 1000 blank ticks", "probe-omission", "-A", 200),
        ("A return after continued-food control + 1000 blank ticks", "probe-food", "-A", 200),
    )
    font = ImageFont.truetype(findfont("DejaVu Sans"), 20)
    small = ImageFont.truetype(findfont("DejaVu Sans"), 15)
    hinge = LoadedHinge()
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.125, 0, 0]
    camera.distance, camera.azimuth, camera.elevation = .55, 90, -90
    width, height, fps = 1280, 540, 1/(4*DT)
    encoder = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(movie)], stdin=subprocess.PIPE)
    receipt = dict(source_sha256=sha256(Path(__file__)), source=str(source.resolve()),
        renderer="Existing LoadedHinge XML and mujoco.Renderer; qpos, qvel and control copied from saved records; mj_forward only",
        body_xml_sha256=hashlib.sha256(XML.encode()).hexdigest(), mujoco_version=mujoco.__version__,
        camera=dict(lookat=[.125, 0, 0], distance=.55, azimuth=90, elevation=-90),
        dt=DT, fps=fps, playback_speed=.25, angular_amplification=1, physics_steps=0, neural_steps=0, stages=[])
    try:
        with mujoco.Renderer(hinge.model, height=320, width=640) as renderer:
            for title, prefix, suffix, ticks in stages:
                records, arrays, summaries = [], [], []
                for label in ("intact", "cut"):
                    name = prefix+"-"+label+suffix
                    path = source / name
                    summary = json.loads((path / "summary.json").read_text())
                    assert summary["artifacts"]["body.npy"] == sha256(path / "body.npy")
                    arrays.append(np.load(path / "body.npy", mmap_mode="r"))
                    summaries.append(summary)
                    records.append(dict(record=name, summary_sha256=sha256(path / "summary.json"),
                        body_sha256=sha256(path / "body.npy"), initial_organs=summary["initial_organs"]))
                for tick in range(ticks):
                    canvas = Image.new("RGB", (width, height), (15, 18, 22))
                    draw = ImageDraw.Draw(canvas)
                    draw.text((16, 8), title, fill="white", font=font)
                    draw.text((16, 37), "Saved 3D hinge states | quarter-speed playback | same top-view camera | actual angles", fill=(190, 200, 210), font=small)
                    for column, label in enumerate(("intact", "cut")):
                        body = arrays[column][tick]
                        hinge.data.qpos[0], hinge.data.qvel[0], hinge.data.ctrl[0] = body[0], body[1], body[3]
                        mujoco.mj_forward(hinge.model, hinge.data)
                        renderer.update_scene(hinge.data, camera=camera)
                        canvas.paste(Image.fromarray(renderer.render()), (640*column, 128))
                        color = (225, 135, 95) if column == 0 else (95, 185, 225)
                        draw.text((16+640*column, 66), "Omission-to-action path "+label, fill=color, font=font)
                        draw.text((16+640*column, 96), records[column]["record"], fill=(190, 200, 210), font=small)
                        food = float(arrays[column][:tick+1, 5].sum())
                        energy = float(body[7:9].sum()-sum(summaries[column]["initial_organs"][:2]))
                        draw.text((16+640*column, 448), f"angle {body[0]:.5f} rad | food {food:.3f} J | retained E change {energy:+.3f} J", fill="white", font=small)
                    draw.text((16, 481), f"Record tick {tick:03d} | elapsed model time {(tick+1)*DT:.3f} s | camera (.125,0,0), .55 m, az 90, elev -90", fill=(190, 200, 210), font=small)
                    draw.text((16, 508), "Chapter jumps omit blank retention. Food well is an angle-gated transducer, not a rendered object. No fly locomotion claim.", fill=(190, 200, 210), font=small)
                    encoder.stdin.write(canvas.tobytes())
                for _ in range(round(fps)):
                    encoder.stdin.write(canvas.tobytes())
                canvas.save(output / (prefix+"-final.png"))
                receipt["stages"].append(dict(title=title, records=records, ticks=ticks, final_hold_seconds=round(fps)/fps))
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Video encoding failed")
    finally:
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
    receipt["movie_sha256"] = sha256(movie)
    receipt["scope"] = "Matched branches from one learned origin. Each return uses its own post-retention state; chapter jumps do not imply a continuous filmed trajectory. All frames are recorded poses, with no interpolation or scale amplification."
    (output / "omission-return-replay.json").write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.source, args.output)

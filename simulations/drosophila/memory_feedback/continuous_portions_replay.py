"""Matched 3D replay of recorded continuous feeding and its energetic loss."""
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
    movie = output / "continuous-portions-replay.mp4"
    if movie.exists():
        raise FileExistsError(movie)
    records, arrays, environments, summaries, food = [], [], [], [], []
    for label in ("retained", "erased"):
        path = source / label
        summary = json.loads((path / "summary.json").read_text())
        extra = json.loads((source / (label+".json")).read_text())
        environment = source / (label+"-environment.npy")
        assert summary["artifacts"]["body.npy"] == sha256(path / "body.npy")
        assert extra["environment_sha256"] == sha256(environment)
        arrays.append(np.load(path / "body.npy", mmap_mode="r"))
        environments.append(np.load(environment, mmap_mode="r"))
        summaries.append(summary)
        food.append(np.cumsum(arrays[-1][:, 5]))
        records.append(dict(record=label, summary_sha256=sha256(path / "summary.json"),
            body_sha256=sha256(path / "body.npy"), environment_sha256=sha256(environment)))
    stages = (("First depletion, withdrawal and renewed collection", 0, 1000),
        ("Later renewal under the same continuous cue", 3700, 4500))
    font = ImageFont.truetype(findfont("DejaVu Sans"), 20)
    small = ImageFont.truetype(findfont("DejaVu Sans"), 15)
    hinge = LoadedHinge()
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.125, 0, 0]
    camera.distance, camera.azimuth, camera.elevation = .55, 90, -90
    width, height, fps = 1280, 560, 1/(4*DT)
    full_result = (f"Full {len(arrays[0])*DT:g} s: retained {summaries[0]['ingested_j']:.3f} J food, E change {summaries[0]['stored_energy_gain_j']:+.3f} J"
        f" | erased {summaries[1]['ingested_j']:.3f} J food, E change {summaries[1]['stored_energy_gain_j']:+.3f} J | E = energy + gut")
    encoder = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(movie)], stdin=subprocess.PIPE)
    receipt = dict(source_sha256=sha256(Path(__file__)), source=str(source.resolve()),
        manifest_sha256=sha256(source / "manifest.json"), records=records,
        renderer="Existing LoadedHinge XML and mujoco.Renderer; saved qpos/qvel/control; mj_forward only",
        body_xml_sha256=hashlib.sha256(XML.encode()).hexdigest(), mujoco_version=mujoco.__version__,
        camera=dict(lookat=[.125, 0, 0], distance=.55, azimuth=90, elevation=-90),
        dt=DT, fps=fps, playback_speed=.25, angular_amplification=1, physics_steps=0, neural_steps=0,
        stages=[], full_record_results={label: dict(food_j=s["ingested_j"], energy_gain_j=s["stored_energy_gain_j"])
            for label, s in zip(("retained", "erased"), summaries, strict=True)})
    try:
        with mujoco.Renderer(hinge.model, height=320, width=640) as renderer:
            for title, start, stop in stages:
                for tick in range(start, stop):
                    canvas = Image.new("RGB", (width, height), (15, 18, 22))
                    draw = ImageDraw.Draw(canvas)
                    draw.text((16, 8), title, fill="white", font=font)
                    draw.text((16, 37), f"{source.name} | constant A | quarter speed | same camera | actual saved poses", fill=(190, 200, 210), font=small)
                    for column, label in enumerate(("retained", "erased")):
                        body, environment = arrays[column][tick], environments[column][tick]
                        hinge.data.qpos[0], hinge.data.qvel[0], hinge.data.ctrl[0] = body[0], body[1], body[3]
                        mujoco.mj_forward(hinge.model, hinge.data)
                        renderer.update_scene(hinge.data, camera=camera)
                        canvas.paste(Image.fromarray(renderer.render()), (640*column, 128))
                        color = (95, 185, 225) if column == 0 else (225, 135, 95)
                        draw.text((16+640*column, 66), "Stored expectation "+("retained" if column == 0 else "initially erased"), fill=color, font=font)
                        draw.text((16+640*column, 96), f"record: {label} | portions loaded {int(environment[4])} | remaining {environment[3]:.3f} J", fill=(190, 200, 210), font=small)
                        energy = float(body[7:9].sum()-sum(summaries[column]["initial_organs"][:2]))
                        draw.text((16+640*column, 448), f"angle {body[0]:.5f} rad | cumulative food {food[column][tick]:.3f} J | E change {energy:+.3f} J", fill="white", font=small)
                    draw.text((16, 479), f"Record tick {tick} | model time {(tick+1)*DT:.3f} s | camera (.125,0,0), .55 m, az 90, elev -90", fill=(190, 200, 210), font=small)
                    draw.text((16, 503), "Omitted: ticks 1000-3699 (10.8 s) and 4500-4999 (2 s). State continues; refill requires withdrawal, no timed reset.", fill=(190, 200, 210), font=small)
                    draw.text((16, 532), full_result, fill=(230, 185, 130), font=small)
                    encoder.stdin.write(canvas.tobytes())
                for _ in range(round(fps)):
                    encoder.stdin.write(canvas.tobytes())
                canvas.save(output / f"continuous-{start}-{stop}-final.png")
                receipt["stages"].append(dict(title=title, start_tick=start, stop_tick_exclusive=stop,
                    rendered_poses=stop-start, final_hold_seconds=round(fps)/fps))
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Video encoding failed")
    finally:
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
    receipt["movie_sha256"] = sha256(movie)
    receipt["scope"] = "Two matched continuously simulated branches. Ticks1000:3700 and4500:5000 are not shown; no acquisition or state reset occurred at chapter boundaries. All visible poses are actual records, with no interpolation or amplification. Food is an angle-gated finite transducer, not a rendered object; hinge is not a fly locomotion reconstruction. E change is energy+gut relative to each original matched start."
    (output / "continuous-portions-replay.json").write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.source, args.output)

"""Small fixed-camera MuJoCo evidence replay, with no neural or body stepping."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .association_route_probe import digest
from .composition_probe import encode
from ..components.body.radian_research_rower import RadianResearchRower
from ..make_run_videos import font


def run(roots, output):
    output = Path(output).resolve()
    if len(roots) != 4 or output.exists() or not shutil.which('ffmpeg'):
        raise ValueError('Need four recordings, a new output and ffmpeg')
    records = []
    for root in map(lambda p: Path(p).resolve(), roots):
        m = json.loads((root/'manifest.json').read_text())
        s = json.loads((root/'summary.json').read_text())
        raw = root/'closed-loop.npz'
        if digest(raw) != s['raw_sha256'] or any(digest(p) != h for p, h in m['source_hashes'].items()):
            raise ValueError('Evidence provenance changed')
        with np.load(raw) as z:
            d = {k: z[k] for k in ('physical_before', 'physical_after', 'ticks', 'applied_force')}
        label = {'none': 'No corrective current', 'position': 'Direct position reflex',
                 'expectation': 'Learned expectation'}[m['mode']]
        if m['reset_prediction']:
            label = 'Expectation weights reset'
        records.append((root, m, d, label))
    count = len(records[0][2]['ticks'])
    if count % 4 or any(not np.array_equal(r[2]['ticks'], records[0][2]['ticks']) for r in records):
        raise ValueError('Need synchronized complete four-tick frame intervals')
    output.mkdir(exist_ok=False)
    body = RadianResearchRower()
    body.restore(records[0][2]['physical_before'][0])
    camera = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [body.data.qpos[0]-.07, body.data.qpos[1], -.03]
    camera.distance = .95; camera.elevation = -58.; camera.azimuth = 90.
    # Appearance only. No geometry, dynamics or state changes are introduced.
    body.model.geom_rgba[1:3] = [.35, .4, .45, 1.]
    body.model.geom_rgba[3] = [.88, .45, .15, 1.]
    body.model.geom_rgba[4] = [.15, .55, .72, 1.]
    width, height = 480, 320
    canvas_width, canvas_height = width*2, (height+68)*2+76
    renderer = mujoco.Renderer(body.model, width=width, height=height)
    video = output/'comparison.mp4'
    process = subprocess.Popen([shutil.which('ffmpeg'), '-n', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{canvas_width}x{canvas_height}',
        '-r', '125/8', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '20',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    frames = [-1]+list(range(3, count, 4))
    try:
        for frame, t in enumerate(frames):
            image = Image.new('RGB', (canvas_width, canvas_height), (244, 242, 234))
            draw = ImageDraw.Draw(image)
            seconds = (t+1)*body.model.opt.timestep
            draw.text((16, 10), 'Recorded MuJoCo states | fixed camera | 0.25x speed', font=font(18), fill=(20, 20, 20))
            draw.text((16, 39), f'Simulation +{seconds:.3f}s | orange: left paddle | blue: right paddle', font=font(15), fill=(30, 30, 30))
            for j, (root, m, d, label) in enumerate(records):
                x, y = (j % 2)*width, 76+(j//2)*(height+68)
                state = d['physical_before'][0] if t < 0 else d['physical_after'][t]
                body.restore(state)
                qpos, qvel = body.data.qpos.copy(), body.data.qvel.copy()
                mujoco.mj_forward(body.model, body.data)
                np.testing.assert_array_equal(body.data.qpos, qpos)
                np.testing.assert_array_equal(body.data.qvel, qvel)
                renderer.update_scene(body.data, camera=camera)
                image.paste(Image.fromarray(renderer.render()), (x, y+56))
                draw.text((x+12, y+3), label, font=font(17), fill=(20, 20, 20))
                force = 0. if t < 0 else float(np.max(np.abs(d['applied_force'][t])))
                stage = 'LOAD ON' if force else 'before load' if t < m['force_start'] else 'load off'
                if not m['torque']:
                    stage = 'no-load control'
                draw.text((x+12, y+29), f'{stage} | external torque {force:.1f} Nm', font=font(14),
                          fill=(155, 35, 20) if force else (40, 40, 40))
            process.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
            if t in (-1, 303, count-1):
                image.save(output/f'frame-{t+1:04d}.png')
        process.stdin.close()
        error = process.stderr.read().decode()
        if process.wait(timeout=60) != 0:
            raise RuntimeError(error)
    except BaseException:
        process.terminate(); process.wait(timeout=10)
        raise
    finally:
        renderer.close()
    manifest = dict(renderer_sha256=digest(__file__), frames=frames, fps='125/8', playback_speed=.25,
        simulated_seconds=count*body.model.opt.timestep, video_sha256=digest(video),
        camera=dict(lookat=camera.lookat, distance=camera.distance, elevation=camera.elevation, azimuth=camera.azimuth),
        runs=[dict(path=str(root), condition=label, raw_sha256=digest(root/'closed-loop.npz'))
              for root, _, _, label in records],
        fidelity='Each frame restores an actual saved integration state and calls mj_forward only for rendering. '
                 'No dynamics integration or interpolation. One initial frame plus every fourth completed tick. '
                 'Same fixed camera; only geom colours changed for identification. No motion magnification.')
    (output/'manifest.json').write_text(encode(manifest)+'\n')
    print(encode(dict(video=str(video), frames=len(frames), bytes=video.stat().st_size)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots', type=Path, nargs=4, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(**vars(p.parse_args()))

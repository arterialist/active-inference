"""Synchronized visual artifacts for embodied evidence runs.

The acceptance traces are still the scientific source of truth.  This module
adds three human-auditable views of the *same* physical ticks: the MuJoCo eye
camera attached to the body, a fixed world-frame third-person camera, and a
top-down 2-D projection.  Frames are sampled at a bounded rate, encoded to
small H.264 files, and accompanied by a manifest that makes missing or failed
capture explicit instead of silently omitting media.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import mujoco
import numpy as np
from PIL import Image, ImageDraw


class EmbodiedVideoRecorder:
    """Capture synchronized POV/third-person/top-down videos for one case."""

    def __init__(self, agent, artifact_dir: Path | None, *, total_ticks: int,
                 substeps: int, fps: int = 15, max_frames: int = 180):
        self.agent = agent
        self.world = agent.world
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self.total_ticks = max(1, int(total_ticks))
        self.substeps = max(1, int(substeps))
        self.fps = max(1, int(fps))
        self.max_frames = max(1, int(max_frames))
        self.sample_every = max(1, int(math.ceil(self.total_ticks / self.max_frames)))
        self.frame_count = 0
        self.errors: list[str] = []
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        self._frame_root: Path | None = None
        self._pov_renderer = None
        self._third_renderer = None
        self._third_camera = None
        self._eye_camera = None
        self._path: list[tuple[float, float]] = []
        self._frame_ticks: list[int] = []
        if self.artifact_dir is None:
            self.enabled = False
            return
        self.enabled = True
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._tmp = tempfile.TemporaryDirectory(prefix="embodied-video-",
                                                     dir=str(self.artifact_dir))
            self._frame_root = Path(self._tmp.name)
            self._pov_renderer = mujoco.Renderer(self.world.model, height=180, width=320)
            self._third_renderer = mujoco.Renderer(self.world.model, height=240, width=360)
            self._third_camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self._third_camera)
            self._third_camera.lookat[:] = [0.0, 0.0, 0.4]
            self._third_camera.distance = max(8.0, float(getattr(self.world, "arena", 7.0)) * 1.35)
            self._third_camera.elevation = -55.0
            self._third_camera.azimuth = 90.0
            self._eye_camera = mujoco.mj_name2id(
                self.world.model, mujoco.mjtObj.mjOBJ_CAMERA, "eye"
            )
            for name in ("pov", "third_person", "top_down"):
                (self._frame_root / name).mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - backend/display dependent
            self.errors.append(f"initialization failed: {type(exc).__name__}: {exc}")
            self.enabled = False

    @staticmethod
    def _save_image(image: np.ndarray, path: Path) -> None:
        Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(
            path, format="JPEG", quality=72, optimize=True
        )

    def _top_down(self) -> Image.Image:
        world = self.world
        width = height = 360
        arena = max(1.0, float(getattr(world, "arena", 7.0)))
        image = Image.new("RGB", (width, height), (8, 11, 17))
        draw = ImageDraw.Draw(image)

        def point(x: float, y: float) -> tuple[int, int]:
            return (
                int(width / 2 + float(x) / arena * (width / 2 - 16)),
                int(height / 2 - float(y) / arena * (height / 2 - 16)),
            )

        draw.rectangle((10, 10, width - 10, height - 10), outline=(42, 55, 70))
        for obstacle in getattr(world, "barriers", ()):
            x, y = point(obstacle.get("x", 0.0), obstacle.get("y", 0.0))
            hx = abs(float(obstacle.get("hx", 0.0))) / arena * (width / 2 - 16)
            hy = abs(float(obstacle.get("hy", 0.0))) / arena * (height / 2 - 16)
            draw.rectangle((x - hx, y - hy, x + hx, y + hy), fill=(42, 58, 82), outline=(112, 135, 165))
        for source, colour in ((getattr(world, "foods", ()), (128, 237, 153)),
                               (getattr(world, "toxins", ()), (224, 64, 80))):
            for item in source:
                if len(item) < 2 or abs(float(item[0])) > arena * 2:
                    continue
                x, y = point(item[0], item[1])
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colour)
        if len(self._path) > 1:
            draw.line([point(x, y) for x, y in self._path], fill=(76, 194, 255), width=2)
        x, y, yaw = world.pose()
        px, py = point(x, y)
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=(245, 245, 245))
        draw.line((px, py, px - 18 * math.cos(yaw), py + 18 * math.sin(yaw)), fill=(255, 158, 0), width=2)
        return image

    def _render(self) -> tuple[np.ndarray, np.ndarray, Image.Image]:
        self._pov_renderer.update_scene(self.world.data, camera=self._eye_camera)
        pov = self._pov_renderer.render()
        self._third_renderer.update_scene(self.world.data, camera=self._third_camera)
        third = self._third_renderer.render()
        return pov, third, self._top_down()

    def observe(self, current=None) -> None:
        """Record the current post-tick state; safe as a run_episode hook."""
        if not self.enabled:
            return
        try:
            x, y, _ = self.world.pose()
            self._path.append((float(x), float(y)))
            tick = int(getattr(current, "t", len(self._path)))
            if self.frame_count and tick % self.sample_every:
                return
            pov, third, top = self._render()
            index = self.frame_count
            self._save_image(pov, self._frame_root / "pov" / f"{index:06d}.jpg")
            self._save_image(third, self._frame_root / "third_person" / f"{index:06d}.jpg")
            top.save(self._frame_root / "top_down" / f"{index:06d}.jpg", format="JPEG", quality=82)
            self._frame_ticks.append(tick)
            self.frame_count += 1
        except Exception as exc:  # pragma: no cover - backend/display dependent
            self.errors.append(f"frame {self.frame_count} failed: {type(exc).__name__}: {exc}")

    def _encode(self, view: str, output: Path) -> bool:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg or self.frame_count <= 0:
            self.errors.append(f"{view}: ffmpeg unavailable or no frames")
            return False
        command = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(self.fps),
                   "-i", str(self._frame_root / view / "%06d.jpg"),
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30",
                   "-movflags", "+faststart", str(output)]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            return output.exists() and output.stat().st_size > 0
        except (OSError, subprocess.CalledProcessError) as exc:
            self.errors.append(f"{view}: encode failed: {exc}")
            return False

    def finalize(self) -> dict:
        """Encode the three views and return a portable artifact manifest."""
        if not self.enabled:
            return {"status": "failed", "frame_count": 0, "errors": self.errors or ["capture disabled"]}
        outputs = {
            "pov": self.artifact_dir / "pov.mp4",
            "third_person": self.artifact_dir / "third_person.mp4",
            "top_down": self.artifact_dir / "top_down.mp4",
        }
        encoded = {name: self._encode(name, path) for name, path in outputs.items()}
        result = {
            "status": "complete" if all(encoded.values()) and not self.errors else "failed",
            "frame_count": self.frame_count,
            "sample_every_neural_ticks": self.sample_every,
            "frame_neural_ticks": list(self._frame_ticks),
            "fps": self.fps,
            "views": {name: str(path.name) if encoded[name] else None for name, path in outputs.items()},
            "errors": list(self.errors),
        }
        (self.artifact_dir / "artifacts.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        for renderer in (self._pov_renderer, self._third_renderer):
            try:
                if renderer is not None:
                    renderer.close()
            except Exception:
                pass
        if self._tmp is not None:
            self._tmp.cleanup()
        return result


def artifact_directory(root: Path | None, case: str, seed: int) -> Path | None:
    """Return a stable per-condition artifact location, if recording is requested."""
    if root is None:
        return None
    return Path(root) / "artifacts" / f"{case}_seed{int(seed)}"

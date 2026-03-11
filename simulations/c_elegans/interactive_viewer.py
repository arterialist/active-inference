"""
C. elegans interactive real-time environment viewer.

2D top-down view of the agar plate with worm trajectory and food.
Click to add food, right-click to remove. No logging or recording.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

from simulations.c_elegans.config import (
    ENV_PLATE_RADIUS_M,
    FOOD_CONSUMPTION_RADIUS_M,
)
from simulations.c_elegans.environment import AgarPlateEnvironment
from simulations.interactive.base import BaseInteractiveViewer

if TYPE_CHECKING:
    from simulations.engine import SimulationEngine
    from simulations.sensorimotor_loop import SensorimotorLoop

# Rolling trajectory length (points)
TRAJECTORY_MAXLEN = 2000
# Rolling metrics length for display
METRICS_MAXLEN = 500
# Display scale: metres to mm
M_TO_MM = 1000.0


def _screen_to_world(
    xdata: float,
    ydata: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[float, float, float]:
    """Convert axes data coordinates (mm) to world position (m)."""
    x_lo, x_hi = xlim
    y_lo, y_hi = ylim
    # xdata, ydata are in mm
    x_m = (xdata / 1000.0) if x_lo <= xdata <= x_hi else 0.0
    y_m = (ydata / 1000.0) if y_lo <= ydata <= y_hi else 0.0
    return (x_m, y_m, 0.0)


class CElegansInteractiveViewer(BaseInteractiveViewer):
    """
    Interactive 2D viewer for C. elegans on agar plate.

    - Left-click: add food at cursor position
    - Right-click: remove nearest food
    - Close window to exit
    """

    def __init__(
        self,
        trajectory_maxlen: int = TRAJECTORY_MAXLEN,
        metrics_maxlen: int = METRICS_MAXLEN,
    ):
        self._trajectory_maxlen = trajectory_maxlen
        self._trajectory_x: deque[float] = deque(maxlen=trajectory_maxlen)
        self._trajectory_y: deque[float] = deque(maxlen=trajectory_maxlen)
        self._metrics_maxlen = metrics_maxlen
        self._running = True

    def run(
        self,
        engine: SimulationEngine,
        loop: SensorimotorLoop,
    ) -> None:
        """Run interactive viewer until window closed."""
        import matplotlib
        # Prefer macosx on macOS, Qt5Agg elsewhere; fallback to TkAgg
        import sys
        if sys.platform == "darwin":
            matplotlib.use("macosx")
        else:
            try:
                matplotlib.use("Qt5Agg")
            except ImportError:
                matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        self._running = True
        loop.reset()

        env = engine.environment
        if not isinstance(env, AgarPlateEnvironment):
            raise TypeError("CElegansInteractiveViewer requires AgarPlateEnvironment")

        fig, (ax_traj, ax_metrics) = plt.subplots(2, 1, figsize=(10, 10), height_ratios=[1.2, 0.6])
        ax_traj.set_xlabel("x (mm)")
        ax_traj.set_ylabel("y (mm)")
        ax_traj.set_title("C. elegans – click to add food, right-click to remove")
        ax_traj.set_aspect("equal")
        ax_traj.grid(True, alpha=0.3)

        # Plate bounds in mm
        half_mm = ENV_PLATE_RADIUS_M * M_TO_MM
        ax_traj.set_xlim(-half_mm, half_mm)
        ax_traj.set_ylim(-half_mm, half_mm)

        # Metrics panel
        ax_metrics.set_xlabel("Step")
        ax_metrics.set_ylabel("Magnitude")
        ax_metrics.set_title("Prediction error & free energy proxy")
        ax_metrics.grid(True, alpha=0.3)

        # Artists
        line_traj, = ax_traj.plot([], [], "b-", linewidth=2, alpha=0.8)
        pt_start = ax_traj.plot([], [], "go", markersize=10, label="Start")[0]
        head_circle = Circle(
            (0, 0),
            FOOD_CONSUMPTION_RADIUS_M * M_TO_MM,
            color="red",
            fill=True,
            alpha=0.8,
            label="Head",
            zorder=4,
        )
        ax_traj.add_patch(head_circle)
        food_artists: list[Any] = []
        line_pe, = ax_metrics.plot([], [], "b-", alpha=0.8, label="Prediction error")
        line_me, = ax_metrics.plot([], [], "orange", alpha=0.8, label="Motor entropy (FE proxy)")
        ax_metrics.legend(loc="upper right")

        def _update_food_display() -> None:
            for a in food_artists:
                a.remove()
            food_artists.clear()
            for pos in env.get_active_food_positions():
                fx = pos[0] * M_TO_MM
                fy = pos[1] * M_TO_MM
                star, = ax_traj.plot(
                    fx, fy, "m*",
                    markersize=18,
                    markeredgecolor="k",
                    markeredgewidth=0.5,
                    zorder=5,
                )
                food_artists.append(star)

        def _on_click(event) -> None:
            if event.inaxes != ax_traj or event.xdata is None or event.ydata is None:
                return
            xlim = ax_traj.get_xlim()
            ylim = ax_traj.get_ylim()
            pos = _screen_to_world(event.xdata, event.ydata, xlim, ylim)
            if event.button == 1:
                env.add_food(pos)
            elif event.button == 3:
                env.remove_food_near(pos)
            _update_food_display()
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", _on_click)
        plt.ion()

        _update_food_display()

        try:
            while self._running and plt.fignum_exists(fig.number):
                step = engine.step()
                body = step.body_state
                pos = body.position
                head = body.head_position

                self._trajectory_x.append(float(pos[0] * M_TO_MM))
                self._trajectory_y.append(float(pos[1] * M_TO_MM))

                line_traj.set_data(list(self._trajectory_x), list(self._trajectory_y))
                if len(self._trajectory_x) > 0:
                    pt_start.set_data([self._trajectory_x[0]], [self._trajectory_y[0]])
                head_circle.center = (head[0] * M_TO_MM, head[1] * M_TO_MM)

                # Update prediction error & free energy
                trace = loop.free_energy_trace
                n_metrics = min(len(trace.ticks), self._metrics_maxlen)
                if n_metrics > 0:
                    ticks = list(trace.ticks)[-n_metrics:]
                    pe = list(trace.prediction_error)[-n_metrics:]
                    me = list(trace.motor_entropy)[-n_metrics:]
                    line_pe.set_data(ticks, pe)
                    line_me.set_data(ticks, me)
                    ax_metrics.relim()
                    ax_metrics.autoscale_view()

                _update_food_display()

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)
        finally:
            plt.ioff()
            plt.close(fig)

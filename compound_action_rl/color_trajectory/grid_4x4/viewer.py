"""Matplotlib viewer for the color-trajectory GridEnv."""

from __future__ import annotations

import argparse

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from grid_environment_2 import (
    GREEN_COLOR,
    GridEnv,
    RED_COLOR,
    UNCOLORED,
)
from grpo import GRPOAgent
from hierarchical import HierarchicalAgent
from random_agent import make_random_agent


STEP_MS = 600
PAUSE_AFTER_EPISODE_MS = 2000

CELL_FACE = {
    UNCOLORED: "#ecf0f1",
    RED_COLOR: "#e74c3c",
    GREEN_COLOR: "#27ae60",
}
OPTIMAL_RED = "#fadbd8"
OPTIMAL_GREEN = "#d5f5e3"
OPTIMAL_BLANK = "#f8f9fa"


class ColorGridViewer:
    def __init__(
        self,
        agent: HierarchicalAgent,
        *,
        title: str,
        greedy: bool = False,
        step_ms: int = STEP_MS,
    ) -> None:
        self.env = GridEnv()
        self.agent = agent
        self.greedy = greedy
        self.step_ms = step_ms
        self.step_count = 0
        self.episode_reward = 0.0
        self.finished = False
        self.waiting_restart = False
        self.last_move = "-"
        self.last_color = "-"
        self.status_text = "Starting..."

        self.fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.5))
        self.ax_grid, self.ax_opt = axes
        self.fig.canvas.manager.set_window_title(title)
        self.fig.patch.set_facecolor("#f5f6fa")
        self.fig.suptitle(
            "Net1: move → Net2: color → env executes both — reward at goal only",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )
        self.status = self.fig.text(
            0.5, 0.03, "", ha="center", va="bottom", fontsize=10, wrap=True
        )

        self.timer = self.fig.canvas.new_timer(interval=self.step_ms)
        self.timer.add_callback(self.tick)
        self.restart_timer = self.fig.canvas.new_timer(interval=PAUSE_AFTER_EPISODE_MS)
        self.restart_timer.add_callback(self.begin_episode)

        self.begin_episode()
        self.timer.start()

    def begin_episode(self) -> None:
        self.restart_timer.stop()
        self.waiting_restart = False
        self.finished = False
        self.step_count = 0
        self.episode_reward = 0.0
        self.last_move = "-"
        self.last_color = "-"
        self.env.reset()
        self.status_text = "Episode started at (0, 0). (0, 0) stays uncolored."
        self.draw()

    def tick(self) -> None:
        if self.waiting_restart:
            return
        if self.finished:
            self.waiting_restart = True
            self.restart_timer.start()
            return
        self.step_once()

    def step_once(self) -> None:
        obs = self.env.get_observation()
        if self.greedy and isinstance(self.agent, GRPOAgent):
            move_action, color_action, _info = self.agent.act_greedy(obs)
        else:
            move_action, color_action, _info = self.agent.act(obs)
        self.last_move, self.last_color = self.agent.action_labels(move_action, color_action)
        _, reward, done, state = self.env.step(move_action, color_action)
        self.step_count += 1
        self.episode_reward = reward

        if done:
            self.finished = True
            if self.env._at_goal():
                end_reason = f"Goal reached at {tuple(state)}"
            else:
                end_reason = f"Episode ended at {tuple(state)} (max {self.env.max_episode_steps} steps)"
            self.status_text = (
                f"{end_reason} after {self.step_count} steps. "
                f"Terminal reward: {reward:.3f}"
            )
        else:
            self.status_text = (
                f"Step {self.step_count}/{self.env.max_episode_steps}: "
                f"net1={self.last_move}, net2={self.last_color}, "
                f"agent at {tuple(state)}."
            )
        self.draw()

    def _draw_grid(
        self,
        ax: plt.Axes,
        colors: np.ndarray,
        *,
        title: str,
        show_agent: bool,
        optimal_style: bool = False,
    ) -> None:
        ax.clear()
        ax.set_title(title, fontsize=11, fontweight="bold")
        size = self.env.grid_size
        pad = 0.15
        ax.set_xlim(-pad, size + pad)
        ax.set_ylim(-pad, size + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.autoscale(False)
        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        ax.set_facecolor("#f5f6fa")

        goal_row, goal_col = self.env.goal_pos

        for row in range(self.env.grid_size):
            for col in range(self.env.grid_size):
                cell = int(colors[row, col])
                if optimal_style:
                    if cell == UNCOLORED:
                        face = OPTIMAL_BLANK
                    elif cell == RED_COLOR:
                        face = OPTIMAL_RED
                    else:
                        face = OPTIMAL_GREEN
                else:
                    face = CELL_FACE.get(cell, CELL_FACE[UNCOLORED])

                rect = patches.Rectangle(
                    (col, row),
                    1,
                    1,
                    facecolor=face,
                    edgecolor="#b2bec3",
                    linewidth=1.2,
                )
                ax.add_patch(rect)

                if row == 0 and col == 0:
                    ax.text(
                        col + 0.5,
                        row + 0.12,
                        "S",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#636e72",
                    )
                if row == goal_row and col == goal_col:
                    ax.add_patch(
                        patches.Rectangle(
                            (col + 0.08, row + 0.08),
                            0.84,
                            0.84,
                            fill=False,
                            edgecolor="#f39c12",
                            linewidth=2.5,
                        )
                    )

        if show_agent:
            ax.add_patch(
                patches.Circle(
                    (self.env._col + 0.5, self.env._row + 0.5),
                    0.22,
                    facecolor="#0984e3",
                    edgecolor="#2d3436",
                    linewidth=1.5,
                    zorder=5,
                )
            )

    def draw(self) -> None:
        self._draw_grid(
            self.ax_grid,
            self.env._colors,
            title="Current coloring",
            show_agent=True,
        )
        self._draw_grid(
            self.ax_opt,
            self.env.optimal_color_pattern(),
            title="Optimal target (blank = unpaintable)",
            show_agent=False,
            optimal_style=True,
        )
        self.status.set_text(self.status_text)
        self.fig.tight_layout(rect=[0, 0.06, 1, 0.94])
        self.fig.canvas.draw_idle()

    def run(self) -> None:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="View GridEnv rollouts.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for a random agent")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a trained GRPO checkpoint (from train.py --save-path).",
    )
    parser.add_argument(
        "--greedy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use greedy actions when viewing a trained checkpoint.",
    )
    parser.add_argument(
        "--step-ms",
        type=int,
        default=STEP_MS,
        help="Milliseconds between steps",
    )
    args = parser.parse_args()

    if args.checkpoint:
        agent = GRPOAgent.from_checkpoint(args.checkpoint)
        title = "Color Trajectory — Learned GRPO Policy"
    else:
        agent = make_random_agent(seed=args.seed)
        title = "Color Trajectory — Random Agent"

    viewer = ColorGridViewer(
        agent,
        title=title,
        greedy=args.greedy and args.checkpoint is not None,
        step_ms=args.step_ms,
    )
    viewer.run()


if __name__ == "__main__":
    main()

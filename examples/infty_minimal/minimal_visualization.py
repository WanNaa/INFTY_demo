"""Minimal trajectory visualization example for INFTY.

Run from repository root:
    python examples/infty_minimal/minimal_visualization.py
"""

import matplotlib
matplotlib.use("Agg")

from infty.plot import visualize_trajectory
from infty.plot.paths import MINIMAL_TRAJECTORY_DIR


def main():
    trajectory = visualize_trajectory(
        optimizer_name="adam",
        n_iter=200,
        lr=0.1,
        output_dir=MINIMAL_TRAJECTORY_DIR,
        grid_size=120,
    )
    print(f"trajectory length: {len(trajectory)}")
    print(f"plot saved under {MINIMAL_TRAJECTORY_DIR}")


if __name__ == "__main__":
    main()

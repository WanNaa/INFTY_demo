"""Minimal trajectory visualization example for INFTY.

Run from repository root:
    python examples/infty_minimal/minimal_visualization.py
"""

import matplotlib
matplotlib.use("Agg")

from infty.plot import visualize_trajectory


def main():
    trajectory = visualize_trajectory(
        optimizer_name="adam",
        n_iter=200,
        lr=0.1,
        output_dir="workdirs/plots/trajectory/minimal",
        grid_size=120,
    )
    print(f"trajectory length: {len(trajectory)}")
    print("plot saved under workdirs/plots/trajectory/minimal")


if __name__ == "__main__":
    main()

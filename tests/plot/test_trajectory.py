from pathlib import Path

import pytest


def test_visualize_trajectory_writes_into_output_dir(tmp_path):
    pytest.importorskip("matplotlib")
    from infty.plot import visualize_trajectory

    trajectory = visualize_trajectory("sgd", n_iter=5, output_dir=tmp_path, grid_size=50)
    assert trajectory.shape[0] == 5
    assert any(Path(tmp_path).rglob("traj_sgd.pdf"))

from pathlib import Path

import pytest


class DummyOptimizer:
    name = "dummy"
    sim_arr = [0.1, -0.2, 0.3]


def test_visualize_conflicts_accepts_sim_arr(tmp_path):
    pytest.importorskip("matplotlib")
    from infty.plot import visualize_conflicts

    result = visualize_conflicts(DummyOptimizer(), task=1, output_dir=tmp_path)
    assert Path(result["sim_path"]).exists()
    assert Path(result["plot_path"]).exists()

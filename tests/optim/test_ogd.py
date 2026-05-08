from pathlib import Path

import torch


def test_ogd_uses_checkpoint_dir_for_basis(tmp_path):
    from infty.optim import OGD

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 2),
    )
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    opt0 = OGD(
        model.parameters(),
        base_optimizer=base_optimizer,
        model=model,
        args={"task_id": 0, "ckp_dir": str(tmp_path)},
    )

    assert Path(opt0.basis_path).exists()
    assert Path(opt0.basis_path).parent == tmp_path.resolve()

    opt1 = OGD(
        model.parameters(),
        base_optimizer=base_optimizer,
        model=model,
        args={"task_id": 1, "ckp_dir": str(tmp_path)},
    )

    assert Path(opt1.basis_path) == Path(opt0.basis_path)

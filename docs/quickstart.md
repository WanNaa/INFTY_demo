# Quick Start

This page shows the minimum patterns needed to use INFTY optimizers in a PyTorch training loop.

## Core pattern

Most INFTY optimizers use the same pattern:

1. create a normal PyTorch optimizer;
2. wrap it with an INFTY optimizer;
3. write a closure that returns `(logits, loss_list)`;
4. call `optimizer.set_closure(loss_fn)`;
5. call `optimizer.step()`.

The closure contract is important:

```python
logits, loss_list = loss_fn()
```

where `loss_list` must be a list of scalar tensors.

## Minimal C-Flat example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from infty.optim import C_Flat

x = torch.randn(64, 16)
y = torch.randint(0, 4, (64,))
loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)

model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

optimizer = C_Flat(
    params=model.parameters(),
    base_optimizer=base_optimizer,
    model=model,
    args={"rho": 0.05, "lamb": 0.2, "strategy": "basic"},
)

def make_loss_fn(inputs, targets):
    def loss_fn():
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        return logits, [loss]
    return loss_fn

model.train()
for inputs, targets in loader:
    optimizer.set_closure(make_loss_fn(inputs, targets))
    logits, loss_list = optimizer.step()
    print(float(sum(loss_list)))
```

## Minimal ZeroFlow example

ZeroFlow estimates update directions without ordinary backpropagation. It still uses the same high-level closure format.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from infty.optim import ZeroFlow

model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
inputs = torch.randn(32, 8)
targets = torch.randint(0, 3, (32,))

base_optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer = ZeroFlow(
    params=model.parameters(),
    base_optimizer=base_optimizer,
    model=model,
    args={
        "inftyopt": "zo_sgd",
        "q": 1,
        "zo_eps": 1e-3,
        "perturbation_mode": "two_side",
    },
)

def loss_fn():
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)
    return logits, [loss]

optimizer.set_closure(loss_fn)
logits, loss_list = optimizer.step()
print(float(sum(loss_list)))
```

## Minimal UniGrad-FS example

UniGrad-FS is intended for two-objective continual-learning settings. For non-first tasks, it expects exactly two loss terms.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from infty.optim import UniGrad_FS

model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
inputs = torch.randn(32, 10)
new_targets = torch.randint(0, 5, (32,))
old_targets = torch.randint(0, 5, (32,))

base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = UniGrad_FS(
    params=model.parameters(),
    base_optimizer=base_optimizer,
    model=model,
    args={
        "task_id": 1,
        "utype": "model-wise",
        "S_T": [0.1],
        "beta": 0.9,
        "rho": 0.05,
    },
)

def loss_fn():
    logits = model(inputs)
    old_loss = F.cross_entropy(logits, old_targets)
    new_loss = F.cross_entropy(logits, new_targets)
    return logits, [old_loss, new_loss]

optimizer.set_closure(loss_fn)
logits, loss_list = optimizer.step()
print([float(loss) for loss in loss_list])
```

## Run the PILOT demo

```bash
cd INFTY_demo
python -m pip install .[examples]
cd examples/PILOT
python main.py --config=exps/memo_scr.json --inftyopt=c_flat --workdir ../../workdirs
python main.py --config=exps/ease.json --inftyopt=zo_sgd_conserve --workdir ../../workdirs
python main.py --config=exps/icarl.json --inftyopt=unigrad_fs --workdir ../../workdirs
```

## Quick visualization example

```python
from infty.plot import visualize_trajectory

visualize_trajectory(
    optimizer_name="adam",
    n_iter=500,
    lr=0.1,
    output_dir="workdirs/plots/diagnostics/trajectory/demo",
    grid_size=200,
)
```

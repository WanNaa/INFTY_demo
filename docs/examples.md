# Examples

This page lists runnable examples and recommended experiment entry points.

## PILOT demo

The repository includes a PILOT-based example under `examples/PILOT`.

Install optional dependencies:

```bash
cd INFTY_demo
python -m pip install .[examples]
```

Run C-Flat:

```bash
cd examples/PILOT
python main.py --config=exps/memo_scr.json --inftyopt=c_flat --workdir ../../workdirs
```

Run ZeroFlow conservative update:

```bash
python main.py --config=exps/ease.json --inftyopt=zo_sgd_conserve --workdir ../../workdirs
```

Run UniGrad-FS:

```bash
python main.py --config=exps/icarl.json --inftyopt=unigrad_fs --workdir ../../workdirs
```

Run a dry-run script for geometry reshaping:

```bash
DRY_RUN=1 bash ../../workdirs/run_scripts/run_geometry_reshaping.sh
```

## Minimal examples included in this documentation package

This documentation package also contains small standalone scripts under:

```text
examples/infty_minimal/
```

| File | Purpose |
| --- | --- |
| `minimal_cflat.py` | One-step/small-loop C-Flat example on toy data. |
| `minimal_zeroflow.py` | ZeroFlow example using a finite-difference update. |
| `minimal_unigrad_fs.py` | Two-loss UniGrad-FS example. |
| `minimal_visualization.py` | Toy trajectory visualization example. |

Run them from the repository root after installing INFTY:

```bash
python examples/infty_minimal/minimal_cflat.py
python examples/infty_minimal/minimal_zeroflow.py
python examples/infty_minimal/minimal_unigrad_fs.py
python examples/infty_minimal/minimal_visualization.py
```

## Writing your own example

A good example should include:

1. environment assumptions;
2. dataset or toy-data creation;
3. base optimizer construction;
4. INFTY optimizer construction;
5. closure definition;
6. `set_closure` and `step` calls;
7. expected output or artifact path.

## Example: C-Flat integration pattern

```python
base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = infty_optim.C_Flat(
    params=model.parameters(),
    base_optimizer=base_optimizer,
    model=model,
    args={"rho": 0.05, "lamb": 0.2},
)

for inputs, targets in loader:
    optimizer.set_closure(make_loss_fn(inputs, targets))
    logits, loss_list = optimizer.step()
```

## Example: loss-landscape visualization pattern

```python
from infty import plot as infty_plot

result = infty_plot.visualize_landscape(
    optimizer=optimizer,
    model=model,
    create_loss_fn=create_loss_fn,
    loader=train_loader,
    task=task_id,
    device=device,
    output_dir="workdirs/plots/diagnostics/landscape/demo",
)
print(result["plot_path"])
```

## Example: conflict visualization pattern

```python
from infty import plot as infty_plot

result = infty_plot.visualize_conflicts(
    optimizer=optimizer,
    task=task_id,
    output_dir="workdirs/plots/diagnostics/conflicts/demo",
)
print(result)
```

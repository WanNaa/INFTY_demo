# API Reference

This reference documents the public interfaces exposed by INFTY. It focuses on arguments, return values, and closure conventions rather than algorithm derivations.

## Imports

```python
from infty import optim as infty_optim
from infty import plot as infty_plot
from infty import analysis as infty_analysis
```

or:

```python
from infty.optim import C_Flat, ZeroFlow, UniGrad_FS
from infty.plot import visualize_landscape, visualize_esd, visualize_conflicts, visualize_trajectory
```

## Closure contract

Most optimizer APIs expect a closure with this format:

```python
def loss_fn():
    logits = model(inputs)
    loss_1 = criterion(logits, targets)
    return logits, [loss_1]
```

The returned value must be:

```python
(logits, loss_list)
```

where `loss_list` is a list of scalar tensors.

## `infty.optim`

### Public optimizer exports

| Class | Family | Description |
| --- | --- | --- |
| `InftyBaseOptimizer` | Base | Shared wrapper for geometry-aware optimizers. |
| `C_Flat` | Geometry reshaping | Flatness-aware continual-learning optimizer. |
| `GAM` | Geometry reshaping | Geometry-aware minimization optimizer. |
| `GSAM` | Geometry reshaping | Generalized SAM-style optimizer. |
| `SAM` | Geometry reshaping | Sharpness-aware minimization baseline. |
| `LookSAM` | Geometry reshaping | Look-ahead SAM-style optimizer. |
| `ZeroFlow` | Zeroth-order updates | Zeroth-order and forward-gradient optimizer. |
| `UniGrad_FS` | Gradient filtering | Unified gradient projection with flatter sharpness. |
| `GradVac` | Gradient filtering | Gradient vaccine-style conflict mitigation. |
| `OGD` | Gradient filtering | Orthogonal gradient descent-style method. |
| `PCGrad` | Gradient filtering | Projected conflicting gradients. |
| `CAGrad` | Gradient filtering | Conflict-averse gradient method. |

### `InftyBaseOptimizer`

```python
InftyBaseOptimizer(
    params,
    base_optimizer,
    model,
    adaptive=False,
    perturb_eps=1e-12,
    grad_reduce="mean",
    **kwargs,
)
```

Base class for geometry-aware INFTY optimizers.

| Parameter | Description |
| --- | --- |
| `params` | Iterable of parameters to optimize. Usually `model.parameters()`. |
| `base_optimizer` | PyTorch optimizer instance that performs the final parameter update. |
| `model` | PyTorch model. Used for running-stat handling and distributed `no_sync`. |
| `adaptive` | Whether perturbations should be scaled adaptively by parameter magnitude. |
| `perturb_eps` | Numerical stabilizer for perturbation scaling. |
| `grad_reduce` | Distributed gradient reduction mode: `"mean"` or `"sum"`. |

Methods:

```python
optimizer.set_closure(loss_fn)
optimizer.step(closure=None, delay=False)
optimizer.delay_step()
optimizer.post_process(train_loader=None)
optimizer.zero_grad(set_to_none=False)
```

### `C_Flat`

```python
C_Flat(
    params,
    base_optimizer,
    model,
    args,
    adaptive=False,
    perturb_eps=1e-12,
    grad_reduce="mean",
    **kwargs,
)
```

Flatness-aware optimizer for continual-learning workflows.

`args` dictionary:

| Key | Default | Description |
| --- | --- | --- |
| `strategy` | `"basic"` | Strategy variant. Supported values: `"basic"`, `"plus"`. |
| `rho` | `0.1` | Perturbation radius. |
| `lamb` | `0.2` | Gradient aggregation coefficient. |
| `rho_scheduler` | `None` | Optional scheduler. |
| `A` | `5.0` | Used by `"plus"` strategy. |
| `k` | `0.01` | Used by `"plus"` strategy. |
| `t0` | `80` | Used by `"plus"` strategy. |
| `cof` | `1.0` | Used by `"plus"` strategy. |

### `ZeroFlow`

```python
ZeroFlow(params, base_optimizer, model, args, **kwargs)
```

Zeroth-order and forward-gradient optimizer.

`args` dictionary:

| Key | Default | Description |
| --- | --- | --- |
| `q` | `1` | Number of random query directions. |
| `inftyopt` | `"zo_sgd"` | Update variant. |
| `perturbation_mode` | `"two_side"` | Finite-difference mode: `"one_side"` or `"two_side"`. |
| `zo_eps` | `0.001` | Perturbation scale. |
| `use_history_grad` | `False` | Whether to smooth estimated directions. |
| `alpha` | `0.9` | Smoothing coefficient. |
| `gradient_sparsity` | `None` | Optional sparsity mask ratio or per-parameter dictionary. |
| `memory_efficient` | `False` | Recompute random directions instead of storing them. |

Supported `inftyopt` values:

```text
zo_sgd
zo_adam
zo_sgd_sign
zo_adam_sign
zo_sgd_conserve
zo_adam_conserve
forward_grad
```

### `UniGrad_FS`

```python
UniGrad_FS(params, base_optimizer, model, args, **kwargs)
```

Gradient-filtering optimizer for two-objective continual-learning updates.

`args` dictionary:

| Key | Default | Description |
| --- | --- | --- |
| `task_id` | `0` if omitted | Current task index. |
| `utype` | `"model-wise"` | `"model-wise"` or `"layer-wise"`. |
| `k_idx` | `[-1]` | Gradient block indices. |
| `S_T` | `[0.1]` | Similarity threshold. |
| `beta` | `0.9` | Threshold update coefficient. |
| `rho` | `0.05` | Perturbation radius. |
| `perturb_eps` | `1e-12` | Numerical stabilizer. |
| `adaptive` | `False` | Adaptive perturbation flag. |

For `task_id > 0`, the closure must return exactly two loss terms:

```python
return logits, [old_loss, new_loss]
```

## `infty.plot`

Public exports:

```python
visualize_loss_landscape
visualize_landscape
visualize_esd
visualize_conflicts
visualize_trajectory
```

`visualize_landscape` is an alias of `visualize_loss_landscape`.

### `visualize_loss_landscape`

```python
visualize_loss_landscape(
    optimizer,
    model,
    create_loss_fn,
    loader,
    task,
    device,
    limit=0.1,
    samples=21,
    output_dir=None,
    dir_path=None,
)
```

Returns:

```python
{
    "eigen_path": "...",
    "loss_path": "...",
    "plot_path": "...",
}
```

### `visualize_esd`

```python
visualize_esd(
    optimizer,
    model,
    create_loss_fn,
    loader,
    task,
    device,
    output_dir=None,
    dir_path=None,
)
```

Returns:

```python
{
    "trace_path": "...",
    "esd_path": "...",
    "plot_path": "...",
}
```

### `visualize_conflicts`

```python
visualize_conflicts(
    optimizer,
    task=None,
    output_dir=None,
    task_id=None,
    dir_path=None,
)
```

The optimizer must expose `sim_list` or `sim_arr`.

### `visualize_trajectory`

```python
visualize_trajectory(
    optimizer_name,
    init=None,
    n_iter=10000,
    lr=0.1,
    output_dir=None,
    grid_size=500,
)
```

Supported `optimizer_name` values:

```text
sgd
adam
adamw
pcgrad
cagrad
unigrad
zo_adam
zo_adam_q4
zo_adam_sign
zo_adam_cons
```

## `infty.analysis`

Public exports:

```python
aggregate_records
analyze_surrogate_batch
plot_summary_records
save_records_as_csv
save_records_as_json
```

### `analyze_surrogate_batch`

```python
analyze_surrogate_batch(
    objective_fn,
    model,
    *,
    estimator_name="zo_sgd",
    zo_eps=1e-3,
    perturbation_mode="two_side",
    q=1,
    variance_seeds=4,
    random_seed=0,
    step_size=1e-3,
    gradient_sparsity=None,
)
```

Compares a zeroth-order surrogate gradient with a first-order gradient on a single objective batch.

### `aggregate_records`

```python
aggregate_records(records)
```

Aggregates per-batch surrogate-quality records by task and split.

### `save_records_as_json`

```python
save_records_as_json(output_path, payload)
```

### `save_records_as_csv`

```python
save_records_as_csv(output_path, rows)
```

### `plot_summary_records`

```python
plot_summary_records(summary_records, output_dir, title_prefix="Surrogate Gradient Quality")
```

Creates summary plots for surrogate-gradient quality metrics.

## Stability notes

INFTY is currently an alpha-stage research library. Prefer named arguments and record the commit hash used in experiments.

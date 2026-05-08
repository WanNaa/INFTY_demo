# Architecture

This page explains the structure of INFTY and the design choices behind the library.

## Module structure

```text
src/infty/
├── optim/
│   ├── geometry_reshaping/
│   ├── gradient_filtering/
│   └── zeroth_order_updates/
├── plot/
├── analysis/
└── utils/
```

## Optimizer families

### Geometry reshaping

Located under:

```text
src/infty/optim/geometry_reshaping/
```

These optimizers modify local loss geometry or perturbation behavior. They generally inherit from `InftyBaseOptimizer`.

### Gradient filtering

Located under:

```text
src/infty/optim/gradient_filtering/
```

These optimizers consume multiple losses and modify gradients to reduce objective conflicts. Shared utilities live in `EasyCLMultiObjOptimizer`.

### Zeroth-order updates

Located under:

```text
src/infty/optim/zeroth_order_updates/
```

These optimizers estimate update directions through finite differences, random perturbations, or forward-gradient methods.

## Optimizer wrapper flow

The typical training flow is:

```text
user training loop
  -> create closure
  -> optimizer.set_closure(loss_fn)
  -> optimizer.step()
       -> optimizer-specific gradient/perturbation logic
       -> base_optimizer.step()
```

This keeps the outer training loop simple while allowing each optimizer to control the number of closure evaluations and gradient operations.

## Closure format

The standard closure format is:

```python
def loss_fn():
    logits = model(inputs)
    loss = criterion(logits, targets)
    return logits, [loss]
```

The use of `loss_list` rather than a single loss allows optimizers to distinguish objectives, such as old-task and new-task losses.

## State handling

Many optimizers store temporary tensors in `self.state[p]` for each parameter `p`. Common state entries include:

- saved gradients;
- perturbation vectors;
- historical gradient estimates;
- similarity values.

When adding new state entries, use descriptive names and ensure temporary perturbations are reverted before returning from `step()`.

## Plotting architecture

Plotting utilities are pure diagnostic tools. They should:

- accept an explicit `output_dir`;
- restore model state after perturbing weights;
- return saved artifact paths;
- avoid hidden benchmark-specific assumptions.

## Analysis architecture

Analysis utilities should be independent of a specific benchmark. The current surrogate-quality analysis compares first-order and zeroth-order directions by measuring:

- gradient norms;
- cosine similarity;
- estimator variance;
- projected gradient statistics;
- estimated loss decrease.

## Public API exports

Public APIs should be exported through package-level `__init__.py` files:

```text
src/infty/optim/__init__.py
src/infty/plot/__init__.py
src/infty/analysis/__init__.py
```

This allows users to write stable imports such as:

```python
from infty.optim import C_Flat
from infty.plot import visualize_landscape
from infty.analysis import analyze_surrogate_batch
```

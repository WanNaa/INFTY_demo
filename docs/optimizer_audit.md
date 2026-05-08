# Optimizer Audit

## Scope

- `src/infty/optim/geometry_reshaping`
- `src/infty/optim/zeroth_order_updates`
- `src/infty/optim/gradient_filtering`

## Findings

1. `OGD` previously hard-coded `./ckp/ogd_basis.pt`, which leaked checkpoint state back into `examples/PILOT`. It now derives the basis path from `args["ckp_dir"]` and stores the resolved file as `ogd_basis.pt`.
2. `GradVac` and `UniGrad_FS` previously recorded similarity values under `sim_arr`, while the plot layer expected `sim_list`. Both optimizers now expose a `sim_list` compatibility property.
3. `GradVac.set_k_idx()` and `UniGrad_FS.set_k_idx()` previously multiplied `S_T` by the number of layers, which changed the threshold magnitude instead of expanding it. The threshold tensor is now expanded once to match the number of gradient blocks.

## Residual Risk

- This pass focused on deterministic path and interface bugs. It does not claim a full paper-by-paper numerical audit of every optimizer variant.

## Tests Added

- `tests/optim/test_geometry_reshaping.py`
- `tests/optim/test_zeroth_order_updates.py`
- `tests/optim/test_gradient_filtering.py`
- `tests/optim/test_ogd.py`

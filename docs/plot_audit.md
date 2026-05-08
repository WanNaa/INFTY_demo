# Plot Audit

## Scope

- `src/infty/plot/visualize_loss_landscape.py`
- `src/infty/plot/visualize_esd.py`
- `src/infty/plot/visualize_conflicts.py`
- `src/infty/plot/visualize_trajectory.py`

## Findings

1. Plot outputs no longer default to `./plots` or `examples/PILOT/plots`; they now resolve into `workdirs/plots/...`.
2. `visualize_conflicts()` now accepts both `sim_list` and `sim_arr`, saves artifacts under an optimizer-specific directory, and avoids crashing on empty similarity histories.
3. `visualize_trajectory()` previously referenced an undefined `traj` variable. It now computes the trajectory before plotting and accepts an explicit `output_dir`.
4. `visualize_loss_landscape()` and `visualize_esd()` now restore the model state and training mode after analysis.

## Residual Risk

- Numerical correctness of the Hessian visualizations was not re-derived from first principles in this pass.

## Tests Added

- `tests/plot/test_conflicts_smoke.py`
- `tests/plot/test_trajectory_smoke.py`
- `tests/plot/test_plot_exports.py`

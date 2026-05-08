# Migration Baseline

## Environment Notes

- Working directory: `/home/fengtao/8_Toolkit/INFTY_main`
- `git status --short --branch` could not run because this directory is not a Git repository.

## Migration Summary

### Moved

- `examples/infty_configs/` -> `workdirs/configs/infty/`
- `examples/PILOT/ckp/` -> `workdirs/checkpoints/pilot_ckp/`
- `examples/PILOT/logs/` -> `workdirs/logs/pilot_logs/`
- `examples/PILOT/plots/` -> `workdirs/plots/pilot_plots/`
- `examples/PILOT/metrics_json/` -> `workdirs/outputs/pilot_metrics_json/`
- `examples/PILOT/run_scripts/` -> `workdirs/run_scripts/legacy/pilot_run_scripts/`
- non-official PILOT helper scripts -> `workdirs/run_scripts/legacy/`

### Retained Differences

- `examples/PILOT/data/` was left in place because dataset paths are still hard-coded under `examples/PILOT/utils/data.py`; moving that cache would require a broader dataset-path refactor.

### Modified

- `examples/PILOT/main.py`
- `examples/PILOT/trainer.py`
- `src/infty/optim/gradient_filtering/ogd.py`
- `src/infty/optim/gradient_filtering/gradvac.py`
- `src/infty/optim/gradient_filtering/unigrad_fs.py`
- `src/infty/plot/...`
- `README.md`
- `.gitignore`

## Validation

- `python -m compileall src examples/PILOT tests`
  - passed
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/optim -q`
  - passed: `5 passed`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/plot -q`
  - skipped: `3 skipped`
  - reason: local `matplotlib` wheel is ABI-incompatible with installed NumPy 2.x
- `DRY_RUN=1 bash workdirs/run_scripts/run_geometry_reshaping.sh`
  - passed
- `DRY_RUN=1 bash workdirs/run_scripts/run_zeroth_order_updates.sh`
  - passed
- `DRY_RUN=1 bash workdirs/run_scripts/run_gradient_filtering.sh`
  - passed

## Additional Notes

- Plain `pytest` failed before test collection because the host environment auto-loaded a global Hydra pytest plugin that is not compatible with the installed Python 3.11 environment. Validation therefore used `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

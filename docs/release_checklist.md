# Release Checklist

Use this checklist before publishing a new INFTY package or paper artifact.

## Code

- [ ] All tests pass.
- [ ] Public APIs import correctly.
- [ ] Optimizer smoke tests run on CPU.
- [ ] Plot smoke tests run with small grids.
- [ ] Example scripts run or are clearly marked as requiring benchmark data.

## Documentation

- [ ] README links to `docs/`.
- [ ] `docs/user_guide.md` reflects current optimizer behavior.
- [ ] `docs/api_reference.md` reflects current public APIs.
- [ ] `docs/developer_guide.md` describes extension workflow.
- [ ] `docs/troubleshooting.md` includes known issues.
- [ ] MkDocs builds with `mkdocs build --strict`.
- [ ] Optional Sphinx docs build if used.

## Packaging

- [ ] Package version is updated.
- [ ] Homepage URL is correct.
- [ ] Dependencies are accurate.
- [ ] Optional example dependencies are accurate.
- [ ] License metadata is correct.

## Reproducibility

- [ ] Release commit hash is recorded.
- [ ] Experiment configs are archived.
- [ ] Random seeds and task orders are documented.
- [ ] Citation information is updated.

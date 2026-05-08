# Roadmap

This roadmap lists recommended improvements for making INFTY more robust as a public research library.

## Documentation roadmap

- [x] Add user guide.
- [x] Add API reference.
- [x] Add developer guide.
- [x] Add troubleshooting page.
- [x] Add examples page.
- [x] Add MkDocs configuration.
- [x] Add optional Sphinx/RST scaffold.
- [ ] Add method-specific tutorials for each optimizer.
- [ ] Add benchmark-specific reproduction guides.
- [ ] Add generated API pages from docstrings.

## Engineering roadmap

- [ ] Add one-step optimizer tests for every public optimizer.
- [ ] Add import tests for `infty.optim`, `infty.plot`, and `infty.analysis`.
- [ ] Add continuous integration for tests.
- [ ] Add documentation build checks.
- [ ] Verify package metadata and homepage URLs before release.
- [ ] Add release tags and changelog entries.

## Research usability roadmap

- [ ] Provide recommended hyperparameter ranges for each optimizer.
- [ ] Provide benchmark-specific configs for common continual-learning datasets.
- [ ] Add reproducibility checklists for paper experiments.
- [ ] Add expected runtime and memory notes for each optimizer family.
- [ ] Add ablation scripts for flatness, conflict, and zeroth-order components.

## API roadmap

- [ ] Stabilize constructor signatures across optimizer families.
- [ ] Add consistent docstrings to every public class.
- [ ] Add type hints where practical.
- [ ] Add migration notes when public APIs change.

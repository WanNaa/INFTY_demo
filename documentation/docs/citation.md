# Citation

If INFTY or its included algorithms are useful in your work, please cite the corresponding software and method papers.

## Software citation

For the toolkit, use a software citation similar to:

```bibtex
@software{infty_toolkit,
  title        = {INFTY: An Optimization Toolkit to Support Continual AI},
  author       = {INFTY contributors},
  year         = {2026},
  url          = {https://github.com/WanNaa/INFTY_demo},
  note         = {Research software, alpha release}
}
```

Replace the year, version, DOI, or official repository URL if a formal release is created.

## Algorithm papers

The README lists the following related works:

- **ZeroFlow**: *Zeroflow: Overcoming catastrophic forgetting is easier than you think.* ICML 2025.
- **C-Flat++**: *C-Flat++: Towards a More Efficient and Powerful Framework for Continual Learning.* arXiv 2025.
- **C-Flat**: *Make Continual Learning Stronger via C-Flat.* NeurIPS 2024.
- **UniGrad-FS**: *UniGrad-FS: Unified Gradient Projection With Flatter Sharpness for Continual Learning.* TII 2024.

For publications, cite the specific algorithm paper matching the optimizer used in your experiments.

## Reproducibility note

When citing INFTY in experiments, report:

- INFTY version or commit hash;
- optimizer name;
- optimizer `args` dictionary;
- base optimizer and hyperparameters;
- PyTorch version;
- benchmark and task order.

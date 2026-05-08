# INFTY Documentation

INFTY is an optimization-centric toolkit for Continual AI. It provides plug-and-play optimizers, gradient-conflict analysis utilities, and visualization tools for diagnosing optimization behavior in continual-learning systems.

The library is designed for research workflows where users need to replace or augment an existing PyTorch optimizer without rewriting the whole training pipeline.

[PyPI Package](https://pypi.org/project/infty/){ .md-button .md-button--primary }

## Documentation map

| Page | Purpose |
| --- | --- |
| [Installation](installation.md) | Install INFTY from PyPI or from source. |
| [Quick Start](quickstart.md) | Run the smallest optimizer examples and the PILOT demo. |
| [User Guide](user_guide.md) | Understand optimizer families, closures, continual-learning workflows, and visualization utilities. |
| [API Reference](api_reference.md) | Look up public classes, functions, arguments, return values, and expected closure contracts. |
| [Examples](examples.md) | Find runnable examples for C-Flat, ZeroFlow, UniGrad-FS, visualization, and PILOT integration. |
| [Developer Guide](developer_guide.md) | Extend the library with new optimizers, plots, and analysis modules. |
| [Architecture](architecture.md) | Understand INFTY's module layout and optimizer wrapper design. |
| [Troubleshooting](troubleshooting.md) | Resolve common installation, closure, training, and plotting issues. |
| [Citation](citation.md) | Cite INFTY and related algorithm papers. |

## Main concepts

INFTY follows four design ideas:

1. **Optimizer wrapping**: most INFTY optimizers wrap a standard PyTorch optimizer such as `torch.optim.SGD` or `torch.optim.Adam`.
2. **Closure-based training**: the training loss is supplied through a closure that returns `(logits, loss_list)`.
3. **Continual-learning awareness**: several optimizers explicitly handle flatness, gradient conflicts, or zeroth-order updates, which are common concerns in continual learning.
4. **Diagnostics as first-class utilities**: the package includes tools for loss landscapes, Hessian ESD, gradient-conflict curves, trajectory visualization, and surrogate-gradient quality analysis.

## Recommended reading order

New users should read:

1. [Installation](installation.md)
2. [Quick Start](quickstart.md)
3. [User Guide](user_guide.md)

Users integrating INFTY into an existing training framework should additionally read:

1. [API Reference](api_reference.md)
2. [Examples](examples.md)
3. [Troubleshooting](troubleshooting.md)

Contributors should read:

1. [Developer Guide](developer_guide.md)
2. [Architecture](architecture.md)
3. [Roadmap](roadmap.md)

## Current status

INFTY is an alpha-stage research library. The core optimizer interfaces are usable, but some APIs may evolve as new continual-learning use cases are added. For experiments intended for publication, record the exact INFTY version or commit hash.

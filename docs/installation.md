# Installation

This page explains how to install INFTY for normal use, example execution, and local development.

## Requirements

INFTY is a Python package built around PyTorch. The package metadata declares the following core dependencies:

- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `scipy`
- `seaborn`
- `tqdm`

Although the package metadata allows Python `>=3.6`, a modern Python environment such as Python 3.8--3.11 is recommended because current PyTorch releases are usually tested against newer Python versions.

## Install from PyPI

```bash
python -m pip install --upgrade pip
python -m pip install infty
```

Verify the installation:

```bash
python -c "from infty.optim import C_Flat, ZeroFlow, UniGrad_FS; print('INFTY import succeeded')"
```

## Install from source

```bash
git clone https://github.com/WanNaa/INFTY_demo.git
cd INFTY_demo
python -m pip install --upgrade pip
python -m pip install .
```

For editable development:

```bash
python -m pip install -e .
```

## Install example dependencies

Some examples require optional packages such as `easydict`, `Pillow`, `PyYAML`, `scikit-learn`, `timm`, and `umap-learn`.

```bash
python -m pip install .[examples]
```

For editable development with examples:

```bash
python -m pip install -e .[examples]
```

## Recommended conda environment

```bash
conda create -n infty python=3.10
conda activate infty
python -m pip install --upgrade pip
python -m pip install infty
```

If you need a specific CUDA build of PyTorch, install PyTorch first using the command matching your CUDA/runtime setup, then install INFTY.

## Documentation dependencies

To build the documentation website locally:

```bash
python -m pip install -r requirements-docs.txt
mkdocs serve
```

To build the optional Sphinx/RST documentation:

```bash
python -m pip install -r requirements-docs.txt
sphinx-build -b html docs/sphinx docs/sphinx/_build/html
```

## Common checks

Check that `infty` is imported from the expected location:

```bash
python -c "import infty; print(infty.__file__)"
```

Check optimizer exports:

```bash
python -c "from infty.optim import C_Flat, ZeroFlow, UniGrad_FS; print(C_Flat, ZeroFlow, UniGrad_FS)"
```

Check plotting exports:

```bash
python -c "from infty.plot import visualize_landscape, visualize_esd, visualize_conflicts, visualize_trajectory; print('plot utilities imported')"
```

## Troubleshooting

If installation fails, see [Troubleshooting](troubleshooting.md). The most common causes are incompatible PyTorch/CUDA versions, installing into the wrong Python environment, or missing optional example dependencies.

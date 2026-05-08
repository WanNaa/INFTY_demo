# INFTY Documentation Package

This package provides a formal documentation structure for the INFTY project. It is designed to address reviewer feedback that the library lacks an API reference, a user guide, and a developer guide.

## How to use

Copy the contents of this package into the repository root:

```bash
cp -r docs examples .github mkdocs.yml requirements-docs.txt .readthedocs.yml Makefile.docs CONTRIBUTING.md README_DOCS_SNIPPET.md /path/to/INFTY_demo/
```

Then copy the content of `README_DOCS_SNIPPET.md` into the repository `README.md`, preferably near the top after the project summary.

## Markdown vs RST

The main documentation is in Markdown under `docs/*.md`, because GitHub renders Markdown directly and MkDocs can publish it as a clean website through GitHub Pages.

A lightweight Sphinx/reStructuredText setup is also included under `docs/sphinx/`. This is optional. It is useful if you later want ReadTheDocs-style auto-generated API pages from Python modules.

Recommended strategy:

- Use `docs/*.md` as the official human-written documentation.
- Use `docs/sphinx/*.rst` only if you want generated API pages or ReadTheDocs compatibility.
- Use MkDocs + GitHub Pages for the public documentation website.

## Local preview

```bash
python -m pip install -r requirements-docs.txt
mkdocs serve
```

Optional Sphinx build:

```bash
python -m pip install -r requirements-docs.txt
sphinx-build -b html docs/sphinx docs/sphinx/_build/html
```

## Included structure

See `PROJECT_TREE.txt` for the complete file tree.

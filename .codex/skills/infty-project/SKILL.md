---
name: infty-project
description: Repository-local guidance for working in the INFTY codebase. Use when Codex edits code, installs dependencies, runs tests, builds docs, or executes Python tooling in this repository and needs project-specific environment conventions.
---

# INFTY Project

## Overview

Follow the repository-specific execution rules before running project commands.
Use the `infty` virtual environment for Python-related work in this repository.

## Use The `infty` Environment

- Activate the `infty` environment before running `python`, `pip`, `pytest`, `mkdocs`, or packaging commands for this project.
- Keep using the active `infty` interpreter throughout the task instead of switching Python environments mid-session.
- Prefer `python -m ...` style commands when installing or invoking Python tooling so the active interpreter is explicit.

## Command Pattern

Use this pattern for project commands:

```bash
conda activate infty
python -m pip install -e .
pytest
```

If the shell is already running inside `infty`, continue with that interpreter and avoid reconfiguring the environment unnecessarily.

## Report Environment Usage

When summarizing work, state whether validation or tooling commands were run inside the `infty` environment.

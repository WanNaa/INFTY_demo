#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#*

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from infty.utils.hessian import hessian


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "workdirs" / "plots" / "esd"


def visualize_esd(optimizer, model, create_loss_fn, loader, task, device, output_dir=None, dir_path=None):
    output_dir = Path(output_dir) if output_dir is not None else Path(dir_path) if dir_path is not None else DEFAULT_OUTPUT_DIR
    output_dir = output_dir.expanduser().resolve()
    optimizer_name = getattr(optimizer, "name", optimizer.__class__.__name__.lower())
    save_dir = output_dir / optimizer_name
    save_dir.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    state_backup = {k: v.detach().clone() for k, v in model.state_dict().items()}

    try:
        model.eval()
        print(f"{'=' * 30}\n[ESD] Computing Hessian trace and spectrum for task {task}...\n{'=' * 30}")
        hessian_comp = hessian(model, create_loss_fn, dataloader=loader, device=device)

        trace_path = save_dir / f"trace_task{task}.pt"
        esd_path = save_dir / f"esd_task{task}.pt"
        fig_path = save_dir / f"fig_task{task}.pdf"

        if not trace_path.exists():
            print(f"[ESD] Estimating Hessian trace for task {task}...")
            trace = hessian_comp.trace()
            mean_trace = np.mean(trace)
            print(f"[ESD] Mean Hessian trace: {mean_trace:.4f}")
            torch.save({"mean_trace": mean_trace}, trace_path)
        else:
            trace = torch.load(trace_path, map_location="cpu")
            mean_trace = trace.get("mean_trace", None)
            if mean_trace is not None:
                print(f"[ESD] Loaded mean Hessian trace: {mean_trace:.4f}")
            else:
                print(f"[ESD] Warning: 'mean_trace' not found in {trace_path}")

        if not esd_path.exists():
            print(f"[ESD] Estimating Empirical Spectral Density (ESD) for task {task}...")
            density_eigen, density_weight = hessian_comp.density()
            torch.save({"density_eigen": density_eigen, "density_weight": density_weight}, esd_path)
            print(f"[ESD] ESD data saved to '{esd_path}'.")
        else:
            print(f"[ESD] ESD file found at '{esd_path}', skipping ESD computation.")
            esd_data = torch.load(esd_path, map_location="cpu")
            density_eigen = esd_data["density_eigen"]
            density_weight = esd_data["density_weight"]

        print(f"[ESD] Plotting ESD for task {task}...")
        get_esd_plot(density_eigen, density_weight, fig_path)
        print(f"[ESD] Done.\n{'=' * 30}")
        return {
            "trace_path": str(trace_path),
            "esd_path": str(esd_path),
            "plot_path": str(fig_path),
        }
    finally:
        model.load_state_dict(state_backup, strict=True)
        model.train(was_training)


def get_esd_plot(eigenvalues, weights, fig_path):
    plt.clf()
    fig, ax = plt.subplots()

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"

    fontsize = 28
    density, grids = density_generate(eigenvalues, weights)
    plt.semilogy(grids, density + 1.0e-7)
    plt.ylabel("Density (Log Scale)", fontsize=fontsize, labelpad=10)
    plt.xlabel("Eigenvalue", fontsize=fontsize, labelpad=10)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.axis([np.min(eigenvalues) - 1, np.max(eigenvalues) + 1, None, None])

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("serif")
    ax.tick_params(axis="both", labelsize=fontsize, which="major", direction="out", length=6, width=2)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(2)

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)


def density_generate(eigenvalues, weights, num_bins=10000, sigma_squared=1e-5, overhead=0.01):
    eigenvalues = np.array(eigenvalues)
    weights = np.array(weights)

    lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
    lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead

    grids = np.linspace(lambda_min, lambda_max, num=num_bins)
    sigma = sigma_squared * max(1, (lambda_max - lambda_min))

    num_runs = eigenvalues.shape[0]
    density_output = np.zeros((num_runs, num_bins))

    for i in range(num_runs):
        for j in range(num_bins):
            x = grids[j]
            tmp_result = gaussian(eigenvalues[i, :], x, sigma)
            density_output[i, j] = np.sum(tmp_result * weights[i, :])
    density = np.mean(density_output, axis=0)
    normalization = np.sum(density) * (grids[1] - grids[0])
    density = density / normalization
    return density, grids


def gaussian(x, x0, sigma_squared):
    return np.exp(-(x0 - x) ** 2 / (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)

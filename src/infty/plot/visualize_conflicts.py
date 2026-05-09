from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .paths import DEFAULT_CONFLICTS_DIR, ensure_parent_dir


DEFAULT_OUTPUT_DIR = DEFAULT_CONFLICTS_DIR


def _get_similarity_values(optimizer):
    sim_values = getattr(optimizer, "sim_list", None)
    if sim_values is None:
        sim_values = getattr(optimizer, "sim_arr", None)
    if sim_values is None:
        raise AttributeError("optimizer must expose sim_list or sim_arr for conflict visualization")
    return np.asarray([float(value) for value in sim_values], dtype=np.float32)


def visualize_conflicts(optimizer, task=None, output_dir=None, task_id=None, dir_path=None):
    if task is None:
        task = task_id if task_id is not None else 0
    if output_dir is None:
        output_dir = dir_path

    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    output_dir = output_dir.expanduser().resolve()

    optimizer_name = getattr(optimizer, "name", optimizer.__class__.__name__.lower())
    save_dir = output_dir / optimizer_name

    sim_values = _get_similarity_values(optimizer)
    sim_path = save_dir / f"sim_list_task{task}.pt"
    ensure_parent_dir(sim_path)
    torch.save(sim_values.tolist(), sim_path)

    if task <= 0:
        return {"optimizer_name": optimizer_name, "task": int(task), "sim_path": str(sim_path)}

    if sim_values.size == 0:
        warning_path = save_dir / f"warning_task{task}.txt"
        ensure_parent_dir(warning_path)
        warning_path.write_text("No similarity values were recorded for conflict visualization.\n", encoding="utf-8")
        return {
            "optimizer_name": optimizer_name,
            "task": int(task),
            "sim_path": str(sim_path),
            "warning_path": str(warning_path),
        }

    figure_path = save_dir / f"similarity_{task}.pdf"
    iterations = np.arange(1, sim_values.size + 1)

    plt.figure(figsize=(12, 8))
    plt.plot(iterations, sim_values, "b-", linewidth=2, alpha=0.8, label="Cosine Similarity")
    plt.grid(True, alpha=0.3)

    fontsize = 24
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("Iterations", fontsize=fontsize)
    plt.ylabel("Cosine Similarity", fontsize=fontsize)

    if sim_values.size > 1:
        plt.xlim(1, sim_values.size)
    y_min = float(sim_values.min())
    y_max = float(sim_values.max())
    y_pad = max(0.05, abs(y_max - y_min) * 0.05)
    plt.ylim(y_min - y_pad, y_max + y_pad)

    mean_sim = float(np.mean(sim_values))
    std_sim = float(np.std(sim_values))
    plt.text(
        0.02,
        0.98,
        f"Mean: {mean_sim:.4f}\nStd: {std_sim:.4f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=fontsize,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    ensure_parent_dir(figure_path)
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "optimizer_name": optimizer_name,
        "task": int(task),
        "sim_path": str(sim_path),
        "plot_path": str(figure_path),
    }

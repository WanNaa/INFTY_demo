import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .paths import DEFAULT_CONFLICTS_DIR, ensure_parent_dir
from .source_utils import resolve_conflict_records, resolve_source_name


DEFAULT_OUTPUT_DIR = DEFAULT_CONFLICTS_DIR

CONFLICT_RECORD_FIELDS = [
    "task",
    "iter",
    "block",
    "cos_before",
    "cos_after",
    "delta_cos",
    "norm_old",
    "norm_new",
    "norm_ratio_old_new",
    "threshold",
    "conflict",
    "below_threshold",
    "gain_old",
    "gain_new",
    "min_gain",
    "w1",
    "w2",
]


def _to_python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_conflict_records(conflict_records, task=None):
    normalized = []
    target_task = None if task is None else int(task)

    for index, raw_record in enumerate(conflict_records):
        raw_task = raw_record.get("task", task if task is not None else 0)
        task_value = int(raw_task)
        if target_task is not None and task_value != target_task:
            continue

        cos_before = float(raw_record.get("cos_before", raw_record.get("similarity", 0.0)))
        threshold = float(raw_record.get("threshold", 0.0))
        cos_after = float(raw_record.get("cos_after", cos_before))
        gain_old = float(raw_record.get("gain_old", 0.0))
        gain_new = float(raw_record.get("gain_new", 0.0))
        normalized.append(
            {
                "task": task_value,
                "iter": int(raw_record.get("iter", index)),
                "block": int(raw_record.get("block", 0)),
                "cos_before": cos_before,
                "cos_after": cos_after,
                "delta_cos": float(raw_record.get("delta_cos", cos_after - cos_before)),
                "norm_old": float(raw_record.get("norm_old", 0.0)),
                "norm_new": float(raw_record.get("norm_new", 0.0)),
                "norm_ratio_old_new": float(raw_record.get("norm_ratio_old_new", 0.0)),
                "threshold": threshold,
                "conflict": bool(raw_record.get("conflict", cos_before < 0.0)),
                "below_threshold": bool(raw_record.get("below_threshold", cos_before < threshold)),
                "gain_old": gain_old,
                "gain_new": gain_new,
                "min_gain": float(raw_record.get("min_gain", min(gain_old, gain_new))),
                "w1": float(raw_record.get("w1", 0.0)),
                "w2": float(raw_record.get("w2", 0.0)),
            }
        )

    normalized.sort(key=lambda record: (record["iter"], record["block"]))
    return normalized


def _write_conflict_records_csv(records, output_path):
    ensure_parent_dir(output_path)
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONFLICT_RECORD_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: _to_python_scalar(record.get(field)) for field in CONFLICT_RECORD_FIELDS})


def summarize_conflict_records(records):
    if not records:
        return {}

    eps = 1e-12
    cos_before = np.asarray([record["cos_before"] for record in records], dtype=np.float64)
    cos_after = np.asarray([record["cos_after"] for record in records], dtype=np.float64)
    thresholds = np.asarray([record["threshold"] for record in records], dtype=np.float64)
    gain_old = np.asarray([record["gain_old"] for record in records], dtype=np.float64)
    gain_new = np.asarray([record["gain_new"] for record in records], dtype=np.float64)
    min_gain = np.asarray([record["min_gain"] for record in records], dtype=np.float64)
    blocks = sorted({int(record["block"]) for record in records})
    iterations = sorted({int(record["iter"]) for record in records})

    return {
        "task": int(records[0]["task"]),
        "num_records": int(len(records)),
        "num_iterations": int(len(iterations)),
        "num_blocks": int(len(blocks)),
        "blocks": blocks,
        "mean_cos_before": float(cos_before.mean()),
        "mean_cos_after": float(cos_after.mean()),
        "mean_delta_cos": float((cos_after - cos_before).mean()),
        "conflict_rate": float(np.mean(cos_before < 0.0)),
        "below_threshold_rate": float(np.mean(cos_before < thresholds)),
        "conflict_severity": float(np.maximum(0.0, -cos_before).mean()),
        "mean_gain_old": float(gain_old.mean()),
        "mean_gain_new": float(gain_new.mean()),
        "mean_min_gain": float(min_gain.mean()),
        "mean_threshold": float(thresholds.mean()),
        "cos_before_std": float(cos_before.std(ddof=0) + eps - eps),
        "cos_after_std": float(cos_after.std(ddof=0) + eps - eps),
    }


def _aggregate_by_iteration(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[int(record["iter"])].append(record)

    rows = []
    for iteration, group_records in sorted(grouped.items()):
        cos_before = np.asarray([record["cos_before"] for record in group_records], dtype=np.float64)
        cos_after = np.asarray([record["cos_after"] for record in group_records], dtype=np.float64)
        thresholds = np.asarray([record["threshold"] for record in group_records], dtype=np.float64)
        gain_old = np.asarray([record["gain_old"] for record in group_records], dtype=np.float64)
        gain_new = np.asarray([record["gain_new"] for record in group_records], dtype=np.float64)
        min_gain = np.asarray([record["min_gain"] for record in group_records], dtype=np.float64)

        rows.append(
            {
                "iter": int(iteration),
                "cos_before_mean": float(cos_before.mean()),
                "cos_after_mean": float(cos_after.mean()),
                "delta_cos_mean": float((cos_after - cos_before).mean()),
                "threshold_mean": float(thresholds.mean()),
                "conflict_rate": float(np.mean(cos_before < 0.0)),
                "below_threshold_rate": float(np.mean(cos_before < thresholds)),
                "conflict_severity": float(np.maximum(0.0, -cos_before).mean()),
                "gain_old_mean": float(gain_old.mean()),
                "gain_new_mean": float(gain_new.mean()),
                "min_gain_mean": float(min_gain.mean()),
            }
        )
    return rows


def _build_heatmap(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[int(record["block"])].append(record)

    blocks = sorted(grouped)
    metric_names = [
        "mean_cos_before",
        "mean_cos_after",
        "conflict_rate",
        "below_threshold_rate",
        "mean_delta_cos",
        "mean_min_gain",
    ]
    matrix = np.zeros((len(metric_names), len(blocks)), dtype=np.float64)

    for block_index, block in enumerate(blocks):
        block_records = grouped[block]
        cos_before = np.asarray([record["cos_before"] for record in block_records], dtype=np.float64)
        cos_after = np.asarray([record["cos_after"] for record in block_records], dtype=np.float64)
        thresholds = np.asarray([record["threshold"] for record in block_records], dtype=np.float64)
        min_gain = np.asarray([record["min_gain"] for record in block_records], dtype=np.float64)
        matrix[:, block_index] = [
            float(cos_before.mean()),
            float(cos_after.mean()),
            float(np.mean(cos_before < 0.0)),
            float(np.mean(cos_before < thresholds)),
            float((cos_after - cos_before).mean()),
            float(min_gain.mean()),
        ]

    return blocks, metric_names, matrix


def _save_figure(fig, pdf_path, png_path):
    ensure_parent_dir(pdf_path)
    fig.savefig(pdf_path, bbox_inches="tight")
    ensure_parent_dir(png_path)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _plot_pre_post(iteration_rows, task, save_dir):
    iterations = np.asarray([row["iter"] for row in iteration_rows], dtype=np.int64)

    fig, axis = plt.subplots(figsize=(11, 6))
    axis.plot(iterations, [row["cos_before_mean"] for row in iteration_rows], label="Cos Before", linewidth=2)
    axis.plot(iterations, [row["cos_after_mean"] for row in iteration_rows], label="Cos After", linewidth=2)
    axis.plot(iterations, [row["threshold_mean"] for row in iteration_rows], label="Threshold", linewidth=2, linestyle="--")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Cosine Similarity")
    axis.set_title(f"Conflict Diagnostics (Task {task})")
    axis.grid(True, alpha=0.3)
    axis.legend(frameon=False)
    fig.tight_layout()

    pdf_path = save_dir / f"conflict_pre_post_task{task}.pdf"
    png_path = save_dir / f"conflict_pre_post_task{task}.png"
    _save_figure(fig, pdf_path, png_path)
    return str(pdf_path), str(png_path)


def _plot_rates(iteration_rows, task, save_dir):
    iterations = np.asarray([row["iter"] for row in iteration_rows], dtype=np.int64)

    fig, axis = plt.subplots(figsize=(11, 6))
    axis.plot(iterations, [row["conflict_rate"] for row in iteration_rows], label="Conflict Rate", linewidth=2)
    axis.plot(
        iterations,
        [row["below_threshold_rate"] for row in iteration_rows],
        label="Below Threshold Rate",
        linewidth=2,
    )
    axis.plot(iterations, [row["conflict_severity"] for row in iteration_rows], label="Conflict Severity", linewidth=2)
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Rate / Severity")
    axis.set_title(f"Conflict Rates (Task {task})")
    axis.grid(True, alpha=0.3)
    axis.legend(frameon=False)
    fig.tight_layout()

    pdf_path = save_dir / f"conflict_rate_task{task}.pdf"
    png_path = save_dir / f"conflict_rate_task{task}.png"
    _save_figure(fig, pdf_path, png_path)
    return str(pdf_path), str(png_path)


def _plot_gains(iteration_rows, task, save_dir):
    iterations = np.asarray([row["iter"] for row in iteration_rows], dtype=np.int64)

    fig, axis = plt.subplots(figsize=(11, 6))
    axis.plot(iterations, [row["gain_old_mean"] for row in iteration_rows], label="Gain Old", linewidth=2)
    axis.plot(iterations, [row["gain_new_mean"] for row in iteration_rows], label="Gain New", linewidth=2)
    axis.plot(iterations, [row["min_gain_mean"] for row in iteration_rows], label="Min Gain", linewidth=2)
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Update Gain")
    axis.set_title(f"Update Gains (Task {task})")
    axis.grid(True, alpha=0.3)
    axis.legend(frameon=False)
    fig.tight_layout()

    pdf_path = save_dir / f"update_gain_task{task}.pdf"
    png_path = save_dir / f"update_gain_task{task}.png"
    _save_figure(fig, pdf_path, png_path)
    return str(pdf_path), str(png_path)


def _plot_heatmap(records, task, save_dir):
    blocks, metric_names, matrix = _build_heatmap(records)

    fig, axis = plt.subplots(figsize=(max(8, len(blocks) * 1.2), 5))
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm")
    axis.set_xticks(np.arange(len(blocks)))
    axis.set_xticklabels(blocks)
    axis.set_yticks(np.arange(len(metric_names)))
    axis.set_yticklabels(metric_names)
    axis.set_xlabel("Block")
    axis.set_ylabel("Metric")
    axis.set_title(f"Layer Conflict Heatmap (Task {task})")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()

    pdf_path = save_dir / f"layer_conflict_heatmap_task{task}.pdf"
    png_path = save_dir / f"layer_conflict_heatmap_task{task}.png"
    _save_figure(fig, pdf_path, png_path)
    return str(pdf_path), str(png_path)


def visualize_conflict_diagnostics(
    optimizer=None,
    task=None,
    output_dir=None,
    task_id=None,
    dir_path=None,
    *,
    source_name=None,
    conflict_records=None,
):
    if task is None:
        task = task_id if task_id is not None else None
    if output_dir is None:
        output_dir = dir_path

    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    output_dir = output_dir.expanduser().resolve()
    optimizer_name = resolve_source_name(source=optimizer, source_name=source_name, default="custom")
    save_dir = output_dir / optimizer_name

    conflict_records = resolve_conflict_records(source=optimizer, conflict_records=conflict_records)
    if conflict_records is None:
        raise AttributeError("source must expose conflict_records, or conflict_records must be provided")

    records = _normalize_conflict_records(conflict_records, task=task)
    if not records:
        warning_path = save_dir / f"warning_task{0 if task is None else int(task)}.txt"
        ensure_parent_dir(warning_path)
        warning_path.write_text("No structured conflict records were recorded for diagnostics.\n", encoding="utf-8")
        return {
            "optimizer_name": optimizer_name,
            "task": None if task is None else int(task),
            "warning_path": str(warning_path),
        }

    task_value = int(records[0]["task"])
    records_csv_path = save_dir / f"conflict_records_task{task_value}.csv"
    summary_json_path = save_dir / f"conflict_summary_task{task_value}.json"
    _write_conflict_records_csv(records, records_csv_path)

    summary = summarize_conflict_records(records)
    summary["optimizer_name"] = optimizer_name
    ensure_parent_dir(summary_json_path)
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    iteration_rows = _aggregate_by_iteration(records)
    plot_paths = []
    plot_paths.extend(_plot_pre_post(iteration_rows, task_value, save_dir))
    plot_paths.extend(_plot_rates(iteration_rows, task_value, save_dir))
    plot_paths.extend(_plot_gains(iteration_rows, task_value, save_dir))
    plot_paths.extend(_plot_heatmap(records, task_value, save_dir))

    return {
        "optimizer_name": optimizer_name,
        "task": task_value,
        "records_csv_path": str(records_csv_path),
        "summary_json_path": str(summary_json_path),
        "plot_paths": plot_paths,
    }

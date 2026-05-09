import csv
import io
import json
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch

from infty.plot.paths import ensure_parent_dir
from infty.utils.running import fast_random_mask_like


def _safe_float(value):
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def _flatten_tensors(tensors):
    if not tensors:
        return torch.zeros(0)
    return torch.cat([tensor.reshape(-1) for tensor in tensors], dim=0)


def _clone_param_data(params):
    return [param.detach().clone() for param in params]


def _restore_param_data(params, saved_data):
    with torch.no_grad():
        for param, saved in zip(params, saved_data):
            param.copy_(saved)


def _apply_direction(params, direction_tensors, scale):
    with torch.no_grad():
        for param, direction in zip(params, direction_tensors):
            param.add_(direction, alpha=scale)


def _evaluate_objective(objective_fn):
    logits, loss_list = objective_fn()
    losses = [loss.reshape(()) for loss in loss_list]
    if not losses:
        raise ValueError("Objective function returned an empty loss list.")
    total_loss = torch.stack(losses).sum()
    return logits, total_loss, [_safe_float(loss) for loss in losses]


def _evaluate_loss_value(objective_fn):
    with torch.no_grad():
        _, total_loss, loss_components = _evaluate_objective(objective_fn)
    return _safe_float(total_loss), loss_components


def _collect_named_params(model):
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def _get_grad_sparsity_by_name(gradient_sparsity, name):
    if gradient_sparsity is None:
        return None
    if isinstance(gradient_sparsity, float):
        return gradient_sparsity
    if isinstance(gradient_sparsity, dict):
        return gradient_sparsity.get(name, None)
    return None


def _sample_noise_like(named_params, random_seed, gradient_sparsity=None):
    torch.manual_seed(int(random_seed))
    device_type = named_params[0][1].device.type if named_params else "cpu"
    sparse_rng = torch.Generator(device=device_type)
    sparse_rng.manual_seed(int(random_seed) + 104729)

    noises = []
    for name, param in named_params:
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=param.shape,
            device=param.device,
            dtype=param.dtype,
        )
        grad_sparsity = _get_grad_sparsity_by_name(gradient_sparsity, name)
        if grad_sparsity is not None:
            noise = noise.clone()
            noise[fast_random_mask_like(noise, grad_sparsity, generator=sparse_rng)] = 0
        noises.append(noise)
    return noises


def _compute_first_order_gradient(objective_fn, params):
    _, total_loss, loss_components = _evaluate_objective(objective_fn)
    grad_tensors = torch.autograd.grad(total_loss, params, allow_unused=True)
    grads = []
    for param, grad_tensor in zip(params, grad_tensors):
        if grad_tensor is None:
            grads.append(torch.zeros_like(param))
        else:
            grads.append(grad_tensor.detach().clone())
    flat_grad = _flatten_tensors(grads)
    return {
        "loss": _safe_float(total_loss),
        "loss_components": loss_components,
        "grad_tensors": grads,
        "grad_flat": flat_grad,
        "grad_norm": _safe_float(flat_grad.norm(p=2)),
    }


def _single_zo_estimate(
    objective_fn,
    named_params,
    zo_eps,
    perturbation_mode,
    q,
    random_seed,
    use_sign=False,
    gradient_sparsity=None,
):
    params = [param for _, param in named_params]
    saved_data = _clone_param_data(params)
    rng = np.random.RandomState(int(random_seed))
    grad_accumulator = [torch.zeros_like(param) for param in params]
    projected_grads = []

    try:
        for q_index in range(int(q)):
            local_seed = int(rng.randint(0, 2**31 - 1)) + q_index
            noises = _sample_noise_like(named_params, local_seed, gradient_sparsity=gradient_sparsity)
            _apply_direction(params, noises, zo_eps)
            loss_plus, _ = _evaluate_loss_value(objective_fn)

            if perturbation_mode == "one_side":
                _apply_direction(params, noises, -zo_eps)
                loss_ref, _ = _evaluate_loss_value(objective_fn)
                projected_grad = (loss_plus - loss_ref) / zo_eps
            elif perturbation_mode == "two_side":
                _apply_direction(params, noises, -2.0 * zo_eps)
                loss_minus, _ = _evaluate_loss_value(objective_fn)
                projected_grad = (loss_plus - loss_minus) / (2.0 * zo_eps)
                _apply_direction(params, noises, zo_eps)
            else:
                raise ValueError(f"Unsupported perturbation mode: {perturbation_mode}")

            scale = np.sign(projected_grad) if use_sign else projected_grad
            projected_grads.append(float(projected_grad))
            for grad_tensor, noise in zip(grad_accumulator, noises):
                grad_tensor.add_(noise, alpha=float(scale) / float(q))
    finally:
        _restore_param_data(params, saved_data)

    grad_flat = _flatten_tensors(grad_accumulator)
    return {
        "grad_tensors": [tensor.detach().clone() for tensor in grad_accumulator],
        "grad_flat": grad_flat,
        "grad_norm": _safe_float(grad_flat.norm(p=2)),
        "projected_grad_mean": float(np.mean(projected_grads)) if projected_grads else 0.0,
        "projected_grad_std": float(np.std(projected_grads)) if projected_grads else 0.0,
    }


def _compute_loss_after_step(objective_fn, params, direction_tensors, step_size):
    saved_data = _clone_param_data(params)
    try:
        _apply_direction(params, direction_tensors, -float(step_size))
        loss_after, loss_components_after = _evaluate_loss_value(objective_fn)
    finally:
        _restore_param_data(params, saved_data)
    return loss_after, loss_components_after


def analyze_surrogate_batch(
    objective_fn,
    model,
    *,
    estimator_name="zo_sgd",
    zo_eps=1e-3,
    perturbation_mode="two_side",
    q=1,
    variance_seeds=4,
    random_seed=0,
    step_size=1e-3,
    gradient_sparsity=None,
):
    if estimator_name == "forward_grad":
        raise ValueError("This analysis module currently supports zeroth-order estimators only, not forward_grad.")
    if int(variance_seeds) < 1:
        raise ValueError("variance_seeds must be at least 1.")
    if int(q) < 1:
        raise ValueError("q must be at least 1.")

    named_params = _collect_named_params(model)
    params = [param for _, param in named_params]
    if not params:
        raise ValueError("No trainable parameters found for surrogate analysis.")

    first_order = _compute_first_order_gradient(objective_fn, params)

    use_sign = "sign" in estimator_name
    zo_estimates = []
    for seed_offset in range(int(variance_seeds)):
        zo_estimates.append(
            _single_zo_estimate(
                objective_fn=objective_fn,
                named_params=named_params,
                zo_eps=zo_eps,
                perturbation_mode=perturbation_mode,
                q=q,
                random_seed=int(random_seed) + seed_offset,
                use_sign=use_sign,
                gradient_sparsity=gradient_sparsity,
            )
        )

    zo_stack = torch.stack([estimate["grad_flat"] for estimate in zo_estimates], dim=0)
    zo_mean_flat = zo_stack.mean(dim=0)
    zo_mean_tensors = []
    cursor = 0
    for param in params:
        next_cursor = cursor + param.numel()
        zo_mean_tensors.append(zo_mean_flat[cursor:next_cursor].view_as(param).detach().clone())
        cursor = next_cursor

    centered = zo_stack - zo_mean_flat.unsqueeze(0)
    variance_trace = centered.pow(2).sum(dim=1).mean()
    variance_mean_coordinate = centered.pow(2).mean()

    fo_flat = first_order["grad_flat"]
    fo_norm = fo_flat.norm(p=2)
    zo_norm = zo_mean_flat.norm(p=2)
    cosine = torch.dot(fo_flat, zo_mean_flat) / (fo_norm * zo_norm + 1e-12)

    base_loss = first_order["loss"]
    loss_after_fo, fo_components_after = _compute_loss_after_step(
        objective_fn, params, first_order["grad_tensors"], step_size
    )
    loss_after_zo, zo_components_after = _compute_loss_after_step(
        objective_fn, params, zo_mean_tensors, step_size
    )

    return {
        "base_loss": float(base_loss),
        "loss_components": first_order["loss_components"],
        "fo_grad_norm": float(first_order["grad_norm"]),
        "zo_grad_norm": float(_safe_float(zo_norm)),
        "cosine_similarity": float(_safe_float(cosine)),
        "norm_ratio": float(_safe_float(zo_norm / (fo_norm + 1e-12))),
        "variance_trace": float(_safe_float(variance_trace)),
        "variance_mean_coordinate": float(_safe_float(variance_mean_coordinate)),
        "projected_grad_mean": float(np.mean([estimate["projected_grad_mean"] for estimate in zo_estimates])),
        "projected_grad_std": float(np.mean([estimate["projected_grad_std"] for estimate in zo_estimates])),
        "loss_after_fo_step": float(loss_after_fo),
        "loss_after_zo_step": float(loss_after_zo),
        "loss_decrease_fo": float(base_loss - loss_after_fo),
        "loss_decrease_zo": float(base_loss - loss_after_zo),
        "loss_components_after_fo_step": [float(x) for x in fo_components_after],
        "loss_components_after_zo_step": [float(x) for x in zo_components_after],
        "variance_seeds": int(variance_seeds),
        "q": int(q),
        "zo_eps": float(zo_eps),
        "step_size": float(step_size),
        "perturbation_mode": perturbation_mode,
        "estimator_name": estimator_name,
    }


def aggregate_records(records):
    if not records:
        return []

    grouped = defaultdict(list)
    for record in records:
        grouped[(record["task"], record["split"])].append(record)

    metric_keys = [
        "base_loss",
        "fo_grad_norm",
        "zo_grad_norm",
        "cosine_similarity",
        "norm_ratio",
        "variance_trace",
        "variance_mean_coordinate",
        "projected_grad_mean",
        "projected_grad_std",
        "loss_after_fo_step",
        "loss_after_zo_step",
        "loss_decrease_fo",
        "loss_decrease_zo",
    ]

    summaries = []
    for (task, split), split_records in sorted(grouped.items()):
        summary = {
            "task": int(task),
            "split": split,
            "objective_type": split_records[0].get("objective_type", "unknown"),
            "split_source": split_records[0].get("split_source", "unknown"),
            "num_batches": len(split_records),
            "known_classes": int(split_records[0].get("known_classes", 0)),
            "total_classes": int(split_records[0].get("total_classes", 0)),
        }
        for key in metric_keys:
            values = np.array([float(record[key]) for record in split_records], dtype=np.float64)
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_std"] = float(values.std())
        summaries.append(summary)
    return summaries


def save_records_as_json(output_path, payload):
    output_path = Path(output_path)
    ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_records_as_csv(output_path, rows):
    output_path = Path(output_path)
    rows = list(rows)
    if not rows:
        return
    ensure_parent_dir(output_path)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary_records(summary_records, output_dir, title_prefix="Surrogate Gradient Quality"):
    output_dir = Path(output_dir)
    if not summary_records:
        return []

    try:
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            import matplotlib.pyplot as plt
    except Exception as exc:
        warning_path = output_dir / "plot_warning.txt"
        ensure_parent_dir(warning_path)
        warning_path.write_text(
            f"Plot generation skipped because matplotlib could not be imported:\n{exc}\n",
            encoding="utf-8",
        )
        return [str(warning_path)]

    plt.style.use("seaborn-v0_8-whitegrid")
    split_groups = defaultdict(list)
    for record in summary_records:
        split_groups[record["split"]].append(record)

    colors = {
        "current_task_train": "#1f77b4",
        "current_task_validation": "#4c78a8",
        "old_task_replay": "#e45756",
        "old_task_validation": "#f58518",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    metric_specs = [
        ("cosine_similarity_mean", "Cosine Similarity", axes[0, 0]),
        ("norm_ratio_mean", "Norm Ratio", axes[0, 1]),
        ("variance_trace_mean", "Estimator Variance", axes[1, 0]),
    ]

    for split, records in split_groups.items():
        records = sorted(records, key=lambda item: item["task"])
        tasks = [record["task"] for record in records]
        color = colors.get(split, None)
        for metric_key, ylabel, axis in metric_specs:
            axis.plot(tasks, [record[metric_key] for record in records], marker="o", linewidth=2, label=split, color=color)
            axis.set_ylabel(ylabel)
            axis.set_xlabel("Task")

    loss_axis = axes[1, 1]
    for split, records in split_groups.items():
        records = sorted(records, key=lambda item: item["task"])
        tasks = [record["task"] for record in records]
        color = colors.get(split, None)
        loss_axis.plot(
            tasks,
            [record["loss_decrease_fo_mean"] for record in records],
            marker="o",
            linewidth=2,
            linestyle="-",
            label=f"{split} FO",
            color=color,
        )
        loss_axis.plot(
            tasks,
            [record["loss_decrease_zo_mean"] for record in records],
            marker="s",
            linewidth=2,
            linestyle="--",
            label=f"{split} ZO",
            color=color,
        )
    loss_axis.set_ylabel("True Loss Decrease")
    loss_axis.set_xlabel("Task")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles_loss, labels_loss = loss_axis.get_legend_handles_labels()
    fig.legend(handles + handles_loss, labels + labels_loss, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title_prefix, fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    pdf_path = output_dir / "surrogate_quality_summary.pdf"
    png_path = output_dir / "surrogate_quality_summary.png"
    ensure_parent_dir(pdf_path)
    fig.savefig(pdf_path, bbox_inches="tight")
    ensure_parent_dir(png_path)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return [str(pdf_path), str(png_path)]

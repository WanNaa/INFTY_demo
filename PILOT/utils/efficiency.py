import csv
import json
import time
from pathlib import Path

import numpy as np
import torch


def synchronize_if_needed(device):
    if isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


class EfficiencyTracker:
    def __init__(self, device, enabled=False):
        self.device = device
        self.enabled = bool(enabled)
        self.current = None

    def start_task(self, task_id, optimizer, epochs):
        if not self.enabled:
            return

        synchronize_if_needed(self.device)
        if isinstance(self.device, torch.device) and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        metadata = {}
        if hasattr(optimizer, "get_execution_path_metadata"):
            metadata = optimizer.get_execution_path_metadata()

        self.current = {
            "task": int(task_id),
            "epochs": int(epochs),
            "iteration_times_ms": [],
            "num_iterations": 0,
            "loss_trace": [],
            "task_start_time": time.perf_counter(),
            "optimizer_metadata": metadata,
            "current_epoch": None,
            "current_step": None,
            "iteration_start_time": None,
        }

    def start_iteration(self, epoch, step):
        if not self.enabled or self.current is None:
            return
        synchronize_if_needed(self.device)
        self.current["current_epoch"] = int(epoch)
        self.current["current_step"] = int(step)
        self.current["iteration_start_time"] = time.perf_counter()

    def end_iteration(self, loss_list):
        if not self.enabled or self.current is None:
            return

        synchronize_if_needed(self.device)
        iteration_end = time.perf_counter()
        iteration_start = self.current.get("iteration_start_time", iteration_end)
        iter_ms = (iteration_end - iteration_start) * 1000.0
        self.current["iteration_times_ms"].append(float(iter_ms))
        self.current["num_iterations"] += 1

        loss_value = 0.0
        for loss in loss_list:
            if torch.is_tensor(loss):
                loss_value += float(loss.detach().item())
            else:
                loss_value += float(loss)
        self.current["loss_trace"].append(float(loss_value))

    def end_task(self):
        if not self.enabled or self.current is None:
            return None

        synchronize_if_needed(self.device)
        task_time_s = time.perf_counter() - self.current["task_start_time"]
        iter_times = np.array(self.current["iteration_times_ms"], dtype=np.float64)
        loss_trace = np.array(self.current["loss_trace"], dtype=np.float64)

        peak_mb = None
        if isinstance(self.device, torch.device) and self.device.type == "cuda" and torch.cuda.is_available():
            peak_mb = float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))

        metadata = self.current.get("optimizer_metadata", {})
        record = {
            "task": int(self.current["task"]),
            "epochs": int(self.current["epochs"]),
            "num_iterations": int(self.current["num_iterations"]),
            "task_train_time_s": float(task_time_s),
            "mean_iteration_time_ms": float(iter_times.mean()) if iter_times.size else 0.0,
            "std_iteration_time_ms": float(iter_times.std()) if iter_times.size else 0.0,
            "median_iteration_time_ms": float(np.median(iter_times)) if iter_times.size else 0.0,
            "min_iteration_time_ms": float(iter_times.min()) if iter_times.size else 0.0,
            "max_iteration_time_ms": float(iter_times.max()) if iter_times.size else 0.0,
            "peak_gpu_memory_mb": peak_mb,
            "loss_mean": float(loss_trace.mean()) if loss_trace.size else 0.0,
            "loss_std": float(loss_trace.std()) if loss_trace.size else 0.0,
            "backward_graph_avoided": bool(metadata.get("backward_graph_avoided", False)),
            "graph_verification_mode": metadata.get("verification_mode", "unknown"),
            "execution_path": metadata.get("execution_path", "unknown"),
            "graph_context": metadata.get("graph_context", "unknown"),
            "iteration_times_ms": [float(v) for v in self.current["iteration_times_ms"]],
        }
        self.current = None
        return record


def build_efficiency_payload(args, task_records, metrics_summary):
    total_iterations = int(sum(record["num_iterations"] for record in task_records))
    total_task_time_s = float(sum(record["task_train_time_s"] for record in task_records))
    weighted_iter_time_ms = 0.0
    if total_iterations > 0:
        weighted_iter_time_ms = float(
            sum(record["mean_iteration_time_ms"] * record["num_iterations"] for record in task_records) / total_iterations
        )

    peak_values = [record["peak_gpu_memory_mb"] for record in task_records if record["peak_gpu_memory_mb"] is not None]
    overall_peak_memory_mb = float(max(peak_values)) if peak_values else None

    return {
        "model_name": args["model_name"],
        "method": args["inftyopt"],
        "seed": args["seed"],
        "dataset": args["dataset"],
        "backbone_type": args["backbone_type"],
        "task_records": task_records,
        "overall_summary": {
            "num_tasks": len(task_records),
            "total_iterations": total_iterations,
            "total_train_time_s": total_task_time_s,
            "mean_iteration_time_ms_weighted": weighted_iter_time_ms,
            "peak_gpu_memory_mb": overall_peak_memory_mb,
            "backward_graph_avoided": bool(task_records[0]["backward_graph_avoided"]) if task_records else False,
            "graph_verification_mode": task_records[0]["graph_verification_mode"] if task_records else "unknown",
            "execution_path": task_records[0]["execution_path"] if task_records else "unknown",
            "graph_context": task_records[0]["graph_context"] if task_records else "unknown",
        },
        "metrics_summary": metrics_summary,
    }


def save_json_payload(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_csv_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

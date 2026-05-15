#!/usr/bin/env python3
import argparse
import copy
import logging
import os
import sys
from pathlib import Path


def _preparse_gpu(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=str, default="0")
    known_args, _ = parser.parse_known_args(argv)
    return str(known_args.gpu)


os.environ.setdefault("CUDA_VISIBLE_DEVICES", _preparse_gpu(sys.argv[1:]))
os.environ.setdefault("MPLBACKEND", "Agg")


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from infty import plot as infty_plot
from main import (
    GEOMETRY_RESHAPING_OPTIMIZERS,
    GRADIENT_FILTERING_OPTIMIZERS,
    ZEROTH_ORDER_UPDATE_OPTIMIZERS,
)
from trainer import (
    _prepare_runtime_dirs,
    compute_forgetting,
    print_args,
    print_forgetting,
    save_efficiency_json,
    save_metrics_json,
    set_device,
    set_random,
    update_matrix_and_curve,
)
from utils import core as core_utils
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters, load_json, load_yaml


SUPPORTED_TRAJECTORY_NAMES = {
    "sgd",
    "adam",
    "adamw",
    "pcgrad",
    "cagrad",
    "unigrad",
    "zo_adam",
    "zo_adam_q4",
    "zo_adam_sign",
    "zo_adam_cons",
}

TRAJECTORY_ALIASES = {
    "forward_grad": "zo_adam",
    "unigrad_fs": "unigrad",
    "zo_sgd": "zo_adam",
    "zo_sgd_sign": "zo_adam_sign",
    "zo_sgd_conserve": "zo_adam_cons",
    "zo_adam_conserve": "zo_adam_cons",
}

ORIGINAL_GET_INFTY_OPTIMIZER = core_utils.get_infty_optimizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a PILOT experiment on one GPU and generate all available plot artifacts."
    )
    parser.add_argument("--gpu", type=str, default="0", help="GPU id to expose as CUDA device 0.")
    parser.add_argument("--inftyopt", type=str, default="ogd", help="select the optimizer")
    parser.add_argument("--config", type=str, default="./exps/ogd.json", help="experiment config file path")
    parser.add_argument("--workdir", type=str, default=str(REPO_ROOT / "workdirs"), help="root directory for runtime artifacts")
    parser.add_argument("--infty_config_dir", type=str, default=None, help="directory containing INFTY yaml configs")
    parser.add_argument("--ckp_dir", type=str, default=None, help="checkpoint directory")
    parser.add_argument("--log_dir", type=str, default=None, help="log directory")
    parser.add_argument("--output_dir", type=str, default=None, help="output directory")
    parser.add_argument("--plot_dir", type=str, default=None, help="plot directory")
    parser.add_argument("--all_tasks", action="store_true", help="generate task-specific plots after every task")
    parser.add_argument("--max_tasks", type=int, default=None, help="cap the number of training tasks to execute")
    parser.add_argument("--landscape_samples", type=int, default=7, help="grid size used by loss-landscape plotting")
    parser.add_argument("--landscape_limit", type=float, default=0.05, help="axis limit used by loss-landscape plotting")
    parser.add_argument("--landscape_eigen_max_iter", type=int, default=10, help="power-iteration steps for landscape Hessian")
    parser.add_argument("--landscape_eigen_tol", type=float, default=1e-3, help="eigen solve tolerance for landscape Hessian")
    parser.add_argument("--esd_trace_max_iter", type=int, default=10, help="Hutchinson iterations for Hessian trace")
    parser.add_argument("--esd_trace_tol", type=float, default=1e-3, help="trace tolerance for Hessian trace")
    parser.add_argument("--esd_density_iter", type=int, default=10, help="Lanczos iterations for ESD density")
    parser.add_argument("--esd_density_runs", type=int, default=1, help="number of stochastic Lanczos runs for ESD density")
    parser.add_argument("--trajectory_iter", type=int, default=200, help="toy trajectory iterations")
    parser.add_argument("--trajectory_lr", type=float, default=0.1, help="toy trajectory learning rate")
    parser.add_argument("--trajectory_grid_size", type=int, default=100, help="toy trajectory contour grid size")
    parser.add_argument("--skip_tsne", action="store_true", help="skip the final t-SNE/UMAP figure")
    return parser.parse_args()


def _resolve_user_path(path_value, base_dir):
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _resolve_runtime_dir(path_value, default_path, base_dir):
    if path_value is None:
        path = default_path
    else:
        path = _resolve_user_path(path_value, base_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_output_subdir(path_value, default_path, relative_root):
    if path_value is None:
        path = default_path
    else:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = relative_root / path
        path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _select_optimizer_config(optimizer_name, infty_config_dir):
    if optimizer_name == "base":
        return None
    if optimizer_name in GEOMETRY_RESHAPING_OPTIMIZERS:
        config_path = infty_config_dir / "geometry_reshaping" / f"{optimizer_name}.yaml"
    elif optimizer_name in GRADIENT_FILTERING_OPTIMIZERS:
        config_path = infty_config_dir / "gradient_filtering" / f"{optimizer_name}.yaml"
    elif optimizer_name in ZEROTH_ORDER_UPDATE_OPTIMIZERS:
        config_path = infty_config_dir / "zeroth_order_updates" / "zeroflow.yaml"
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")

    if not config_path.is_file():
        raise FileNotFoundError(f"Missing optimizer config: {config_path}")
    return config_path


def _tracking_get_infty_optimizer(params, base_optimizer, model, args):
    optimizer = ORIGINAL_GET_INFTY_OPTIMIZER(params=params, base_optimizer=base_optimizer, model=model, args=args)
    setattr(model, "_infty_last_optimizer", optimizer)
    if hasattr(model, "module"):
        setattr(model.module, "_infty_last_optimizer", optimizer)
    return optimizer


def _install_optimizer_tracking():
    core_utils.get_infty_optimizer = _tracking_get_infty_optimizer


def _build_runtime_args(cli_args):
    optimizer_name = cli_args.inftyopt.lower()

    config_json_path = _resolve_user_path(cli_args.config, SCRIPT_DIR)
    workdir = _resolve_user_path(cli_args.workdir, SCRIPT_DIR)
    infty_config_dir = _resolve_runtime_dir(cli_args.infty_config_dir, workdir / "configs" / "infty", SCRIPT_DIR)
    ckp_dir = _resolve_runtime_dir(cli_args.ckp_dir, workdir / "checkpoints", SCRIPT_DIR)
    log_dir = _resolve_runtime_dir(cli_args.log_dir, workdir / "logs", SCRIPT_DIR)
    output_dir = _resolve_runtime_dir(cli_args.output_dir, workdir / "outputs", SCRIPT_DIR)
    plot_dir = _resolve_runtime_dir(cli_args.plot_dir, workdir / "plots", SCRIPT_DIR)

    optimizer_config_path = _select_optimizer_config(optimizer_name, infty_config_dir)
    if optimizer_config_path is not None:
        print(f"Loading optimizer config: {optimizer_config_path}")

    args = load_json(str(config_json_path))
    if optimizer_config_path is not None:
        optimizer_args = load_yaml(str(optimizer_config_path))
        args.update(optimizer_args)

    args["inftyopt"] = optimizer_name
    args["device"] = [str(cli_args.gpu)]
    args["workdir"] = str(workdir)
    args["infty_config_dir"] = str(infty_config_dir)
    args["ckp_dir"] = str(ckp_dir)
    args["log_dir"] = str(log_dir)
    args["output_dir"] = str(output_dir)
    args["plot_dir"] = str(plot_dir)
    args["metrics_json_dir"] = str(
        _resolve_output_subdir(args.get("metrics_json_dir"), output_dir / "metrics_json", output_dir)
    )
    args["efficiency_json_dir"] = str(
        _resolve_output_subdir(args.get("efficiency_json_dir"), output_dir / "efficiency_json", output_dir)
    )
    args["conflict_stats_dir"] = str(
        _resolve_output_subdir(args.get("conflict_stats_dir"), output_dir / "conflict_stats", output_dir)
    )
    args["sharpness_json_dir"] = str(
        _resolve_output_subdir(args.get("sharpness_json_dir"), output_dir / "sharpness_json", output_dir)
    )
    return args


def _plot_targets(learner, targets):
    model_name = learner.args["model_name"].lower()
    if model_name == "ease":
        return torch.where(
            targets - learner._known_classes >= 0,
            targets - learner._known_classes,
            -1,
        )
    return targets


def _make_plot_loss_factory(learner):
    def create_loss_fn(inputs, targets, model=None):
        plot_targets = _plot_targets(learner, targets)
        return learner.create_loss_fn(inputs, plot_targets, model=model)

    return create_loss_fn


def _supports_hessian_plots(learner):
    return hasattr(learner, "create_loss_fn") and callable(learner.create_loss_fn)


def _extract_last_optimizer(learner):
    network = getattr(learner, "_network", None)
    if network is None:
        return None
    optimizer = getattr(network, "_infty_last_optimizer", None)
    if optimizer is not None:
        return optimizer
    if hasattr(network, "module"):
        return getattr(network.module, "_infty_last_optimizer", None)
    return None


def _resolve_trajectory_name(args):
    inftyopt = str(args["inftyopt"]).lower()
    if inftyopt in SUPPORTED_TRAJECTORY_NAMES:
        return inftyopt, None
    if inftyopt in TRAJECTORY_ALIASES:
        alias = TRAJECTORY_ALIASES[inftyopt]
        return alias, f"trajectory visualizer uses closest available alias '{alias}' for optimizer '{inftyopt}'"
    base_optimizer = str(args.get("optimizer", "")).lower()
    if base_optimizer in SUPPORTED_TRAJECTORY_NAMES:
        return base_optimizer, f"trajectory visualizer falls back to base optimizer '{base_optimizer}' for optimizer '{inftyopt}'"
    return None, f"trajectory visualizer has no supported solver for optimizer '{inftyopt}'"


def _run_landscape_plot(learner, args, cli_args, task_id, plot_root):
    if not _supports_hessian_plots(learner):
        logging.info("[Plot] Skip loss landscape: model '%s' has no create_loss_fn.", learner.args["model_name"])
        return None

    result = infty_plot.visualize_loss_landscape(
        optimizer=None,
        model=learner._network,
        create_loss_fn=_make_plot_loss_factory(learner),
        loader=learner.train_loader,
        task=task_id,
        device=learner._device,
        limit=cli_args.landscape_limit,
        samples=cli_args.landscape_samples,
        eigen_max_iter=cli_args.landscape_eigen_max_iter,
        eigen_tol=cli_args.landscape_eigen_tol,
        output_dir=plot_root / "diagnostics" / "landscape",
        source_name=args["inftyopt"],
    )
    logging.info("[Plot] Loss landscape saved to %s", result["plot_path"])
    return result


def _run_esd_plot(learner, args, cli_args, task_id, plot_root):
    if not _supports_hessian_plots(learner):
        logging.info("[Plot] Skip ESD: model '%s' has no create_loss_fn.", learner.args["model_name"])
        return None

    result = infty_plot.visualize_esd(
        optimizer=None,
        model=learner._network,
        create_loss_fn=_make_plot_loss_factory(learner),
        loader=learner.train_loader,
        task=task_id,
        device=learner._device,
        output_dir=plot_root / "diagnostics" / "esd",
        source_name=args["inftyopt"],
        trace_max_iter=cli_args.esd_trace_max_iter,
        trace_tol=cli_args.esd_trace_tol,
        density_iter=cli_args.esd_density_iter,
        density_runs=cli_args.esd_density_runs,
    )
    logging.info("[Plot] ESD saved to %s", result["plot_path"])
    return result


def _run_conflict_plot(learner, args, task_id, plot_root):
    optimizer = _extract_last_optimizer(learner)
    if optimizer is None:
        logging.info("[Plot] Skip conflicts: no optimizer instance was captured.")
        return None

    try:
        result = infty_plot.visualize_conflicts(
            optimizer=optimizer,
            task=task_id,
            output_dir=plot_root / "diagnostics" / "conflicts",
            source_name=args["inftyopt"],
        )
    except AttributeError as exc:
        logging.info("[Plot] Skip conflicts: %s", exc)
        return None

    if "plot_path" in result:
        logging.info("[Plot] Conflict curve saved to %s", result["plot_path"])
    elif "warning_path" in result:
        logging.info("[Plot] Conflict plot warning saved to %s", result["warning_path"])
    else:
        logging.info("[Plot] Conflict artifacts saved for task %s.", task_id)
    return result


def _run_trajectory_plot(args, cli_args, plot_root):
    trajectory_name, note = _resolve_trajectory_name(args)
    if note:
        logging.info("[Plot] %s.", note)
    if trajectory_name is None:
        return None

    infty_plot.visualize_trajectory(
        trajectory_name,
        n_iter=cli_args.trajectory_iter,
        lr=cli_args.trajectory_lr,
        output_dir=plot_root / "diagnostics" / "trajectory",
        grid_size=cli_args.trajectory_grid_size,
    )
    figure_path = plot_root / "diagnostics" / "trajectory" / f"traj_{trajectory_name}.pdf"
    logging.info("[Plot] Trajectory saved to %s", figure_path)
    return {"plot_path": str(figure_path), "optimizer_name": trajectory_name}


def _run_tsne_plot(learner, plot_root):
    tsne_dir = plot_root / "diagnostics" / "tsne"
    tsne_dir.mkdir(parents=True, exist_ok=True)
    figure_name = f"{learner.args['model_name']}{learner._total_classes}tsne.pdf"
    current_dir = Path.cwd()
    try:
        os.chdir(tsne_dir)
        learner.tsne()
    finally:
        os.chdir(current_dir)
    figure_path = tsne_dir / figure_name
    logging.info("[Plot] t-SNE/UMAP figure saved to %s", figure_path)
    return {"plot_path": str(figure_path)}


def _run_task_plots(learner, args, cli_args, task_id):
    plot_root = Path(args["plot_dir"]).expanduser().resolve()
    plot_summary = {}

    try:
        plot_summary["landscape"] = _run_landscape_plot(learner, args, cli_args, task_id, plot_root)
    except Exception as exc:
        logging.exception("[Plot] Loss landscape failed on task %s: %s", task_id, exc)

    try:
        plot_summary["esd"] = _run_esd_plot(learner, args, cli_args, task_id, plot_root)
    except Exception as exc:
        logging.exception("[Plot] ESD failed on task %s: %s", task_id, exc)

    try:
        plot_summary["conflicts"] = _run_conflict_plot(learner, args, task_id, plot_root)
    except Exception as exc:
        logging.exception("[Plot] Conflict plot failed on task %s: %s", task_id, exc)

    return plot_summary


def _run_single_seed(cli_args, args):
    args = _prepare_runtime_dirs(copy.deepcopy(args))
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    log_root = Path(args.get("log_dir", str(REPO_ROOT / "workdirs" / "logs"))).expanduser().resolve()
    logs_dir = log_root / (
        f"{args['model_name']}-{args['backbone_type']}-{args['dataset']}-"
        f"{init_cls}-{args['increment']}-{args['seed']}"
    )
    logs_dir.mkdir(parents=True, exist_ok=True)
    logfilename = logs_dir / f"{args['inftyopt']}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=str(logfilename)),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    set_random(args["seed"])
    set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    args["nb_classes"] = data_manager.nb_classes
    args["nb_tasks"] = data_manager.nb_tasks
    learner = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []
    tasks_to_run = data_manager.nb_tasks
    if cli_args.max_tasks is not None:
        tasks_to_run = min(tasks_to_run, max(int(cli_args.max_tasks), 1))
    last_task_index = tasks_to_run - 1
    last_plot_summary = {}

    for task in range(tasks_to_run):
        logging.info("All params: %s", count_parameters(learner._network))
        logging.info("Trainable params: %s", count_parameters(learner._network, True))
        learner.incremental_train(data_manager)

        if cli_args.all_tasks or task == last_task_index:
            last_plot_summary = _run_task_plots(learner, args, cli_args, task)

        cnn_accy, nme_accy = learner.eval_task()
        learner.after_task()
        update_matrix_and_curve(cnn_accy, nme_accy, cnn_matrix, nme_matrix, cnn_curve, nme_curve)

    cnn_forgetting = compute_forgetting(cnn_matrix, last_task_index)
    if args.get("print_forget", True):
        print_forgetting(cnn_matrix, nme_matrix, last_task_index)
    if args.get("save_metrics_json", False):
        save_metrics_json(args, cnn_curve, cnn_matrix, cnn_forgetting)
    if args.get("save_efficiency_json", False):
        save_efficiency_json(args, learner, cnn_curve, cnn_forgetting)

    plot_root = Path(args["plot_dir"]).expanduser().resolve()
    try:
        last_plot_summary["trajectory"] = _run_trajectory_plot(args, cli_args, plot_root)
    except Exception as exc:
        logging.exception("[Plot] Trajectory plot failed: %s", exc)

    if not cli_args.skip_tsne:
        try:
            last_plot_summary["tsne"] = _run_tsne_plot(learner, plot_root)
        except Exception as exc:
            logging.exception("[Plot] t-SNE/UMAP plot failed: %s", exc)

    return last_plot_summary


def main():
    cli_args = parse_args()
    runtime_args = _build_runtime_args(cli_args)
    _install_optimizer_tracking()

    seeds = copy.deepcopy(runtime_args["seed"])
    if not isinstance(seeds, (list, tuple)):
        seeds = [seeds]
    device_spec = copy.deepcopy(runtime_args["device"])

    plot_summaries = []
    for seed in seeds:
        seed_args = copy.deepcopy(runtime_args)
        seed_args["seed"] = seed
        seed_args["device"] = copy.deepcopy(device_spec)
        plot_summaries.append(_run_single_seed(cli_args, seed_args))

    print("=" * 80)
    print("Plot run finished.")
    print(f"GPU binding: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Artifacts root: {runtime_args['plot_dir']}")
    print(f"Seeds: {seeds}")
    print("=" * 80)
    for summary in plot_summaries:
        for plot_name, plot_result in summary.items():
            if not plot_result:
                continue
            if isinstance(plot_result, dict) and "plot_path" in plot_result:
                print(f"[{plot_name}] {plot_result['plot_path']}")


if __name__ == "__main__":
    main()

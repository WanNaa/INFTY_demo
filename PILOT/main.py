import argparse
from pathlib import Path

from trainer import train
from utils.toolkit import load_json, load_yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

GEOMETRY_RESHAPING_OPTIMIZERS = {
    "sam",
    "gsam",
    "looksam",
    "gam",
    "c_flat",
    "c_flat_plus",
}
ZEROTH_ORDER_UPDATE_OPTIMIZERS = {
    "zo_sgd",
    "zo_sgd_sign",
    "zo_sgd_conserve",
    "zo_adam",
    "zo_adam_sign",
    "zo_adam_conserve",
    "forward_grad",
}
GRADIENT_FILTERING_OPTIMIZERS = {
    "pcgrad",
    "gradvac",
    "cagrad",
    "unigrad_fs",
    "ogd",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inftyopt", type=str, default="ogd", help="select the optimizer")
    parser.add_argument("--config", type=str, default="./exps/ogd.json", help="experiment config file path")
    parser.add_argument("--workdir", type=str, default=str(REPO_ROOT / "workdirs"), help="root directory for runtime artifacts")
    parser.add_argument("--infty_config_dir", type=str, default=None, help="directory containing INFTY yaml configs")
    parser.add_argument("--ckp_dir", type=str, default=None, help="checkpoint directory")
    parser.add_argument("--log_dir", type=str, default=None, help="log directory")
    parser.add_argument("--output_dir", type=str, default=None, help="output directory")
    parser.add_argument("--plot_dir", type=str, default=None, help="plot directory")
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


def main():
    cli_args = parse_args()
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

    train(args)


if __name__ == "__main__":
    main()

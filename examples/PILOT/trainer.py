import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from utils import factory
from utils.data_manager import DataManager
from utils.efficiency import build_efficiency_payload, save_csv_rows, save_json_payload
from utils.toolkit import count_parameters


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def _resolve_path(path_value, base_dir):
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _prepare_runtime_dirs(args):
    workdir = _resolve_path(args.get("workdir", str(REPO_ROOT / "workdirs")), SCRIPT_DIR)
    args["workdir"] = str(workdir)

    primary_defaults = {
        "ckp_dir": workdir / "checkpoints",
        "log_dir": workdir / "logs",
        "output_dir": workdir / "outputs",
        "plot_dir": workdir / "plots",
    }
    for key, default_path in primary_defaults.items():
        raw_value = args.get(key)
        path = _resolve_path(raw_value, workdir) if raw_value is not None else default_path
        path.mkdir(parents=True, exist_ok=True)
        args[key] = str(path)

    output_dir = Path(args["output_dir"])
    derived_defaults = {
        "metrics_json_dir": output_dir / "metrics_json",
        "efficiency_json_dir": output_dir / "efficiency_json",
        "conflict_stats_dir": output_dir / "conflict_stats",
        "sharpness_json_dir": output_dir / "sharpness_json",
    }
    for key, default_path in derived_defaults.items():
        raw_value = args.get(key)
        path = _resolve_path(raw_value, output_dir) if raw_value is not None else default_path
        path.mkdir(parents=True, exist_ok=True)
        args[key] = str(path)

    return args


def train(args):
    args = _prepare_runtime_dirs(copy.deepcopy(args))
    seeds = copy.deepcopy(args["seed"])
    if not isinstance(seeds, (list, tuple)):
        seeds = [seeds]
    device = copy.deepcopy(args["device"])
    for seed in seeds:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
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
        args["dataset"], args["shuffle"], args["seed"],
        args["init_cls"], args["increment"], args,
    )
    args["nb_classes"] = data_manager.nb_classes
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []
    for task in range(data_manager.nb_tasks):
        logging.info(f"All params: {count_parameters(model._network)}")
        logging.info(f"Trainable params: {count_parameters(model._network, True)}")
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()
        update_matrix_and_curve(cnn_accy, nme_accy, cnn_matrix, nme_matrix, cnn_curve, nme_curve)
    cnn_forgetting = compute_forgetting(cnn_matrix, task)
    if args.get("print_forget", True):
        print_forgetting(cnn_matrix, nme_matrix, task)
    if args.get("save_metrics_json", False):
        save_metrics_json(args, cnn_curve, cnn_matrix, cnn_forgetting)
    if args.get("save_efficiency_json", False):
        save_efficiency_json(args, model, cnn_curve, cnn_forgetting)


def update_matrix_and_curve(cnn_accy, nme_accy, cnn_matrix, nme_matrix, cnn_curve, nme_curve):
    logging.info(f"CNN: {cnn_accy['grouped']}")
    cnn_keys = [k for k in cnn_accy["grouped"] if "-" in k]
    cnn_matrix.append([cnn_accy["grouped"][k] for k in cnn_keys])
    cnn_curve["top1"].append(cnn_accy["top1"])
    cnn_curve["top5"].append(cnn_accy["top5"])
    logging.info(f"CNN top1 curve: {cnn_curve['top1']}")
    logging.info(f"CNN top5 curve: {cnn_curve['top5']}")
    print("Average Accuracy (CNN):", sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))
    logging.info(f"Average Accuracy (CNN): {sum(cnn_curve['top1']) / len(cnn_curve['top1'])}")
    # if nme_accy is not None:
    #     logging.info(f"NME: {nme_accy['grouped']}")
    #     nme_keys = [k for k in nme_accy["grouped"] if '-' in k]
    #     nme_matrix.append([nme_accy["grouped"][k] for k in nme_keys])
    #     nme_curve["top1"].append(nme_accy["top1"])
    #     nme_curve["top5"].append(nme_accy["top5"])
    #     logging.info(f"NME top1 curve: {nme_curve['top1']}")
    #     logging.info(f"NME top5 curve: {nme_curve['top5']}")
    #     print('Average Accuracy (NME):', sum(nme_curve["top1"]) / len(nme_curve["top1"]))
    #     logging.info(f"Average Accuracy (NME): {sum(nme_curve['top1']) / len(nme_curve['top1'])}")


def print_forgetting(cnn_matrix, nme_matrix, task):
    if cnn_matrix:
        np_acctable = np.zeros([task + 1, task + 1])
        for idx, line in enumerate(cnn_matrix):
            np_acctable[idx, :len(line)] = np.array(line)
        np_acctable = np_acctable.T
        forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        print("Accuracy Matrix (CNN):")
        print(np_acctable)
        logging.info(f"Forgetting (CNN): {forgetting}")
    # if nme_matrix:
    #     np_acctable = np.zeros([task + 1, task + 1])
    #     for idx, line in enumerate(nme_matrix):
    #         np_acctable[idx, :len(line)] = np.array(line)
    #     np_acctable = np_acctable.T
    #     forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
    #     print('Accuracy Matrix (NME):')
    #     print(np_acctable)
    #     logging.info(f'Forgetting (NME): {forgetting}')


def compute_forgetting(cnn_matrix, task):
    if not cnn_matrix or task <= 0:
        return 0.0
    np_acctable = np.zeros([task + 1, task + 1])
    for idx, line in enumerate(cnn_matrix):
        np_acctable[idx, :len(line)] = np.array(line)
    np_acctable = np_acctable.T
    return float(np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task]))


def save_metrics_json(args, cnn_curve, cnn_matrix, cnn_forgetting):
    output_dir = Path(args.get("metrics_json_dir", "./metrics_json")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    weight_name = "alpha_aux" if "alpha_aux" in args else "alpha_kd" if "alpha_kd" in args else "alpha"
    weight_value = float(args.get(weight_name, 1.0))
    weight_tag = f"{weight_name.replace('_', '')}{str(weight_value).replace('-', 'm').replace('.', 'p')}"

    payload = {
        "model_name": args["model_name"],
        "method": args["inftyopt"],
        "seed": args["seed"],
        "weight_name": weight_name,
        "weight_value": weight_value,
        "cnn_curve_top1": [float(x) for x in cnn_curve["top1"]],
        "cnn_curve_top5": [float(x) for x in cnn_curve["top5"]],
        "cnn_matrix": [[float(v) for v in row] for row in cnn_matrix],
        "last_accuracy": float(cnn_curve["top1"][-1]) if cnn_curve["top1"] else 0.0,
        "mean_accuracy": float(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])) if cnn_curve["top1"] else 0.0,
        "forgetting": float(cnn_forgetting),
    }

    if args.get("feedback_stress_enabled", False):
        payload.update(
            {
                "feedback_stress_enabled": True,
                "feedback_stress_mode": args.get("feedback_stress_mode", "none"),
                "feedback_stress_level": float(args.get("feedback_stress_level", 0.0)),
                "feedback_sparse_mode": args.get("feedback_sparse_mode", "steps"),
                "feedback_keep_prob": float(args.get("feedback_keep_prob", 1.0)),
                "feedback_label_keep_prob": float(args.get("feedback_label_keep_prob", 1.0)),
                "feedback_noise_type": args.get("feedback_noise_type", "none"),
                "feedback_noise_level": float(args.get("feedback_noise_level", 0.0)),
                "feedback_noise_application": args.get("feedback_noise_application", "multiplicative"),
            }
        )

    output_path = output_dir / build_metrics_filename(args, weight_tag)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    logging.info(f"[Metrics] Saved summary to {output_path}")


def save_efficiency_json(args, model, cnn_curve, cnn_forgetting):
    output_dir = Path(args.get("efficiency_json_dir", "./efficiency_json")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    task_records = model.get_efficiency_task_records() if hasattr(model, "get_efficiency_task_records") else []
    metrics_summary = {
        "last_accuracy": float(cnn_curve["top1"][-1]) if cnn_curve["top1"] else 0.0,
        "mean_accuracy": float(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])) if cnn_curve["top1"] else 0.0,
        "forgetting": float(cnn_forgetting),
    }
    payload = build_efficiency_payload(args, task_records, metrics_summary)

    base_name = f"{args['model_name']}_{args['inftyopt']}_seed{args['seed']}"
    json_path = output_dir / f"{base_name}.json"
    csv_path = output_dir / f"{base_name}.csv"
    save_json_payload(str(json_path), payload)
    csv_rows = []
    for record in task_records:
        csv_record = dict(record)
        csv_record.pop("iteration_times_ms", None)
        csv_rows.append(csv_record)
    save_csv_rows(str(csv_path), csv_rows)
    logging.info(f"[Efficiency] Saved JSON to {json_path}")
    logging.info(f"[Efficiency] Saved CSV to {csv_path}")


def build_metrics_filename(args, weight_tag):
    suffix = ""
    if args.get("feedback_stress_enabled", False):
        mode = args.get("feedback_stress_mode", "none")
        if mode == "sparse_feedback":
            variant = args.get("feedback_sparse_mode", "steps")
        elif mode == "noisy_feedback":
            variant = args.get("feedback_noise_type", "gaussian")
        elif mode == "reduced_supervision":
            variant = "labels"
        else:
            variant = "none"
        level_tag = str(float(args.get("feedback_stress_level", 0.0))).replace("-", "m").replace(".", "p")
        suffix = f"_{mode}_{variant}_lvl{level_tag}"
    return f"{args['model_name']}_{args['inftyopt']}_seed{args['seed']}_{weight_tag}{suffix}.json"


def set_device(args):
    gpus = [torch.device("cpu") if d == -1 else torch.device(f"cuda:{d}") for d in args["device"]]
    args["device"] = gpus


def set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))

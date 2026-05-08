# INFTY 源码重构计划书（Codex 执行版）

> 用途：把本文直接交给 Codex，让其在 INFTY 仓库根目录中按步骤执行重构、修复、测试与文档更新。
>
> 执行假设：当前工作目录是 INFTY 仓库根目录；仓库包含 `src/infty/optim`、`src/infty/plot`、`examples/PILOT`、`examples/infty_configs`、`examples/run_scripts`。
>
> 重要原则：不要一次性大范围重写所有算法。先完成目录迁移和路径参数化，再做确定性 bug 修复，最后补算法论文一致性审查与测试。

---

## 0. Codex 总任务

请在 INFTY 仓库中完成一次结构化重构，目标如下：

1. 检查 `src/infty/optim/` 下每个优化方法实现是否与对应原论文算法流程一致，记录审查结论并修复确定性 bug。
2. 检查 `src/infty/plot/` 下中间可视化代码的正确性、合理性和输出路径规范，修复确定性 bug。
3. 对比 PILOT 官方源码结构，在 `examples/` 下新建 `workdirs/`；将不属于 PILOT 官方结构的文件、文件夹和运行产物按类别移动到 `workdirs/`。
4. 重写 `run_scripts`，要求脚本可复现、可 dry-run、路径无关、日志和产物统一写入 `workdirs/`。
5. 补充测试和文档，确保后续维护者可以复现迁移逻辑和算法审查结论。

请按本文的 P0 到 P6 顺序执行。每个阶段完成后，运行对应验收命令；如果某条命令因环境缺依赖失败，请把失败原因写入 `docs/migration_report.md`，不要静默跳过。

---

## 1. 外部结构基线

### 1.1 INFTY 当前公开结构基线

公开 INFTY 仓库顶层包含：

```text
examples/
img/
src/infty/
.gitignore
LICENSE
README.md
pyproject.toml
setup.py
```

当前公开 `examples/` 下包含：

```text
examples/PILOT/
examples/infty_configs/
examples/run_scripts/
```

当前公开 `src/infty/optim/` 下包含：

```text
src/infty/optim/geometry_reshaping/
src/infty/optim/zeroth_order_updates/
src/infty/optim/gradient_filtering/
src/infty/optim/__init__.py
```

当前公开 `src/infty/plot/` 下包含：

```text
src/infty/plot/__init__.py
src/infty/plot/visualize_conflicts.py
src/infty/plot/visualize_esd.py
src/infty/plot/visualize_loss_landscape.py
src/infty/plot/visualize_trajectory.py
```

参考来源：

- https://github.com/THUDM/INFTY
- https://github.com/THUDM/INFTY/tree/main/examples
- https://github.com/THUDM/INFTY/tree/main/src/infty/optim
- https://github.com/THUDM/INFTY/tree/main/src/infty/plot

### 1.2 PILOT 官方结构基线

公开 LAMDA-PILOT 官方顶层包含：

```text
backbone/
exps/
models/
resources/
utils/
.gitignore
LICENSE
README.md
main.py
trainer.py
```

参考来源：

- https://github.com/LAMDA-CL/LAMDA-PILOT

### 1.3 本次重构的目录判定规则

`examples/PILOT/` 应尽量只保留 PILOT 官方结构和 INFTY 必须注入的最小改动。

以下内容不属于 PILOT 官方源码结构，应放入 `workdirs/`：

```text
INFTY 专用配置
运行脚本
日志
checkpoint
plot 输出
实验输出
缓存
临时文件
wandb 输出
本地 notebook checkpoint
```

---

## 2. 目标目录结构

重构后的目标结构：

```text
INFTY/
├── src/
│   └── infty/
│       ├── optim/
│       │   ├── geometry_reshaping/
│       │   │   ├── base.py
│       │   │   ├── sam.py
│       │   │   ├── gsam.py
│       │   │   ├── looksam.py
│       │   │   ├── gam.py
│       │   │   └── c_flat.py
│       │   ├── zeroth_order_updates/
│       │   │   └── zeroflow.py
│       │   ├── gradient_filtering/
│       │   │   ├── base.py
│       │   │   ├── pcgrad.py
│       │   │   ├── cagrad.py
│       │   │   ├── gradvac.py
│       │   │   ├── unigrad_fs.py
│       │   │   └── ogd.py
│       │   └── __init__.py
│       ├── plot/
│       │   ├── __init__.py
│       │   ├── landscape.py
│       │   ├── esd.py
│       │   ├── conflicts.py
│       │   └── trajectory.py
│       └── utils/
├── examples/
│   ├── PILOT/
│   │   ├── backbone/
│   │   ├── exps/
│   │   ├── models/
│   │   ├── resources/
│   │   ├── utils/
│   │   ├── .gitignore
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── main.py
│   │   └── trainer.py
│   └── workdirs/
│       ├── configs/
│       │   └── infty/
│       │       ├── geometry_reshaping/
│       │       ├── zeroth_order_updates/
│       │       └── gradient_filtering/
│       ├── run_scripts/
│       │   ├── _common.sh
│       │   ├── run_one.sh
│       │   ├── run_geometry_reshaping.sh
│       │   ├── run_zeroth_order_updates.sh
│       │   ├── run_gradient_filtering.sh
│       │   └── run_plots.sh
│       ├── logs/
│       ├── checkpoints/
│       ├── plots/
│       ├── outputs/
│       ├── cache/
│       └── tmp/
├── tests/
│   ├── optim/
│   ├── plot/
│   └── examples/
└── docs/
    ├── optimizer_audit.md
    ├── plot_audit.md
    └── migration_report.md
```

注意：如果本地仓库已有不同文件名，不要机械删除。先分类移动，再在 `docs/migration_report.md` 记录差异。

---

## 3. P0：建立基线

### 3.1 创建分支并记录当前状态

在仓库根目录执行：

```bash
git status --short
git checkout -b refactor/infty-workdirs-and-scripts
mkdir -p docs
```

生成当前结构快照：

```bash
{
  echo "# Migration Baseline"
  echo
  echo "## Git status"
  git status --short
  echo
  echo "## Root tree"
  find . -maxdepth 3 -type d | sort
  echo
  echo "## examples tree"
  find examples -maxdepth 4 -type d -o -type f | sort
  echo
  echo "## src/infty tree"
  find src/infty -maxdepth 4 -type d -o -type f | sort
} > docs/migration_report.md
```

### 3.2 基线验收

```bash
test -f docs/migration_report.md
```

---

## 4. P1：迁移 `workdirs`

### 4.1 新建 workdirs

```bash
mkdir -p workdirs/{configs,run_scripts,logs,checkpoints,plots,outputs,cache,tmp}
mkdir -p workdirs/configs/infty
```

### 4.2 移动 INFTY 配置和旧脚本

优先使用 `git mv`，没有纳入 Git 的文件用 `mv`。

```bash
if [ -d examples/infty_configs ]; then
  git mv examples/infty_configs workdirs/configs/infty || mv examples/infty_configs workdirs/configs/infty
fi

if [ -d examples/run_scripts ]; then
  mkdir -p workdirs/run_scripts/legacy
  shopt -s dotglob nullglob
  for item in examples/run_scripts/*; do
    git mv "$item" workdirs/run_scripts/legacy/ 2>/dev/null || mv "$item" workdirs/run_scripts/legacy/
  done
  rmdir examples/run_scripts 2>/dev/null || true
fi
```

如果上面的 `examples/infty_configs` 被整体移动到了 `workdirs/configs/infty/infty_configs`，请再执行一次规范化：

```bash
if [ -d workdirs/configs/infty/infty_configs ]; then
  shopt -s dotglob nullglob
  for item in workdirs/configs/infty/infty_configs/*; do
    git mv "$item" workdirs/configs/infty/ 2>/dev/null || mv "$item" workdirs/configs/infty/
  done
  rmdir workdirs/configs/infty/infty_configs 2>/dev/null || true
fi
```

最终应得到：

```text
workdirs/configs/infty/geometry_reshaping/
workdirs/configs/infty/zeroth_order_updates/
workdirs/configs/infty/gradient_filtering/
```

### 4.3 迁移 PILOT 下的运行产物

```bash
mkdir -p workdirs/{checkpoints,logs,plots,outputs,cache,tmp}

move_dir_if_exists() {
  src="$1"
  dst="$2"
  if [ -d "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    git mv "$src" "$dst" 2>/dev/null || mv "$src" "$dst"
  fi
}

move_dir_if_exists examples/PILOT/ckp workdirs/checkpoints/pilot_ckp
move_dir_if_exists examples/PILOT/checkpoints workdirs/checkpoints/pilot_checkpoints
move_dir_if_exists examples/PILOT/logs workdirs/logs/pilot_logs
move_dir_if_exists examples/PILOT/plots workdirs/plots/pilot_plots
move_dir_if_exists examples/PILOT/outputs workdirs/outputs/pilot_outputs
move_dir_if_exists examples/PILOT/results workdirs/outputs/pilot_results
move_dir_if_exists examples/PILOT/wandb workdirs/logs/wandb
```

### 4.4 清理缓存类文件

不要提交缓存。执行：

```bash
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
find . -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +
```

### 4.5 更新 `.gitignore`

在仓库根 `.gitignore` 中确保包含：

```gitignore
# Runtime artifacts
workdirs/logs/
workdirs/checkpoints/
workdirs/plots/
workdirs/outputs/
workdirs/cache/
workdirs/tmp/
workdirs/**/wandb/

# Python/cache artifacts
__pycache__/
*.py[cod]
.pytest_cache/
.ipynb_checkpoints/

# Model artifacts
*.pt
*.pth
*.ckpt
*.safetensors
```

但不要忽略：

```text
workdirs/configs/infty/
workdirs/run_scripts/
```

因为配置和脚本需要纳入版本控制。

### 4.6 P1 验收

```bash
test -d workdirs/configs/infty
test -d workdirs/run_scripts
find examples/PILOT -maxdepth 1 -type d | sort
```

`examples/PILOT` 顶层不应再有 `ckp`、`logs`、`plots`、`outputs`、`results`、`wandb`。

---

## 5. P2：参数化 PILOT 入口路径

目标：所有 INFTY 配置、日志、checkpoint、plot、output 都通过 `--workdir` 或派生路径控制，不能再硬编码写入 `examples/PILOT`。

### 5.1 修改 `examples/PILOT/main.py`

在 argparse 中新增参数：

```python
parser.add_argument("--workdir", type=str, default="../../workdirs")
parser.add_argument("--infty_config_dir", type=str, default=None)
parser.add_argument("--log_dir", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--plot_dir", type=str, default=None)
```

如果已有 `--ckp_dir`，保留；如果没有，新增：

```python
parser.add_argument("--ckp_dir", type=str, default=None)
```

解析后统一路径：

```python
from pathlib import Path

workdir = Path(args.workdir).expanduser().resolve()
args.workdir = str(workdir)
args.infty_config_dir = str(Path(args.infty_config_dir).expanduser().resolve()) if args.infty_config_dir else str(workdir / "configs" / "infty")
args.ckp_dir = str(Path(args.ckp_dir).expanduser().resolve()) if args.ckp_dir else str(workdir / "checkpoints")
args.log_dir = str(Path(args.log_dir).expanduser().resolve()) if args.log_dir else str(workdir / "logs")
args.output_dir = str(Path(args.output_dir).expanduser().resolve()) if args.output_dir else str(workdir / "outputs")
args.plot_dir = str(Path(args.plot_dir).expanduser().resolve()) if args.plot_dir else str(workdir / "plots")

for path in [args.infty_config_dir, args.ckp_dir, args.log_dir, args.output_dir, args.plot_dir]:
    Path(path).mkdir(parents=True, exist_ok=True)
```

然后把原先类似下面的硬编码：

```python
../infty_configs/geometry_reshaping
../infty_configs/gradient_filtering
../infty_configs/zeroth_order_updates
./ckp
logs/...
plots/...
```

统一替换为：

```python
Path(args.infty_config_dir) / "geometry_reshaping"
Path(args.infty_config_dir) / "gradient_filtering"
Path(args.infty_config_dir) / "zeroth_order_updates"
Path(args.ckp_dir)
Path(args.log_dir)
Path(args.plot_dir)
```

如果代码把 `args` 转成字典传入 trainer，需要确保以下字段进入字典：

```python
workdir
infty_config_dir
ckp_dir
log_dir
output_dir
plot_dir
```

### 5.2 修改 `examples/PILOT/trainer.py`

把所有硬编码输出路径改为使用 `args`。

典型修改：

```python
from pathlib import Path

log_dir = Path(args.get("log_dir", str(REPO_ROOT / "workdirs" / "logs"))).expanduser().resolve()
log_dir.mkdir(parents=True, exist_ok=True)
```

如果原代码中有：

```python
logging.basicConfig(filename="logs/...")
```

改为：

```python
logging.basicConfig(filename=str(log_dir / "train.log"), ...)
```

如果 trainer 需要按数据集、方法、seed 建子目录，使用：

```python
run_log_dir = log_dir / str(args.get("model_name", "unknown")) / f"seed_{args.get('seed', 'na')}"
run_log_dir.mkdir(parents=True, exist_ok=True)
```

### 5.3 修改 OGD checkpoint 路径

检查：

```text
src/infty/optim/gradient_filtering/ogd.py
```

若存在硬编码：

```python
./ckp/ogd_basis.pt
```

改为可传入路径，优先从 optimizer 参数或 args 中传：

```python
from pathlib import Path

class OGD(...):
    def __init__(self, ..., basis_path=None, ckp_dir=None, **kwargs):
        if basis_path is None:
            ckp_root = Path(ckp_dir or REPO_ROOT / "workdirs" / "checkpoints").expanduser().resolve()
            basis_path = ckp_root / "ogd_basis.pt"
        self.basis_path = Path(basis_path).expanduser().resolve()
        self.basis_path.parent.mkdir(parents=True, exist_ok=True)
```

保存和加载均使用：

```python
torch.save(obj, self.basis_path)
torch.load(self.basis_path, map_location="cpu")
```

不要继续向 `examples/PILOT/ckp` 写文件。

### 5.4 P2 验收

```bash
python -m compileall examples/PILOT src/infty
DRY_RUN=1 bash workdirs/run_scripts/legacy/run_util.sh 2>/dev/null || true
```

P2 结束后，代码搜索不应再出现以下硬编码写入：

```bash
grep -R "\.\/ckp\|logs/\|\.\/plots\|\.\.\/infty_configs" -n examples/PILOT src/infty || true
```

允许在文档、README、legacy 脚本中出现，但不应在实际入口逻辑中出现。

---

## 6. P3：重写 run_scripts

目标：替换旧 `examples/run_scripts`，新脚本放在：

```text
workdirs/run_scripts/
```

保留旧脚本到：

```text
workdirs/run_scripts/legacy/
```

### 6.1 写入 `_common.sh`

创建 `workdirs/run_scripts/_common.sh`：

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${WORKDIR}/.." && pwd)"
PILOT_DIR="${REPO_ROOT}/examples/PILOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_DIR="${INFTY_CONFIG_DIR:-${WORKDIR}/configs/infty}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs}"
CKP_ROOT="${CKP_ROOT:-${WORKDIR}/checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKDIR}/outputs}"
PLOT_ROOT="${PLOT_ROOT:-${WORKDIR}/plots}"

mkdir -p "${LOG_ROOT}" "${CKP_ROOT}" "${OUTPUT_ROOT}" "${PLOT_ROOT}"

require_file() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    echo "[ERROR] Missing file: ${file}" >&2
    exit 1
  fi
}

require_dir() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    echo "[ERROR] Missing directory: ${dir}" >&2
    exit 1
  fi
}

slugify() {
  echo "$1" | tr '/: ' '___' | tr -cd '[:alnum:]_.-'
}

run_cmd() {
  echo "[RUN] $*"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "$@"
  fi
}
```

### 6.2 写入 `run_one.sh`

创建 `workdirs/run_scripts/run_one.sh`：

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

OPTIMIZER="${1:?Usage: run_one.sh <inftyopt> <pilot_config> [tag]}"
PILOT_CONFIG="${2:?Usage: run_one.sh <inftyopt> <pilot_config> [tag]}"
TAG="${3:-default}"

require_dir "${PILOT_DIR}"
require_file "${PILOT_DIR}/${PILOT_CONFIG}"
require_dir "${CONFIG_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SAFE_TAG="$(slugify "${TAG}")"
SAFE_OPT="$(slugify "${OPTIMIZER}")"
RUN_NAME="${SAFE_TAG}_${SAFE_OPT}_${STAMP}"
RUN_LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
RUN_CKP_DIR="${CKP_ROOT}/${RUN_NAME}"
RUN_OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
RUN_PLOT_DIR="${PLOT_ROOT}/${RUN_NAME}"

mkdir -p "${RUN_LOG_DIR}" "${RUN_CKP_DIR}" "${RUN_OUTPUT_DIR}" "${RUN_PLOT_DIR}"

cd "${PILOT_DIR}"

CMD=(
  "${PYTHON_BIN}" main.py
  --inftyopt "${OPTIMIZER}"
  --config "${PILOT_CONFIG}"
  --workdir "${WORKDIR}"
  --infty_config_dir "${CONFIG_DIR}"
  --ckp_dir "${RUN_CKP_DIR}"
  --log_dir "${RUN_LOG_DIR}"
  --output_dir "${RUN_OUTPUT_DIR}"
  --plot_dir "${RUN_PLOT_DIR}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[DRY-RUN]'
  printf ' %q' "${CMD[@]}"
  printf '\n'
else
  "${CMD[@]}" 2>&1 | tee "${RUN_LOG_DIR}/stdout.log"
fi
```

### 6.3 写入 `run_geometry_reshaping.sh`

创建 `workdirs/run_scripts/run_geometry_reshaping.sh`：

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

METHODS=("memo_scr")
OPTS=("sam" "gsam" "looksam" "gam" "c_flat" "c_flat_plus")

for method in "${METHODS[@]}"; do
  for opt in "${OPTS[@]}"; do
    "${SCRIPT_DIR}/run_one.sh" "${opt}" "exps/${method}.json" "geometry_reshaping_${method}"
  done
done
```

### 6.4 写入 `run_zeroth_order_updates.sh`

创建 `workdirs/run_scripts/run_zeroth_order_updates.sh`：

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

METHODS=("ease")
OPTS=("zo_sgd" "zo_sgd_sign" "zo_sgd_conserve" "zo_adam" "zo_adam_sign" "zo_adam_conserve" "forward_grad")

for method in "${METHODS[@]}"; do
  for opt in "${OPTS[@]}"; do
    "${SCRIPT_DIR}/run_one.sh" "${opt}" "exps/${method}.json" "zeroth_order_updates_${method}"
  done
done
```

### 6.5 写入 `run_gradient_filtering.sh`

创建 `workdirs/run_scripts/run_gradient_filtering.sh`：

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

METHODS=("icarl")
OPTS=("pcgrad" "gradvac" "cagrad" "unigrad_fs" "ogd")

for method in "${METHODS[@]}"; do
  for opt in "${OPTS[@]}"; do
    "${SCRIPT_DIR}/run_one.sh" "${opt}" "exps/${method}.json" "gradient_filtering_${method}"
  done
done
```

### 6.6 写入 `run_plots.sh`

创建 `workdirs/run_scripts/run_plots.sh`：

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

require_dir "${PLOT_ROOT}"

cat <<EOF
[INFO] Plot scripts are invoked from Python APIs under src/infty/plot.
[INFO] Use --plot_dir or output_dir in Python calls to write into:
       ${PLOT_ROOT}
EOF
```

### 6.7 设置执行权限

```bash
chmod +x workdirs/run_scripts/*.sh
```

### 6.8 P3 验收

```bash
DRY_RUN=1 bash workdirs/run_scripts/run_geometry_reshaping.sh
DRY_RUN=1 bash workdirs/run_scripts/run_zeroth_order_updates.sh
DRY_RUN=1 bash workdirs/run_scripts/run_gradient_filtering.sh
```

验收要求：

1. 不依赖当前 shell 工作目录。
2. 不硬编码 `conda activate` 或 `source activate infty`。
3. 所有输出路径均在 `workdirs/` 下。
4. 配置不存在时报明确错误。
5. 支持环境变量覆盖：`PYTHON_BIN`、`INFTY_CONFIG_DIR`、`LOG_ROOT`、`CKP_ROOT`、`OUTPUT_ROOT`、`PLOT_ROOT`、`DRY_RUN`。

---

## 7. P4：`src/infty/optim` 论文一致性审查

### 7.1 审查文档模板

创建 `docs/optimizer_audit.md`：

```markdown
# Optimizer Audit

## Scope

- `src/infty/optim/geometry_reshaping`
- `src/infty/optim/zeroth_order_updates`
- `src/infty/optim/gradient_filtering`

## Method-level audit

For each optimizer, record:

1. Paper / algorithm source
2. Expected algorithm steps
3. Actual code path
4. Mismatch, if any
5. Fix applied, if any
6. Tests added
7. Verdict: correct / reasonable with caveat / needs follow-up
```

### 7.2 通用检查标准

每个优化器至少检查：

```text
1. forward/backward 次数是否与论文伪代码一致
2. zero_grad 时机是否正确
3. base_optimizer.step() 是否只在预期位置调用
4. delay=True 或类似延迟参数是否不会更新参数
5. None grad、requires_grad=False 参数是否安全跳过
6. state_dict/load_state_dict 是否保留必要状态
7. CPU 下是否能运行最小 toy test
8. CUDA/DDP 分支是否没有明显路径错误
9. 是否污染 examples/PILOT 工作目录
10. 是否与 PILOT trainer 的调用约定一致
```

### 7.3 平坦损失景观类优化器

目录：

```text
src/infty/optim/geometry_reshaping/
```

涉及方法：

```text
SAM
GSAM
LookSAM
GAM
C-Flat
C-Flat+
```

重点审查：

```text
1. 扰动 epsilon 的范数和 scale 是否按论文公式实现
2. perturb 后是否在第二次 backward/step 前恢复参数
3. BN running stats 是否在扰动 forward 中被正确处理
4. closure 是否被调用正确次数
5. base optimizer 是否只更新一次
6. 多参数组的 rho / alpha / adaptive 配置是否生效
7. delay=True 时是否只统计或缓存而不更新
8. 与 PILOT 的 loss_fn / data loader 调用约定是否一致
```

最小测试建议：

```python
# tests/optim/test_geometry_reshaping.py
import torch


def make_model():
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))


def toy_batch():
    x = torch.randn(6, 4)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    return x, y


def test_flat_optimizer_step_smoke():
    from infty.optim import SAM

    model = make_model()
    base = torch.optim.SGD
    opt = SAM(model.parameters(), base, lr=0.01, rho=0.05)
    criterion = torch.nn.CrossEntropyLoss()
    x, y = toy_batch()

    def closure():
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        return loss

    before = [p.detach().clone() for p in model.parameters()]
    opt.step(closure=closure)
    after = [p.detach().clone() for p in model.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(before, after))
```

Codex 应根据真实类名和构造签名调整 import 与参数。

### 7.4 梯度禁用 / 零阶优化器

目录：

```text
src/infty/optim/zeroth_order_updates/
```

涉及方法：

```text
ZeroFlow
ZO-SGD
ZO-SGD-Sign
ZO-SGD-Conserve
ZO-Adam
ZO-Adam-Sign
ZO-Adam-Conserve
ForwardGrad
```

重点审查：

```text
1. 有限差分公式是一侧还是双侧
2. q > 1 时是否正确平均多个随机方向
3. 随机方向是否可 seed 复现
4. memory_efficient=True 是否能复现方向 / mask
5. conservative update 是否先临时更新再回滚，语义是否正确
6. sign 版本是否只取方向符号而不破坏 scale
7. Adam 状态 m/v 更新是否按 Adam 语义实现
8. 是否真的不依赖 backward
9. no_grad/inference_mode 是否使用合理
10. 参数恢复是否精确
```

最小测试建议：

```python
# tests/optim/test_zeroflow.py
import torch


def test_zeroflow_smoke_no_backward_required():
    from infty.optim import ZeroFlow

    model = torch.nn.Linear(3, 1)
    opt = ZeroFlow(model.parameters(), lr=1e-3, q=2)
    x = torch.randn(5, 3)
    y = torch.randn(5, 1)

    def closure():
        pred = model(x)
        return torch.nn.functional.mse_loss(pred, y)

    before = [p.detach().clone() for p in model.parameters()]
    opt.step(closure=closure)
    after = [p.detach().clone() for p in model.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(before, after))
```

Codex 应根据真实 ZeroFlow 构造签名调整测试。

### 7.5 梯度冲突类优化器

目录：

```text
src/infty/optim/gradient_filtering/
```

涉及方法：

```text
PCGrad
CAGrad
GradVac
UniGrad-FS
OGD
```

重点审查：

```text
1. 多目标 loss 是否被分别 backward 并提取梯度
2. 梯度 flatten / unflatten 是否保持参数顺序
3. None grad 是否填 0 或跳过，是否与论文一致
4. PCGrad 是否只在 dot(g_i, g_j) < 0 时投影
5. CAGrad 约束优化是否稳定，是否处理 scipy 缺失
6. GradVac similarity threshold 是否更新正确
7. UniGrad-FS layer-wise 阈值是否没有重复乘法副作用
8. OGD basis 是否保存在 workdir/checkpoints
9. similarity 记录字段是否统一为 sim_list 或通过属性访问兼容
10. task_id / stage 切换时状态是否重置合理
```

确定性风险点：

```text
1. GradVac / UniGrad-FS 可能使用 sim_arr，而 visualize_conflicts 读取 sim_list，需统一或兼容。
2. GradVac.set_k_idx() / UniGrad_FS.set_k_idx() 如果存在 self.S_T = self.S_T * len(self.k_idx)，要检查是否误把阈值数值乘以层数。若设计意图是每层一个阈值，应改为 expand/repeat 或只初始化一次。
3. OGD 不允许继续硬编码 ./ckp/ogd_basis.pt。
```

建议兼容接口：

```python
class SomeGradientConflictOptimizer(...):
    @property
    def sim_list(self):
        if hasattr(self, "sim_arr"):
            return self.sim_arr
        return getattr(self, "_sim_list", [])
```

或者在 plot 层兼容读取：

```python
sim_values = getattr(optimizer, "sim_list", None)
if sim_values is None:
    sim_values = getattr(optimizer, "sim_arr", None)
if sim_values is None:
    raise AttributeError("optimizer must expose sim_list or sim_arr for conflict visualization")
```

### 7.6 P4 验收

```bash
python -m compileall src/infty/optim
pytest tests/optim -q
```

如果 `pytest` 暂时因外部依赖失败，至少保证：

```bash
python -m compileall src/infty/optim tests/optim
```

并在 `docs/optimizer_audit.md` 写清楚未完成项。

---

## 8. P5：`src/infty/plot` 可视化重构与修复

### 8.1 审查文档模板

创建 `docs/plot_audit.md`：

```markdown
# Plot Audit

## Scope

- `src/infty/plot/visualize_loss_landscape.py`
- `src/infty/plot/visualize_esd.py`
- `src/infty/plot/visualize_conflicts.py`
- `src/infty/plot/visualize_trajectory.py`

## Checks

1. API correctness
2. Model state restoration
3. Output path handling
4. Numerical stability
5. Loader compatibility
6. File existence and naming
7. Known issues and fixes
```

### 8.2 文件命名与兼容策略

推荐把文件重命名：

```bash
git mv src/infty/plot/visualize_loss_landscape.py src/infty/plot/landscape.py
git mv src/infty/plot/visualize_esd.py src/infty/plot/esd.py
git mv src/infty/plot/visualize_conflicts.py src/infty/plot/conflicts.py
git mv src/infty/plot/visualize_trajectory.py src/infty/plot/trajectory.py
```

如果担心破坏外部 import，可以保留兼容 stub：

```python
# src/infty/plot/visualize_loss_landscape.py
from .landscape import *
```

其他三个旧文件同理。

### 8.3 修改 `src/infty/plot/__init__.py`

应导出统一 API，并保留历史别名：

```python
from .landscape import visualize_loss_landscape
from .esd import visualize_esd
from .conflicts import visualize_conflicts
from .trajectory import visualize_trajectory

visualize_landscape = visualize_loss_landscape

__all__ = [
    "visualize_loss_landscape",
    "visualize_landscape",
    "visualize_esd",
    "visualize_conflicts",
    "visualize_trajectory",
]
```

### 8.4 修复 output_dir 默认值

所有 plot 函数都必须支持：

```python
output_dir: str | Path | None = None
```

默认规则：

```python
from pathlib import Path

if output_dir is None:
    output_dir = Path("workdirs/plots") / "<plot_type>"
else:
    output_dir = Path(output_dir)
output_dir = output_dir.expanduser().resolve()
output_dir.mkdir(parents=True, exist_ok=True)
```

不要默认写到 `./plots` 或 `examples/PILOT/plots`。

### 8.5 修复 `visualize_conflicts`

必须兼容 optimizer 的 `sim_list` 和 `sim_arr`：

```python
def _get_similarity_values(optimizer):
    sim_values = getattr(optimizer, "sim_list", None)
    if sim_values is None:
        sim_values = getattr(optimizer, "sim_arr", None)
    if sim_values is None:
        raise AttributeError("optimizer must expose sim_list or sim_arr for conflict visualization")
    return sim_values
```

保存路径建议：

```text
workdirs/plots/conflicts/{optimizer_name}/task_{task}.pdf
```

如果 similarity 为空，给出明确错误或生成空图并记录 warning，不要直接崩溃在 matplotlib 内部。

### 8.6 修复 `visualize_trajectory`

如果当前函数内出现：

```python
plot_contour(F, init, traj, optimizer_name)
```

但 `traj` 未定义，则修复为：

```python
traj = run(optimizer_name, lr, init, n_iter)
plot_contour(F, init, traj, optimizer_name, output_dir=output_dir)
```

要求：

```text
1. run() 返回 trajectory
2. plot_contour() 接收 output_dir
3. 所有文件写到 output_dir
4. 函数参数有默认值，smoke test 可以直接调用
```

### 8.7 模型状态恢复要求

对于 landscape / ESD 这类会改动模型参数或模式的可视化函数，必须做到：

```python
was_training = model.training
state_backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
try:
    ...
finally:
    model.load_state_dict(state_backup, strict=True)
    model.train(was_training)
```

如果 state_dict 过大导致性能问题，可以只备份参数向量，但必须在 `docs/plot_audit.md` 说明取舍。

### 8.8 Plot 测试建议

创建：

```text
tests/plot/test_conflicts_smoke.py
tests/plot/test_trajectory_smoke.py
tests/plot/test_plot_exports.py
```

示例：

```python
# tests/plot/test_conflicts_smoke.py
from pathlib import Path


class DummyOptimizer:
    name = "dummy"
    sim_arr = [0.1, -0.2, 0.3]


def test_visualize_conflicts_accepts_sim_arr(tmp_path):
    from infty.plot import visualize_conflicts

    visualize_conflicts(DummyOptimizer(), output_dir=tmp_path, task=0)
    assert any(Path(tmp_path).rglob("*.pdf")) or any(Path(tmp_path).rglob("*.png"))
```

```python
# tests/plot/test_plot_exports.py

def test_plot_exports():
    import infty.plot as p

    assert hasattr(p, "visualize_landscape")
    assert hasattr(p, "visualize_loss_landscape")
    assert hasattr(p, "visualize_esd")
    assert hasattr(p, "visualize_conflicts")
    assert hasattr(p, "visualize_trajectory")
```

### 8.9 P5 验收

```bash
python -m compileall src/infty/plot tests/plot
pytest tests/plot -q
```

---

## 9. P6：最终回归与 README 更新

### 9.1 更新 README

在 README 中更新 Quick Start 路径：

旧方式如果是：

```bash
cd examples/PILOT
python main.py --config=./exps/memo_scr.json --inftyopt sam
```

改为：

```bash
cd examples/PILOT
python main.py \
  --config=./exps/memo_scr.json \
  --inftyopt sam \
  --workdir ../../workdirs \
  --infty_config_dir ../../workdirs/configs/infty \
  --ckp_dir ../../workdirs/checkpoints/demo_sam \
  --log_dir ../../workdirs/logs/demo_sam \
  --output_dir ../../workdirs/outputs/demo_sam \
  --plot_dir ../../workdirs/plots/demo_sam
```

新增脚本运行方式：

```bash
DRY_RUN=1 bash workdirs/run_scripts/run_geometry_reshaping.sh
bash workdirs/run_scripts/run_geometry_reshaping.sh
```

新增 plot API 示例：

```python
from infty import plot as infty_plot

infty_plot.visualize_landscape(
    optimizer=optimizer,
    model=model,
    create_loss_fn=create_loss_fn,
    loader=train_loader,
    task=task_id,
    device=device,
    output_dir="workdirs/plots/landscape/demo",
)
```

### 9.2 追加迁移报告

向 `docs/migration_report.md` 追加：

```markdown
## Migration Summary

### Moved

- `examples/infty_configs` -> `workdirs/configs/infty`
- `examples/run_scripts` -> `workdirs/run_scripts/legacy`
- runtime artifacts under `examples/PILOT` -> `workdirs/*`

### Modified

- `examples/PILOT/main.py`
- `examples/PILOT/trainer.py`
- `src/infty/optim/...`
- `src/infty/plot/...`
- `README.md`
- `.gitignore`

### Validation

Paste command outputs here.
```

### 9.3 最终验收命令

```bash
python -m compileall src examples/PILOT tests
pytest tests/optim -q
pytest tests/plot -q
DRY_RUN=1 bash workdirs/run_scripts/run_geometry_reshaping.sh
DRY_RUN=1 bash workdirs/run_scripts/run_zeroth_order_updates.sh
DRY_RUN=1 bash workdirs/run_scripts/run_gradient_filtering.sh
```

如果安装了 `ruff`：

```bash
ruff check src examples/PILOT tests
```

### 9.4 最终 Git 检查

```bash
git status --short
find examples/PILOT -maxdepth 1 -type d | sort
find workdirs -maxdepth 3 -type d | sort
```

确认：

```text
1. examples/PILOT 不含日志、checkpoint、plot、output、wandb 运行产物。
2. workdirs/configs/infty 存在并纳入版本控制。
3. workdirs/run_scripts 存在并纳入版本控制。
4. run_scripts dry-run 可正常打印命令。
5. docs 下有 optimizer_audit、plot_audit、migration_report。
```

---

## 10. 迁移矩阵

| 当前路径或文件类型 | 目标路径 | 处理方式 |
|---|---|---|
| `examples/infty_configs/` | `workdirs/configs/infty/` | `git mv` |
| `examples/run_scripts/` | `workdirs/run_scripts/legacy/` | 迁移保留，后续新脚本替代 |
| `examples/PILOT/ckp/` | `workdirs/checkpoints/pilot_ckp/` | 移动 |
| `examples/PILOT/checkpoints/` | `workdirs/checkpoints/pilot_checkpoints/` | 移动 |
| `examples/PILOT/logs/` | `workdirs/logs/pilot_logs/` | 移动 |
| `examples/PILOT/plots/` | `workdirs/plots/pilot_plots/` | 移动 |
| `examples/PILOT/outputs/` | `workdirs/outputs/pilot_outputs/` | 移动 |
| `examples/PILOT/results/` | `workdirs/outputs/pilot_results/` | 移动 |
| `examples/PILOT/wandb/` | `workdirs/logs/wandb/` | 移动 |
| `*.pt`, `*.pth`, `*.ckpt` | `workdirs/checkpoints/` | 运行产物，不放 PILOT 源码目录 |
| `*.pdf`, `*.png`, `*.jpg` | `workdirs/plots/` 或 `outputs/figures/` | 可视化产物 |
| `__pycache__/`, `.pytest_cache/` | 删除 | 不提交 |
| `.ipynb_checkpoints/` | 删除 | 不提交 |

---

## 11. 优先级最高的确定性修复项

Codex 请优先修复以下问题：

1. `examples/PILOT/main.py` 中读取 `../infty_configs` 的逻辑，改为 `--infty_config_dir`。
2. `examples/PILOT/trainer.py` 中写入 `logs/...` 的逻辑，改为 `--log_dir`。
3. `src/infty/optim/gradient_filtering/ogd.py` 中写入 `./ckp/ogd_basis.pt` 的逻辑，改为 `--ckp_dir` 或 optimizer 参数派生路径。
4. `src/infty/plot/visualize_trajectory.py` 中如果存在未定义的 `traj`，必须修复为先调用 `run(...)` 得到轨迹。
5. `src/infty/plot/visualize_conflicts.py` 中如果只读取 `sim_list`，必须兼容 `sim_arr`。
6. `src/infty/plot/__init__.py` 中必须同时导出 `visualize_landscape` 和 `visualize_loss_landscape`。
7. 旧 run scripts 不要直接删除，先移动到 `workdirs/run_scripts/legacy/`。

---

## 12. 不要做的事情

1. 不要把日志、checkpoint、plot 继续写到 `examples/PILOT/`。
2. 不要把 `workdirs/logs/`、`checkpoints/`、`plots/`、`outputs/` 纳入 Git。
3. 不要在 run scripts 中硬编码 `source activate infty` 或具体 conda 环境名。
4. 不要为了通过测试而跳过真实算法逻辑。
5. 不要在没有文档说明的情况下修改论文公式。
6. 不要从 PILOT 官方仓库复制 LICENSE/README/resources 内容，除非项目维护者明确允许；如果本地缺这些文件，只在 `docs/migration_report.md` 说明差异。
7. 不要在 plot 函数中无条件改动模型参数后不恢复。

---

## 13. Codex 可执行检查清单

按顺序完成并打勾：

```text
[ ] P0：创建分支并生成 docs/migration_report.md 基线
[ ] P1：创建 workdirs 并迁移 infty_configs/run_scripts/运行产物
[ ] P1：更新 .gitignore
[ ] P2：main.py 增加 workdir/infty_config_dir/log_dir/output_dir/plot_dir/ckp_dir 参数
[ ] P2：trainer.py 改为使用 log_dir/output_dir
[ ] P2：OGD basis/checkpoint 路径改为 workdir/checkpoints
[ ] P3：写入新的 run_scripts
[ ] P3：dry-run 三类脚本通过
[ ] P4：创建 docs/optimizer_audit.md
[ ] P4：修复 optim 层确定性 bug
[ ] P4：补 tests/optim
[ ] P5：创建 docs/plot_audit.md
[ ] P5：修复 plot API、output_dir、trajectory、conflicts
[ ] P5：补 tests/plot
[ ] P6：更新 README Quick Start
[ ] P6：最终 compileall / pytest / dry-run 验收
[ ] P6：git status 检查并整理 migration_report
```

---

## 14. 建议提交信息

完成后建议使用以下提交信息：

```text
refactor: isolate INFTY workdirs and normalize scripts
```

如果拆成多个提交：

```text
refactor: move INFTY configs and runtime artifacts into workdirs
refactor: parameterize PILOT runtime paths
refactor: rewrite INFTY run scripts
fix: normalize plot APIs and output directories
fix: make optimizer checkpoints use workdir
test: add optimizer and plot smoke tests
```

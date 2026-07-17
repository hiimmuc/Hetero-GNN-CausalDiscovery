# GHCF Ablation Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Note on verification style:** this repo has no test suite (confirmed in CLAUDE.md). "Tests" in this
> plan are smoke-run verifications (dry-run counts, tiny real runs) rather than pytest, matching the
> project's existing convention (train.py / cross_validation.py have no unit tests either).

**Goal:** Train and cross-validate every valid combination of GHCF model variant, loss mode, adjacency
type, and GNN layer type, then produce a notebook comparing them across all ranking metrics.

**Architecture:** A pruned full-factorial config grid (1080 entries) generated in Python, executed by a
parallel subprocess orchestrator (16 workers) that shells out to the existing `main.py train --cv` CLI
per config, with a resumable JSONL manifest recording each config's parsed `cv_results.json` metrics.
A `--run_tag` addition to the core CLI keeps each config's `checkpoints/cv_*/` output directory unique
and identifiable under parallel execution. A final notebook loads the manifest and plots comparisons.

**Tech Stack:** Python 3.10 (conda env `eHMI`), PyTorch, existing `main.py`/`cross_validation.py` CLI,
`concurrent.futures.ThreadPoolExecutor` for the worker pool, pandas/matplotlib/seaborn for the notebook.

## Global Constraints

- All commands run from `Experiment3/Workspace/Models/` (paths in `config.py` are resolved relative to it).
- CV settings for every config: `--cv --n_folds 5 --stratified --cv_epochs 500 --patience 50` (per user request).
- `NORMALIZE_INTERACTIONS` stays enabled (default) for every config — required by causal/ace, and required
  for `RATING_THRESHOLD = 0.0` to mean "above-average rating" rather than "almost everything" (verified:
  RATING is z-scored when normalization is on; the threshold is read identically by `train.py`'s hybrid-loss
  BPR positive/negative split and `batch_processing.py`'s ranking-metric ground truth).
- Batch size (16) and learning rate (0.001, AdamW + cosine annealing) stay at the repo's existing
  `config.py` defaults for every config — held constant so the sweep isolates architecture/loss/adjacency
  effects instead of confounding them with per-config tuning. Verified via smoke test: converges cleanly
  (~epoch 55 early stop) with no divergence.
- 16 parallel workers (user's explicit choice, accepting CPU oversubscription — 32 cores, ~4 internal
  workers per job — as a wall-clock ceiling rather than throughput-optimal).
- Disk is not a concern: a 2-fold smoke run used ~2.9MB; 1080 configs × 5 folds is estimated well under 10GB
  against 359GB free.

### Grid axes and pruning rules

| Axis | Values | Applies when |
|---|---|---|
| `interaction_set` | `default` (all 4 types), `single` (RATING only), `multiple` (RATING+BFI_PERCEPTION) | always |
| `causal_mode` | `none`, `causal` (GHCF_CE), `ace_learnable`, `ace_fixed_5050`, `ace_fixed_7030`, `ace_fixed_3070` | `causal`/`ace_*` only valid when `BFI_PERCEPTION` is in `interaction_set` (i.e. `default`/`multiple`; code raises `ValueError` otherwise for `GHCF_CE`/`GHCF_ACE`) |
| `bound_bfi` | `True`, `False` | only when `BFI_PERCEPTION` present (`default`/`multiple`); collapses to a single "n/a" value for `single` (flag is a no-op there — `GHCF.py:386` only reads it for the `BFI_PERCEPTION` head) |
| `freeze_sem` | `True`, `False` | only when `causal_mode != none` (no SEM head otherwise); collapses to a single "n/a" value for `causal_mode=none` |
| `loss` | `mse`, `hybrid` | always |
| `adj_type` | `pre`, `norm`, `mean`, `plain` | always |
| `gnn_type` | `gcn`, `sage`, `gat` | always, full factorial (user's explicit choice over the smaller "backbone-only" option) |

Resulting counts: `single` = 1×1×1×2×4×3 = 24; `default` = (2×1 + 4×2×2 + 4×2×2×... see below) — computed
programmatically in Task 2, expected total **1080** (single: 24, default: 528, multiple: 528).

ACE fixed-blend weights (`ace_fixed_5050`/`_7030`/`_3070`) map to `--ace_w_mlp`/`--ace_w_causal` of
`0.5/0.5`, `0.7/0.3`, `0.3/0.7` respectively (`w_mlp` first). `ace_learnable` uses
`--ace_weight_mode learnable` with no weight flags.

---

## Task 1: Add `--run_tag` for unique, identifiable CV output directories

**Files:**
- Modify: `Experiment3/Workspace/Models/src/utility/parser.py` (`add_cv_arguments`, ~line 212-241)
- Modify: `Experiment3/Workspace/Models/src/cross_validation.py` (`CrossValidationTrainer.setup_cv_directory`, ~line 135-150)

**Interfaces:**
- Produces: `args.run_tag: str` (default `""`, backward compatible), and
  `checkpoints/cv_{run_tag}_{timestamp}/` directory naming when `run_tag` is non-empty (falls back to the
  current `checkpoints/cv_{timestamp}` when empty, so existing manual CV usage is unaffected).

Why this is necessary (not optional): `setup_cv_directory` names directories `cv_{timestamp}` at
second resolution. Under 16-way parallel launches, two subprocesses starting in the same second would
collide on the same directory and silently corrupt each other's fold outputs. A tag derived from the
config's own identity guarantees uniqueness regardless of timing.

- [ ] **Step 1: Add the `--run_tag` argument**

In `src/utility/parser.py`, inside `add_cv_arguments`, after the existing `--normalize_interactions`
argument:

```python
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help=(
            "Optional tag prepended to the CV output directory name "
            "(checkpoints/cv_{run_tag}_{timestamp}/) so parallel sweep runs "
            "don't collide on second-resolution timestamps."
        ),
    )
```

- [ ] **Step 2: Use the tag when naming the CV directory**

In `src/cross_validation.py`, replace `setup_cv_directory`:

```python
    def setup_cv_directory(self) -> None:
        """Setup cross-validation output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_tag = getattr(self.args, "run_tag", "") or ""
        dir_name = f"cv_{run_tag}_{timestamp}" if run_tag else f"cv_{timestamp}"
        self.cv_dir = Path(f"checkpoints/{dir_name}")
        self.cv_dir.mkdir(parents=True, exist_ok=True)
```

(rest of the method — writing `cv_config.json` — is unchanged)

- [ ] **Step 3: Verify with a real smoke run**

```bash
cd Experiment3/Workspace/Models
python src/main.py train --cv --n_folds 2 --cv_epochs 5 --run_tag smoketest_tag > /tmp/tag_smoke.log 2>&1
ls checkpoints/ | grep smoketest_tag
```

Expected: a directory named `cv_smoketest_tag_<timestamp>` exists, and `cv_results.json`/`cv_summary.csv`
are present inside it. Also run once *without* `--run_tag` to confirm the old `cv_<timestamp>` naming
still works unchanged.

- [ ] **Step 4: Commit**

```bash
cd /home/namdp15/Workspace/Personal/APMV-exp3
git add Experiment3/Workspace/Models/src/utility/parser.py Experiment3/Workspace/Models/src/cross_validation.py
git commit -m "feat: add --run_tag for unique CV output dirs under parallel sweeps"
```

---

## Task 2: Sweep grid generator

**Files:**
- Create: `Experiment3/Workspace/Models/scripts/sweep/__init__.py` (empty)
- Create: `Experiment3/Workspace/Models/scripts/sweep/grid.py`

**Interfaces:**
- Produces: `build_grid() -> list[dict]`, where each dict has keys:
  `tag: str`, `interaction_set: str`, `causal_mode: str`, `bound_bfi: bool | None`,
  `freeze_sem: bool | None`, `loss: str`, `adj_type: str`, `gnn_type: str`, `cli_args: list[str]`
  (the full argv list after `train`, ready to append to `["python", "src/main.py", "train"]`).
- Consumed by: Task 3's orchestrator (`from sweep.grid import build_grid`).

- [ ] **Step 1: Write `grid.py`**

```python
"""Generates the full pruned-factorial ablation grid for the GHCF sweep."""

from typing import Dict, List, Optional

INTERACTION_SETS = {
    "default": ["RATING", "BFI_PERCEPTION", "VOICE_QUALITY", "SUBJECTIVE_FEELING"],
    "single": ["RATING"],
    "multiple": ["RATING", "BFI_PERCEPTION"],
}

CAUSAL_MODES = [
    "none",
    "causal",
    "ace_learnable",
    "ace_fixed_5050",
    "ace_fixed_7030",
    "ace_fixed_3070",
]

ACE_FIXED_SPLITS = {
    "ace_fixed_5050": (0.5, 0.5),
    "ace_fixed_7030": (0.7, 0.3),
    "ace_fixed_3070": (0.3, 0.7),
}

LOSSES = ["mse", "hybrid"]
ADJ_TYPES = ["pre", "norm", "mean", "plain"]
GNN_TYPES = ["gcn", "sage", "gat"]

CV_COMMON_ARGS = [
    "--cv",
    "--n_folds", "5",
    "--stratified",
    "--cv_epochs", "500",
    "--patience", "50",
]


def _causal_modes_for(interaction_set: str) -> List[str]:
    if "BFI_PERCEPTION" not in INTERACTION_SETS[interaction_set]:
        return ["none"]
    return CAUSAL_MODES


def _bound_bfi_values(interaction_set: str) -> List[Optional[bool]]:
    if "BFI_PERCEPTION" not in INTERACTION_SETS[interaction_set]:
        return [None]
    return [True, False]


def _freeze_sem_values(causal_mode: str) -> List[Optional[bool]]:
    if causal_mode == "none":
        return [None]
    return [True, False]


def _causal_cli_args(causal_mode: str, freeze_sem: Optional[bool]) -> List[str]:
    if causal_mode == "none":
        return []
    if causal_mode == "causal":
        args = ["--causal"]
    elif causal_mode == "ace_learnable":
        args = ["--ace", "--ace_weight_mode", "learnable"]
    else:
        w_mlp, w_causal = ACE_FIXED_SPLITS[causal_mode]
        args = [
            "--ace", "--ace_weight_mode", "fixed",
            "--ace_w_mlp", str(w_mlp), "--ace_w_causal", str(w_causal),
        ]
    args.append("--freeze_sem" if freeze_sem else "--no-freeze_sem")
    return args


def _tag(interaction_set, causal_mode, bound_bfi, freeze_sem, loss, adj_type, gnn_type) -> str:
    parts = [interaction_set, causal_mode]
    if bound_bfi is not None:
        parts.append(f"bb{int(bound_bfi)}")
    if freeze_sem is not None:
        parts.append(f"fs{int(freeze_sem)}")
    parts += [loss, adj_type, gnn_type]
    return "_".join(parts)


def build_grid() -> List[Dict]:
    grid = []
    for interaction_set, interactions in INTERACTION_SETS.items():
        for causal_mode in _causal_modes_for(interaction_set):
            for bound_bfi in _bound_bfi_values(interaction_set):
                for freeze_sem in _freeze_sem_values(causal_mode):
                    for loss in LOSSES:
                        for adj_type in ADJ_TYPES:
                            for gnn_type in GNN_TYPES:
                                tag = _tag(
                                    interaction_set, causal_mode, bound_bfi,
                                    freeze_sem, loss, adj_type, gnn_type,
                                )
                                cli_args = (
                                    ["--interactions", *interactions]
                                    + CV_COMMON_ARGS
                                    + ["--loss", loss, "--adj_type", adj_type, "--gnn_type", gnn_type]
                                    + _causal_cli_args(causal_mode, freeze_sem)
                                    + ["--run_tag", tag]
                                )
                                if bound_bfi is not None:
                                    cli_args.append("--bound_bfi" if bound_bfi else "--no-bound_bfi")
                                grid.append({
                                    "tag": tag,
                                    "interaction_set": interaction_set,
                                    "causal_mode": causal_mode,
                                    "bound_bfi": bound_bfi,
                                    "freeze_sem": freeze_sem,
                                    "loss": loss,
                                    "adj_type": adj_type,
                                    "gnn_type": gnn_type,
                                    "cli_args": cli_args,
                                })
    return grid


if __name__ == "__main__":
    grid = build_grid()
    tags = [g["tag"] for g in grid]
    print(f"Total configs: {len(grid)}")
    print(f"Unique tags: {len(set(tags))}")
    assert len(tags) == len(set(tags)), "duplicate tags detected"
```

- [ ] **Step 2: Verify grid size and tag uniqueness**

```bash
cd Experiment3/Workspace/Models
python scripts/sweep/grid.py
```

Expected: `Total configs: 1080` and `Unique tags: 1080` (no assertion error). If the count differs, recheck
the pruning rules against the table above before proceeding — do not adjust the expected number without
re-deriving it by hand first.

- [ ] **Step 3: Commit**

```bash
cd /home/namdp15/Workspace/Personal/APMV-exp3
git add Experiment3/Workspace/Models/scripts/
git commit -m "feat: add ablation sweep grid generator (1080 configs)"
```

---

## Task 3: Parallel sweep orchestrator with resumable manifest

**Files:**
- Create: `Experiment3/Workspace/Models/scripts/sweep/run_sweep.py`

**Interfaces:**
- Consumes: `sweep.grid.build_grid()` from Task 2.
- Produces: `checkpoints/sweep_<sweep_id>/manifest.jsonl` (one JSON line per completed/failed config:
  `tag`, all grid fields, `status` ("ok"/"failed"), `cv_dir`, `mean_metrics`, `std_metrics`, `wall_time_s`),
  and `checkpoints/sweep_<sweep_id>/logs/<tag>.log` (full stdout/stderr per config).

- [ ] **Step 1: Write `run_sweep.py`**

```python
"""Orchestrates the GHCF ablation sweep: launches main.py train --cv per grid
config across a worker pool, with a resumable JSONL manifest."""

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from grid import build_grid

MODELS_DIR = Path(__file__).resolve().parents[2]
_manifest_lock = threading.Lock()


def load_completed_tags(manifest_path: Path) -> set:
    if not manifest_path.exists():
        return set()
    completed = set()
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("status") == "ok":
                completed.add(record["tag"])
    return completed


def append_manifest(manifest_path: Path, record: dict) -> None:
    with _manifest_lock:
        with open(manifest_path, "a") as f:
            f.write(json.dumps(record) + "\n")


def find_cv_dir(tag: str) -> str:
    matches = sorted(MODELS_DIR.glob(f"checkpoints/cv_{tag}_*"))
    return str(matches[-1]) if matches else ""


def run_one(config: dict, log_dir: Path, manifest_path: Path) -> None:
    tag = config["tag"]
    log_path = log_dir / f"{tag}.log"
    cmd = ["python", "src/main.py", "train"] + config["cli_args"]
    start = time.time()
    with open(log_path, "w") as log_file:
        log_file.write(f"CMD: {' '.join(cmd)}\n\n")
        log_file.flush()
        proc = subprocess.run(
            cmd, cwd=str(MODELS_DIR), stdout=log_file, stderr=subprocess.STDOUT
        )
    wall_time = time.time() - start

    record = {k: v for k, v in config.items() if k != "cli_args"}
    cv_dir = find_cv_dir(tag)
    record.update({"wall_time_s": wall_time, "cv_dir": cv_dir})

    if proc.returncode != 0 or not cv_dir:
        record["status"] = "failed"
        record["mean_metrics"] = None
        record["std_metrics"] = None
        append_manifest(manifest_path, record)
        print(f"[FAILED] {tag} (see {log_path})")
        return

    results_path = Path(cv_dir) / "cv_results.json"
    try:
        with open(results_path) as f:
            cv_results = json.load(f)
        record["status"] = "ok"
        record["mean_metrics"] = cv_results["mean_metrics"]
        record["std_metrics"] = cv_results["std_metrics"]
    except Exception as exc:
        record["status"] = "failed"
        record["mean_metrics"] = None
        record["std_metrics"] = None
        record["error"] = str(exc)

    append_manifest(manifest_path, record)
    print(f"[{record['status'].upper()}] {tag} ({wall_time:.0f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, default=None, help="Resume an existing sweep dir name")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N configs (testing)")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    grid = build_grid()
    if args.limit:
        grid = grid[: args.limit]

    if args.dry_run:
        print(f"Planned configs: {len(grid)}")
        for config in grid[:5]:
            print(" ", config["tag"], "->", " ".join(config["cli_args"]))
        print("  ...")
        return

    sweep_id = args.sweep_id or time.strftime("%Y%m%d_%H%M%S")
    sweep_dir = MODELS_DIR / "checkpoints" / f"sweep_{sweep_id}"
    log_dir = sweep_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sweep_dir / "manifest.jsonl"

    completed = load_completed_tags(manifest_path)
    pending = [c for c in grid if c["tag"] not in completed]
    print(f"Sweep dir: {sweep_dir}")
    print(f"Total: {len(grid)}  Already done: {len(completed)}  Pending: {len(pending)}")

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, c, log_dir, manifest_path) for c in pending]
        for future in futures:
            future.result()

    print("Sweep complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify `--dry_run` shows 1080 configs**

```bash
cd Experiment3/Workspace/Models
python scripts/sweep/run_sweep.py --dry_run
```

Expected: `Planned configs: 1080` followed by 5 example command lines.

- [ ] **Step 3: Commit**

```bash
cd /home/namdp15/Workspace/Personal/APMV-exp3
git add Experiment3/Workspace/Models/scripts/sweep/run_sweep.py
git commit -m "feat: add parallel ablation sweep orchestrator with resumable manifest"
```

---

## Task 4: Smoke-test on a small subset before the full run

**Files:** none (verification only)

- [ ] **Step 1: Run a 3-config, 2-worker smoke test**

```bash
cd Experiment3/Workspace/Models
python scripts/sweep/run_sweep.py --sweep_id smoketest --limit 3 --workers 2
```

Expected: 3 lines like `[OK] <tag> (Ns)`, and `checkpoints/sweep_smoketest/manifest.jsonl` containing 3
JSON lines each with non-null `mean_metrics`.

- [ ] **Step 2: Verify resume-skip logic**

```bash
python scripts/sweep/run_sweep.py --sweep_id smoketest --limit 3 --workers 2
```

Expected: `Already done: 3  Pending: 0`, no new subprocesses launched, script exits immediately.

- [ ] **Step 3: Inspect one config's log and manifest record for correctness**

Read `checkpoints/sweep_smoketest/logs/<any-tag>.log` and confirm the printed `CMD:` line matches the
expected CLI flags for that tag (e.g. an `ace_fixed_7030` tag should show
`--ace --ace_weight_mode fixed --ace_w_mlp 0.7 --ace_w_causal 0.3`). Confirm the manifest record's
`mean_metrics` contains the expected metric keys (`precision@3`, `recall@10`, `ndcg@10`, `hit@10`, `auc`, etc).

- [ ] **Step 4: Clean up the smoketest sweep directory**

```bash
rm -rf checkpoints/sweep_smoketest checkpoints/cv_smoketest_*
```

(Only remove `sweep_smoketest`/`cv_smoketest_*` — never a broader `checkpoints/*` glob.)

---

## Task 5: Launch the full sweep in the background

**Files:** none (execution only)

- [ ] **Step 1: Confirm with the user that the Task 4 smoke test looked correct**, then launch:

```bash
cd Experiment3/Workspace/Models
nohup python scripts/sweep/run_sweep.py --sweep_id full --workers 16 \
  > checkpoints/sweep_full_stdout.log 2>&1 &
```

Run this via the harness's background-Bash mechanism (not a bare `nohup &` in an interactive shell) so
progress can be monitored and the harness notifies on completion.

- [ ] **Step 2: Periodically check progress**

```bash
wc -l checkpoints/sweep_full/manifest.jsonl
```

Compare against 1080 to track completion. Report status to the user at reasonable intervals rather than
waiting silently for the full multi-hour run.

---

## Task 6: Sweep analysis / comparison notebook

**Files:**
- Create: `Experiment3/Workspace/Models/notebooks/sweep_analysis.ipynb`

**Interfaces:**
- Consumes: `checkpoints/sweep_full/manifest.jsonl` (schema from Task 3: `tag`, `interaction_set`,
  `causal_mode`, `bound_bfi`, `freeze_sem`, `loss`, `adj_type`, `gnn_type`, `status`, `mean_metrics`,
  `std_metrics`).

- [ ] **Step 1: Load the skill `dataviz` before writing any chart cells** (per the standing instruction to
  invoke it before producing charts/plots).

- [ ] **Step 2: Build the notebook** with:
  1. Load `manifest.jsonl` into a flat pandas DataFrame (one row per config, `mean_metrics` keys expanded
     into columns, `status == "ok"` filtered).
  2. Overall leaderboard table: top-N configs by `recall@10` (the CV framework's own primary metric, per
     `cross_validation.py`'s `_aggregate_results`), with all metrics shown.
  3. Per-axis comparison plots: for each of `interaction_set`, `causal_mode`, `bound_bfi`, `freeze_sem`,
     `loss`, `adj_type`, `gnn_type`, a grouped bar/box plot of each key metric (precision/recall/ndcg/hit
     @{3,5,10}, auc) aggregated across all other axes, with error bars from `std_metrics`.
  4. A causal-mode-focused view isolating the effect of `freeze_sem` and `bound_bfi` within
     `causal`/`ace_*` modes specifically (the two flags that only exist for those modes).
  5. An ACE fixed-weight comparison (`ace_fixed_5050` vs `_7030` vs `_3070`) against `ace_learnable`.
  6. Save summary tables as CSV alongside the notebook for reuse outside Jupyter.

- [ ] **Step 3: Run the notebook end-to-end** against whatever fraction of `manifest.jsonl` exists at the
  time (the notebook should work on partial sweep results, not require all 1080 rows) and confirm every
  cell executes without error.

- [ ] **Step 4: Commit**

```bash
cd /home/namdp15/Workspace/Personal/APMV-exp3
git add Experiment3/Workspace/Models/notebooks/
git commit -m "feat: add ablation sweep comparison notebook"
```

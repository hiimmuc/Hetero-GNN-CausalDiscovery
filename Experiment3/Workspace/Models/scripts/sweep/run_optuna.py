"""Drives the GHCF ablation search with Optuna instead of grid.py's exhaustive
grid. Reuses run_sweep.py's GPU-memory-aware worker pool and subprocess
pattern; each worker thread calls study.ask()/study.tell() against a single
SQLite-backed study instead of pulling from a precomputed list, so the search
is TPE-driven rather than exhaustive. Trials run to completion (no pruning)."""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import optuna
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from run_sweep import SlotTracker, _tick_worker_bars, find_cv_dir, wait_for_gpu_memory
from search_space import suggest_params

MODELS_DIR = Path(__file__).resolve().parents[2]
OBJECTIVE_METRIC = "ndcg@10"
_manifest_lock = threading.Lock()


def append_manifest(manifest_path: Path, record: dict) -> None:
    with _manifest_lock, open(manifest_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def run_trial(
    trial: optuna.Trial,
    log_dir: Path,
    manifest_path: Path,
    threads_per_worker: int,
    tracker: SlotTracker,
    overall_bar: tqdm,
) -> float:
    sampled = suggest_params(trial)
    config, cli_args = sampled["config"], sampled["cli_args"]
    tag = f"trial{trial.number}"
    log_path = log_dir / f"{tag}.log"
    cmd = ["python", "src/main.py", "train"] + cli_args

    wait_for_gpu_memory()

    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = str(threads_per_worker)

    slot = tracker.acquire(tag)
    try:
        start = time.time()
        with open(log_path, "w") as log_file:
            log_file.write(f"CMD: {' '.join(cmd)}\n\n")
            log_file.write(f"THREADS_PER_WORKER: {threads_per_worker}\n\n")
            log_file.flush()
            proc = subprocess.run(
                cmd, cwd=str(MODELS_DIR), stdout=log_file, stderr=subprocess.STDOUT, env=env
            )
        wall_time = time.time() - start
    finally:
        tracker.release(slot)

    record = dict(config, tag=tag, wall_time_s=wall_time)
    cv_dir = find_cv_dir(tag)
    record["cv_dir"] = cv_dir

    if proc.returncode != 0 or not cv_dir:
        record["status"] = "failed"
        append_manifest(manifest_path, record)
        tqdm.write(f"[FAILED] {tag} (see {log_path})")
        overall_bar.update(1)
        raise optuna.TrialPruned()

    try:
        with open(Path(cv_dir) / "cv_results.json") as f:
            cv_results = json.load(f)
        value = cv_results["mean_metrics"][OBJECTIVE_METRIC]
        record["status"] = "ok"
        record["mean_metrics"] = cv_results["mean_metrics"]
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        append_manifest(manifest_path, record)
        tqdm.write(f"[FAILED] {tag} (see {log_path})")
        overall_bar.update(1)
        raise optuna.TrialPruned()

    append_manifest(manifest_path, record)
    tqdm.write(f"[OK] {tag} {OBJECTIVE_METRIC}={value:.4f} ({wall_time:.0f}s)")
    overall_bar.update(1)
    return value


def worker_loop(
    study: optuna.Study,
    remaining: list,
    remaining_lock: threading.Lock,
    log_dir: Path,
    manifest_path: Path,
    threads_per_worker: int,
    tracker: SlotTracker,
    overall_bar: tqdm,
) -> None:
    while True:
        with remaining_lock:
            if remaining[0] <= 0:
                return
            remaining[0] -= 1

        trial = study.ask()
        try:
            value = run_trial(trial, log_dir, manifest_path, threads_per_worker, tracker, overall_bar)
        except optuna.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            continue
        study.tell(trial, value)


def dry_run(n: int) -> None:
    study = optuna.create_study(direction="maximize")
    for _ in range(n):
        trial = study.ask()
        sampled = suggest_params(trial)
        print(sampled["config"], "->", " ".join(sampled["cli_args"]))
        study.tell(trial, state=optuna.trial.TrialState.FAIL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study_name", type=str, default=None, help="Resume an existing study (also the sqlite filename)"
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--n_trials", type=int, required=True)
    parser.add_argument("--dry_run", action="store_true", help="Print sampled configs, run nothing")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(min(args.n_trials, 5))
        return

    study_name = args.study_name or time.strftime("%Y%m%d_%H%M%S")
    sweep_dir = MODELS_DIR / "checkpoints" / f"optuna_{study_name}"
    log_dir = sweep_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sweep_dir / "manifest.jsonl"

    storage = f"sqlite:///{sweep_dir}.db"
    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize", load_if_exists=True
    )

    already_done = len(
        [t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE,)]
    )
    threads_per_worker = max(1, (os.cpu_count() or args.workers) // args.workers)
    print(f"Study: {study_name} ({storage})")
    print(f"Target trials: {args.n_trials}  Already complete: {already_done}")
    print(f"Workers: {args.workers}  Threads/worker: {threads_per_worker}")

    tracker = SlotTracker(args.workers)
    overall_bar = tqdm(total=args.n_trials, position=0, desc="optuna", dynamic_ncols=True)
    worker_bars = [
        tqdm(total=0, position=i + 1, bar_format="{desc}", dynamic_ncols=True)
        for i in range(args.workers)
    ]
    stop_ticker = threading.Event()
    ticker = threading.Thread(
        target=_tick_worker_bars, args=(tracker, worker_bars, stop_ticker), daemon=True
    )
    ticker.start()

    remaining = [args.n_trials]
    remaining_lock = threading.Lock()

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    worker_loop,
                    study,
                    remaining,
                    remaining_lock,
                    log_dir,
                    manifest_path,
                    threads_per_worker,
                    tracker,
                    overall_bar,
                )
                for _ in range(args.workers)
            ]
            for future in futures:
                future.result()
    finally:
        stop_ticker.set()
        ticker.join(timeout=5)
        for bar in worker_bars:
            bar.close()
        overall_bar.close()

    trials_df = study.trials_dataframe()
    trials_df.to_csv(sweep_dir / "trials.csv", index=False)

    best = study.best_trial
    print("\nSearch complete.")
    print(f"Best {OBJECTIVE_METRIC}: {best.value:.4f}")
    print(f"Best params: {best.params}")
    print(f"Trials CSV: {sweep_dir / 'trials.csv'}")


if __name__ == "__main__":
    main()

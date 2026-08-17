"""Orchestrates the GHCF ablation sweep: launches main.py train --cv per grid
config across a worker pool, with a resumable JSONL manifest."""

import argparse
import importlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

MODELS_DIR = Path(__file__).resolve().parents[2]
_manifest_lock = threading.Lock()
MIN_FREE_MIB = 2000  # don't launch a new job onto a GPU with less free than this


class SlotTracker:
    """Tracks which config each worker slot is currently running, for the
    live per-worker progress display."""

    def __init__(self, workers: int):
        self.free = list(range(workers))
        self.active: dict = {}  # slot -> (tag, start_time)
        self.lock = threading.Lock()

    def acquire(self, tag: str) -> int:
        with self.lock:
            slot = self.free.pop()
            self.active[slot] = (tag, time.time())
            return slot

    def release(self, slot: int) -> None:
        with self.lock:
            self.free.append(slot)
            del self.active[slot]

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.active)


def _tick_worker_bars(tracker: SlotTracker, bars: list, stop_event: threading.Event) -> None:
    """Background thread: refreshes each worker's bar with its current tag
    and running time (or 'idle') every couple seconds."""
    while not stop_event.is_set():
        snapshot = tracker.snapshot()
        for slot, bar in enumerate(bars):
            if slot in snapshot:
                tag, start = snapshot[slot]
                bar.set_description_str(f"[w{slot}] {tag} ({(time.time() - start) / 60:.1f}m)")
            else:
                bar.set_description_str(f"[w{slot}] idle")
            bar.refresh()
        stop_event.wait(2)


def _free_gpu_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return MIN_FREE_MIB  # nvidia-smi unavailable: don't block the sweep on it


def wait_for_gpu_memory(min_free_mib: int = MIN_FREE_MIB, poll_interval: int = 15) -> None:
    """Block until the GPU has enough free memory to safely start a new job.

    Prevents the exact failure mode observed in the first full-sweep attempt:
    launching new jobs onto an already-crowded GPU (whether crowded by our own
    workers or an unrelated tenant like a vLLM server) causes CUDA OOM crashes
    for some jobs and multi-hour allocator-thrashing stalls for the rest.
    """
    waited = 0
    while _free_gpu_mib() < min_free_mib:
        if waited == 0:
            tqdm.write(f"  ...waiting for >= {min_free_mib}MiB free GPU memory")
        time.sleep(poll_interval)
        waited += poll_interval


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
    with _manifest_lock, open(manifest_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def find_cv_dir(tag: str, root: str = "checkpoints") -> str:
    root_path = Path(root) if Path(root).is_absolute() else MODELS_DIR / root
    matches = sorted(root_path.glob(f"cv_{tag}_*"))
    return str(matches[-1]) if matches else ""


def run_one(
    config: dict,
    log_dir: Path,
    manifest_path: Path,
    threads_per_worker: int,
    tracker: SlotTracker,
    overall_bar: tqdm,
    ckpt_root: Path,
) -> None:
    tag = config["tag"]
    log_path = log_dir / f"{tag}.log"
    cmd = (
        ["python", "src/main.py", "train"]
        + config["cli_args"]
        + [
            "--checkpoint_root",
            str(ckpt_root),
        ]
    )

    wait_for_gpu_memory()

    # PyTorch defaults to an intra-op thread pool sized to the whole machine's
    # core count *per process*. Running N of these concurrently oversubscribes
    # the CPU by ~N x (e.g. 16 workers x 24 threads = 384 threads / 32 cores),
    # which thrashes far worse than the GPU-side contention alone -- this is
    # what turned a ~2x-worker / ~1x-slowdown ratio into a ~9x slowdown at 16
    # workers. Capping each subprocess to its fair share of cores fixes it.
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
                cmd,
                cwd=str(MODELS_DIR),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
        wall_time = time.time() - start
    finally:
        tracker.release(slot)

    record = {k: v for k, v in config.items() if k != "cli_args"}
    cv_dir = find_cv_dir(tag, root=str(ckpt_root))
    record.update({"wall_time_s": wall_time, "cv_dir": cv_dir})

    if proc.returncode != 0 or not cv_dir:
        record["status"] = "failed"
        record["mean_metrics"] = None
        record["std_metrics"] = None
        append_manifest(manifest_path, record)
        tqdm.write(f"[FAILED] {tag} (see {log_path})")
        overall_bar.update(1)
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
    tqdm.write(f"[{record['status'].upper()}] {tag} ({wall_time:.0f}s)")
    overall_bar.update(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep_id", type=str, default=None, help="Resume an existing sweep dir name"
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--limit", type=int, default=None, help="Only run the first N configs (testing)"
    )
    parser.add_argument(
        "--grid_module",
        type=str,
        default="grid",
        help="Module (in this dir) exposing build_grid(), e.g. full (default) or selective",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    build_grid = importlib.import_module("grid_" + args.grid_module).build_grid
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
    ckpt_root = sweep_dir / "ckpt"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    manifest_path = sweep_dir / "manifest.jsonl"

    completed = load_completed_tags(manifest_path)
    pending = [c for c in grid if c["tag"] not in completed]
    threads_per_worker = max(1, (os.cpu_count() or args.workers) // args.workers)
    print(f"Sweep dir: {sweep_dir}")
    print(f"Total: {len(grid)}  Already done: {len(completed)}  Pending: {len(pending)}")
    print(f"Workers: {args.workers}  Threads/worker: {threads_per_worker}")

    from concurrent.futures import ThreadPoolExecutor

    tracker = SlotTracker(args.workers)
    overall_bar = tqdm(
        total=len(grid), initial=len(completed), position=0, desc="sweep", dynamic_ncols=True
    )
    worker_bars = [
        tqdm(total=0, position=i + 1, bar_format="{desc}", dynamic_ncols=True)
        for i in range(args.workers)
    ]
    stop_ticker = threading.Event()
    ticker = threading.Thread(
        target=_tick_worker_bars, args=(tracker, worker_bars, stop_ticker), daemon=True
    )
    ticker.start()

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    run_one,
                    c,
                    log_dir,
                    manifest_path,
                    threads_per_worker,
                    tracker,
                    overall_bar,
                    ckpt_root,
                )
                for c in pending
            ]
            for future in futures:
                future.result()
    finally:
        stop_ticker.set()
        ticker.join(timeout=5)
        for bar in worker_bars:
            bar.close()
        overall_bar.close()

    print("Sweep complete.")


if __name__ == "__main__":
    main()

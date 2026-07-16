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

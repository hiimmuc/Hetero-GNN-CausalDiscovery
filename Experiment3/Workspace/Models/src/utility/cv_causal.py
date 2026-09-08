"""Per-fold causal discovery artifacts for cross-validation.

Design: docs/superpowers/specs/2026-09-04-per-fold-causal-discovery-cv-design.md

The offline ``run_causal_discovery.py`` fits one LiNGAM DAG on the whole
dataset. Under k-fold CV that DAG (and its z-score stats) has seen every
fold's held-out participants, which optimistically biases the ``--causal`` /
``--ace`` metrics. This module fits DirectLiNGAM + bootstrap once per fold on
that fold's *training* participants and caches the K artifact sets so CV /
sweep runs load the artifact matching the fold they are on.

Nothing here writes under ``causal_outputs/`` — the offline global artifacts
are left untouched.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from configs.config import (
    BASE_DIR,
    CAUSAL_CONFIG_PATH,
    CV_CAUSAL_CACHE_DIR,
    CV_CAUSAL_JOBS,
    CV_CAUSAL_LOCK_TIMEOUT,
    CV_CAUSAL_MIN_FOLD_ROWS,
    RANDOM_SEED,
    SET,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "lingam_causal_model"
ROBUST_NPY = f"{MODEL_NAME}_robust_model_data.npy"
NORM_STATS_CSV = f"{MODEL_NAME}_normalization_stats.csv"
MANIFEST_NAME = "folds_manifest.json"

# interaction type -> component feature columns. Mirror of
# utility.dataloader.eHMIDataset._get_interaction_columns for the z-scored
# targets (BFI_PERCEPTION uses fixed-scale MinMax and is never re-normalized).
RENORM_INTERACTION_COLUMNS = {
    "RATING": ["Rating score"],
    "VOICE_QUALITY": ["MOS-X2 Q1", "MOS-X2 Q2", "MOS-X2 Q3", "MOS-X2 Q4"],
    "SUBJECTIVE_FEELING": ["SQ1", "SQ2", "SQ3", "SQ4", "SQ5"],
}


# --------------------------------------------------------------------------- #
# cache key + paths
# --------------------------------------------------------------------------- #
def causal_fold_key(
    n_folds: int,
    stratified: bool,
    causal_config_path: str = CAUSAL_CONFIG_PATH,
    set_name: str = SET,
    seed: int = RANDOM_SEED,
) -> str:
    """Stable id for one (split params, causal config) combination.

    ``causal_config`` bytes already cover n_sampling, prior knowledge, feature
    lists and thresholds, so they need not be hashed separately.
    """
    cfg_bytes = Path(causal_config_path).read_bytes()
    h = hashlib.sha1()
    h.update(f"{set_name}|{int(n_folds)}|{bool(stratified)}|{int(seed)}|".encode())
    h.update(cfg_bytes)
    return h.hexdigest()[:16]


def fold_cache_dir(key: str) -> Path:
    root = Path(CV_CAUSAL_CACHE_DIR)
    if not root.is_absolute():
        root = BASE_DIR / root
    return root / key


def fold_dir(key: str, fold_idx: int) -> Path:
    """0-based fold_idx -> ``.../<key>/fold_{idx+1}/``."""
    return fold_cache_dir(key) / f"fold_{fold_idx + 1}"


def fold_artifact_npy(key: str, fold_idx: int) -> Path:
    return fold_dir(key, fold_idx) / "models" / ROBUST_NPY


def fold_norm_stats_csv(key: str, fold_idx: int) -> Path:
    return fold_dir(key, fold_idx) / "models" / NORM_STATS_CSV


# --------------------------------------------------------------------------- #
# lockfile (atomic O_CREAT|O_EXCL, steal when stale)
# --------------------------------------------------------------------------- #
class _FoldLock:
    def __init__(self, path: Path, timeout: int = CV_CAUSAL_LOCK_TIMEOUT):
        self.path = path
        self.timeout = timeout
        self._fd: int | None = None

    def __enter__(self) -> "_FoldLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                self._fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self._fd, f"{os.getpid()} {time.time()}\n".encode())
                return self
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age > self.timeout:
                    logger.warning("Stealing stale fold lock %s (age %.0fs)", self.path, age)
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                time.sleep(5)

    def __exit__(self, *exc) -> None:
        if self._fd is not None:
            os.close(self._fd)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


# --------------------------------------------------------------------------- #
# build one fold's artifact
# --------------------------------------------------------------------------- #
def build_fold_artifact(
    fold_idx: int,
    train_participant_ids: set,
    key: str,
    causal_config_path: str = CAUSAL_CONFIG_PATH,
    n_jobs: int = CV_CAUSAL_JOBS,
) -> dict:
    """Fit DirectLiNGAM + bootstrap for one fold and save it under the cache.

    Returns a manifest fold-record dict. Assumes the caller holds the fold lock.
    """
    from models.CausalModel import CausalModel  # local import: heavy deps

    out_dir = fold_dir(key, fold_idx)
    cm = CausalModel(causal_config_path)
    cm.config["data"]["output_dir"] = str(out_dir)
    cm.config["model"]["name"] = MODEL_NAME
    cm.config["visualization"]["save_graphs"] = False
    cm.config["model"].setdefault("bootstrap", {})["n_jobs"] = int(n_jobs)
    cm._setup_directories()
    cm.participant_ids = set(train_participant_ids)

    cm.load_data()
    n_rows = len(cm.data)
    if n_rows < CV_CAUSAL_MIN_FOLD_ROWS:
        logger.warning(
            "Fold %d: only %d training rows (< %d) after participant filter",
            fold_idx + 1,
            n_rows,
            CV_CAUSAL_MIN_FOLD_ROWS,
        )
    cm.fit_model()
    cm.run_bootstrap_analysis()
    cm.evaluate_model()
    cm.save_model()

    causal_order = [int(i) for i in cm.get_causal_order()]
    robust = getattr(cm, "robust_adjacency_matrix", None)
    n_edges = int(np.count_nonzero(robust)) if robust is not None else None
    return {
        "idx": fold_idx,
        "n_train_participants": len(set(train_participant_ids)),
        "n_train_rows": int(n_rows),
        "train_participant_ids": sorted(str(p) for p in train_participant_ids),
        "artifact_path": str(fold_artifact_npy(key, fold_idx)),
        "causal_order": causal_order,
        "n_edges": n_edges,
    }


def ensure_fold_artifact(
    fold_idx: int,
    train_participant_ids: set,
    key: str,
    causal_config_path: str = CAUSAL_CONFIG_PATH,
    n_jobs: int = CV_CAUSAL_JOBS,
) -> Path:
    """Return the fold's robust-model-data path, building it under a lock if absent."""
    npy = fold_artifact_npy(key, fold_idx)
    if npy.exists():
        return npy
    lock = fold_dir(key, fold_idx) / ".lock"
    with _FoldLock(lock):
        if npy.exists():  # another worker finished while we waited
            return npy
        logger.info("Building per-fold causal artifact: fold %d (key %s)", fold_idx + 1, key)
        build_fold_artifact(fold_idx, train_participant_ids, key, causal_config_path, n_jobs)
    if not npy.exists():
        raise RuntimeError(f"Fold {fold_idx + 1} build finished but {npy} is missing")
    return npy


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def manifest_path(key: str) -> Path:
    return fold_cache_dir(key) / MANIFEST_NAME


def write_folds_manifest(
    key: str,
    n_folds: int,
    stratified: bool,
    fold_records: list[dict],
    causal_config_path: str = CAUSAL_CONFIG_PATH,
) -> Path:
    cfg_sha = hashlib.sha1(Path(causal_config_path).read_bytes()).hexdigest()
    payload = {
        "key": key,
        "set": SET,
        "n_folds": int(n_folds),
        "stratified": bool(stratified),
        "seed": RANDOM_SEED,
        "causal_config": str(causal_config_path),
        "causal_config_sha1": cfg_sha,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "folds": sorted(fold_records, key=lambda r: r["idx"]),
    }
    p = manifest_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    return p


def read_folds_manifest(key: str) -> dict | None:
    p = manifest_path(key)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def assert_split_matches(manifest: dict, train_ids_by_fold: list[set]) -> None:
    """Raise if the CV run's own split disagrees with the cached manifest."""
    m_folds = {r["idx"]: set(map(str, r["train_participant_ids"])) for r in manifest["folds"]}
    for i, ids in enumerate(train_ids_by_fold):
        got = set(map(str, ids))
        want = m_folds.get(i)
        if want is None:
            raise RuntimeError(f"Cached causal manifest has no fold {i}")
        if got != want:
            raise RuntimeError(
                f"Fold {i + 1}: CV split disagrees with cached causal artifacts.\n"
                f"  cache : {sorted(want)}\n  run   : {sorted(got)}\n"
                f"Delete {fold_cache_dir(manifest['key'])} and re-run the precompute."
            )


# --------------------------------------------------------------------------- #
# fold-local interaction-target normalization
# --------------------------------------------------------------------------- #
def norm_lookup_from_csv(path: str | Path) -> dict[str, tuple[float, float]]:
    """{feature -> (mean, std)} from a causal normalization_stats.csv."""
    df = pd.read_csv(path)
    return {
        str(r["feature"]): (float(r["mean"]), float(r["std"])) for _, r in df.iterrows()
    }


def refold_normalize_interactions(
    interactions: dict[str, list | np.ndarray],
    global_lookup: dict[str, tuple[float, float]],
    fold_lookup: dict[str, tuple[float, float]],
) -> dict[str, list]:
    """Re-express globally z-scored RATING/VQ/SF targets in a fold's z-space.

        x_fold = (x_global * g_std + g_mean - f_mean) / f_std

    Rows are ``[user_idx, item_idx, v0, ...]``; only columns 2: are touched.
    Interaction types not in RENORM_INTERACTION_COLUMNS (e.g. BFI_PERCEPTION)
    pass through unchanged, as does any type whose columns are missing from
    either lookup (with a warning).

    Returns a dict of ``list`` (of 1-D np.ndarray rows), matching the container
    shape ``CrossValidationSplitter.split_interactions`` produces.
    """
    out: dict[str, list] = {}
    for itype, rows in interactions.items():
        arr = np.asarray(rows, dtype=float)
        cols = RENORM_INTERACTION_COLUMNS.get(itype)
        if cols is None or arr.size == 0:
            out[itype] = list(arr) if arr.size else []
            continue
        if not all(c in global_lookup and c in fold_lookup for c in cols):
            missing = [c for c in cols if c not in global_lookup or c not in fold_lookup]
            logger.warning(
                "refold_normalize: %s columns %s missing from stats; "
                "leaving on global normalization",
                itype,
                missing,
            )
            out[itype] = list(arr)
            continue
        g_mean = np.array([global_lookup[c][0] for c in cols])
        g_std = np.array([global_lookup[c][1] for c in cols])
        f_mean = np.array([fold_lookup[c][0] for c in cols])
        f_std = np.array([fold_lookup[c][1] for c in cols])
        adj = arr.copy()
        adj[:, 2:] = (arr[:, 2:] * g_std + g_mean - f_mean) / f_std
        out[itype] = list(adj)
    return out

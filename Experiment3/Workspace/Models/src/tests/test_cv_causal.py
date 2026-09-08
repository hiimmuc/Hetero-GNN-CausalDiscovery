"""Smoke tests for per-fold causal discovery in CV.

No pytest in this repo — run directly:

    cd Experiment3/Workspace/Models
    python src/tests/test_cv_causal.py

Covers the pure logic (fold-split parity, cache key, normalization round-trip,
manifest guard). The end-to-end path is exercised by actually running
`run_causal_discovery.py --cv` + `main.py train --cv --causal`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from configs.config import CAUSAL_CONFIG_PATH, DATABASE_PATH
from cross_validation import CrossValidationSplitter
from utility.cv_causal import (
    assert_split_matches,
    causal_fold_key,
    refold_normalize_interactions,
)


def test_split_participant_ids_matches_split_interactions():
    df = pd.read_excel(DATABASE_PATH)
    sp = CrossValidationSplitter(n_folds=5, stratified=True)

    folds_pid = sp.split_participant_ids(df)

    unique_users = list(df["Participant ID"].unique())
    u2i = {u: i for i, u in enumerate(unique_users)}
    i2u = {i: u for u, i in u2i.items()}
    rated = df[df["Rating score"].notna()]
    inter = {"RATING": [np.array([u2i[p], 0, 1.0]) for p in rated["Participant ID"]]}
    folds_int = sp.split_interactions(inter)
    folds_from_int = [{i2u[int(r[0])] for r in tr["RATING"]} for tr, _ in folds_int]

    assert folds_pid == folds_from_int, "split_participant_ids diverged from split_interactions"
    assert all(len(s) > 0 for s in folds_pid)


def test_cache_key_stability_and_sensitivity():
    k = lambda nf, st: causal_fold_key(nf, st, CAUSAL_CONFIG_PATH)
    assert k(5, True) == k(5, True)
    assert k(5, True) != k(3, True)
    assert k(5, True) != k(5, False)
    assert len(k(5, True)) == 16


def test_normalization_round_trip_and_identity():
    g = {"Rating score": (5.0, 2.0)}
    f = {"Rating score": (4.0, 3.0)}
    x_global = np.array([[0.0, 1.0, 0.5], [2.0, 3.0, -1.2]])

    out = np.array(refold_normalize_interactions({"RATING": list(x_global)}, g, f)["RATING"])
    raw = out[:, 2:] * 3.0 + 4.0
    raw_expect = x_global[:, 2:] * 2.0 + 5.0
    assert np.abs(raw - raw_expect).max() < 1e-9

    out_id = np.array(refold_normalize_interactions({"RATING": list(x_global)}, g, g)["RATING"])
    assert np.abs(out_id - x_global).max() < 1e-9

    # index columns stay exactly integral-valued
    assert np.array_equal(out[:, :2], x_global[:, :2])


def test_bfi_passthrough_and_missing_column_guard():
    g = {"Rating score": (5.0, 2.0)}
    f = {"Rating score": (4.0, 3.0)}
    bfi = np.array([[0.0, 1.0, 0.3, 0.4, 0.5, 0.6, 0.7]])
    out = refold_normalize_interactions({"BFI_PERCEPTION": list(bfi)}, g, f)["BFI_PERCEPTION"]
    assert np.allclose(np.array(out), bfi)

    # RATING but stats missing -> passthrough
    rating = np.array([[0.0, 1.0, 0.9]])
    out2 = refold_normalize_interactions({"RATING": list(rating)}, {}, {})["RATING"]
    assert np.allclose(np.array(out2), rating)


def test_assert_split_matches():
    manifest = {
        "key": "abc",
        "folds": [
            {"idx": 0, "train_participant_ids": ["p1", "p2"]},
            {"idx": 1, "train_participant_ids": ["p3"]},
        ],
    }
    assert_split_matches(manifest, [{"p1", "p2"}, {"p3"}])  # ok
    try:
        assert_split_matches(manifest, [{"p1"}, {"p3"}])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError on split mismatch")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")

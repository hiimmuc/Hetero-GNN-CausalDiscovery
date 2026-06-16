# `src/` Refactor Plan

Goal: remove dead code, eliminate duplication, and align the package with SOLID
principles without changing model behavior. Work is ordered so each phase is
independently shippable and low-risk. Phases 1–3 are mechanical and safe;
phases 4–5 touch design and need a test pass after each step.

The three feature changes already landed (parallel bootstrap, BFI/TIPI common-scale
differencing, GAT mode) are **not** part of this plan and are excluded from its diffs.

---

## Phase 1 — Dead code & demo blocks (safe, no behavior change)

Every module carries an `if __name__ == "__main__":` block (16 files). For
libraries (`models/*.py`, `utility/*.py`) these are ad-hoc smoke tests/demos that
drift from the real code and inflate the files.

- Move the demo/smoke-test blocks out of `models/HeteroGNN.py`, `models/GHCF.py`,
  `models/CausalHead.py`, `utility/helper.py`, `utility/dataloader.py`,
  `utility/metrics.py`, `utility/gradnorm.py`, `utility/batch_processing.py`
  into real tests under `tests/` (pytest). Entry-point `__main__` blocks in
  `main.py`, `run_cv.py`, `cross_validation.py`, `evaluate.py` stay.
- Delete commented-out dead lines (e.g. the old `torch.sparse.FloatTensor`
  comments in `HeteroGNN.py` and `helper.py`).
- `CausalModel.main()` duplicates `print_model_info()` /
  `print_system_of_equations()` calls already made inside
  `run_complete_analysis()`; collapse to a single call path.

## Phase 2 — Import hygiene (safe)

`sys.path.append(...)` appears 6 times and every model module has a
`try: from X import ... except ImportError: from models.X import ...` shim
(`CausalMixin`, `GHCF_Causal`, `GHCF`, `GHCF_Refine`, `batch_processing`,
`dataloader`).

- Make `src/` a proper package (add `__init__.py` where missing, run via
  `python -m`), then use absolute imports (`from models.X import ...`) everywhere
  and delete the `try/except` shims and all `sys.path.append` lines.
- Single source of truth: `configs/config.py` already centralizes constants;
  remove the hard-coded `INTERACTION_TYPES` fallback dict inside `GHCF.__init__`
  (lines ~96–104) and the duplicated column lists in `dataloader._get_interaction_columns`
  and `CausalHead.__main__`.

## Phase 3 — Normalization de-duplication (DRY)

Scaling logic is copy-pasted across three places with subtly different rules:
`utility/helper.normalize_features` (mixed strategy), the inline block in
`dataloader._process_interactions_for_split` (BFI MinMax / z-score), and the
new common-scale differencing in `models/CausalModel`.

- Extract a single `utility/scaling.py` with `minmax(x, lo, hi)` and
  `zscore(x)` plus a `FeatureScaler` that maps a column name → strategy using
  the scales already defined in `configs/config.py` (`BFI_SCALE`, `TIPI_SCALE`,
  `MOS_X2_SCALE`, `SQ_SCALE`). Have the dataloader, helper, and causal model all
  call it. This guarantees the GNN path and the causal/SEM path stay on the same
  scale definitionally, not by coincidence.

## Phase 4 — Split the `CausalModel` god-class (SRP)

`models/CausalModel.py` (~980 lines) mixes config loading, IO/dir setup, data
prep, fitting, bootstrap, visualization, evaluation, persistence, inference, and
pretty-printing in one class — and uses a `type("MockModel", ...)` hack
(two sites) to fake a fitted estimator on load.

- Split along responsibilities:
    - `causal/data.py` — load + preprocess + normalization stats.
    - `causal/estimator.py` — build/fit LiNGAM, bootstrap (the parallel code lives
    here), robust-effect calculation.
    - `causal/persistence.py` — save/load model data, replacing the `MockModel`
    hack with a small `FittedCausalModel` dataclass (`adjacency_matrix`,
    `causal_order`, `features`).
    - `causal/visualization.py` — graphviz graph + system-of-equations printing.
    - `CausalModel` becomes a thin orchestrator (`run_complete_analysis`).
- The `predict()` SEM propagation here and `SEMHead.forward` in
  `models/CausalHead.py` implement the **same** causal-order propagation twice;
  extract one `propagate_sem(adjacency, causal_order, inputs)` helper and have
  both call it (numpy vs torch variants can share the ordering logic).

## Phase 5 — Unify entry points & model construction (DRY / OCP)

There are four separate `argparse` setups (`utility/parser.py`,
`cross_validation.py`, `run_cv.py`, `evaluate.py`, `analyze_cv.py`) and model
construction is duplicated between `train.Trainer._create_model` and
`evaluate.py` (lines ~140 & ~166).

- Make `utility/parser.py` the only argument parser; have `run_cv.py` /
  `cross_validation.py` / `evaluate.py` reuse it (add subcommand/args rather than
  re-declaring). This also removes the `getattr(self.args, "gnn_type", ...)`
  fallback now needed because CV uses a different parser.
- Introduce a `build_model(cfg, args, data_dims)` factory used by both train and
  evaluate. `GHCF`, `GHCF_Causal`, `GHCF_Refine` selection becomes a registry
  (`{"base": GHCF, "causal": GHCF_Causal, "refine": GHCF_Refine}`) so adding a
  variant doesn't touch `if/else` chains (OCP). Note `GHCF_Causal` and
  `GHCF_Refine` already share the `_compute_bfi_tipi_difference` →
  `_predict_with_causal_model` flow via `CausalMixin`; fold the remaining
  duplicated `forward`/`predict_interaction` glue into the mixin.

---

## Suggested order & safety net

1. Add a minimal pytest harness first (move existing `__main__` smoke tests in).
2. Phases 1→2→3 (mechanical, run tests after each).
3. Phases 4→5 (design changes, run tests + one end-to-end train/eval after each).

No phase should change numerical outputs; the test harness from step 1 is what
guarantees that as the structure moves.

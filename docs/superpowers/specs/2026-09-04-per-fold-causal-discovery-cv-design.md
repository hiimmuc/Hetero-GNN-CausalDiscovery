# Per-fold causal discovery in cross-validation

**Date:** 2026-09-04
**Status:** approved design, pending implementation plan
**Scope:** `Experiment3/Workspace/Models/` — cross-validation path only

## Problem

The two-stage causal recommender fits a LiNGAM DAG once, offline, on the
**entire** `database_similarity.xlsx` (all 1019 rows). Its coefficients seed the
GNN's SEM head, and its z-score stats (`lingam_causal_model_normalization_stats.csv`)
also normalize the `RATING` / `VOICE_QUALITY` / `SUBJECTIVE_FEELING` interaction
targets.

`cross_validation.py` then splits **users** into K folds. Every fold's SEM head —
weights and normalization — has already seen the rows belonging to that fold's
held-out users. The `--causal` / `--ace` CV metrics are therefore optimistically
biased relative to a plain GHCF run whose adjacency, features and normalization
are all fold-local (the GNN message-passing graph is already rebuilt per fold via
`_build_fold_adj_matrices`; the causal component is the remaining leak).

## Goal

In CV, fit causal discovery separately per fold on that fold's **training**
participants only, and make both the SEM head and the interaction-target
normalization fully fold-local. No change to the single train/test split path
(`train.py`) or to the standalone offline pipeline (`run_causal_discovery.py`).

## Decisions (locked)

| Question | Decision |
|---|---|
| Normalization coupling | Fully fold-local: SEM head **and** interaction targets re-normalized with fold-train stats. |
| Run paths affected | CV only. |
| Bootstrap cost per fold | Keep `n_sampling: 5000` (same as offline). |
| Default vs opt-in | Default **on** for any `--cv` run with `--causal`/`--ace`; `--cv_global_causal` opts out. |
| Cache lock | Atomic `O_CREAT | O_EXCL` lockfile with stale-lock timeout. No new dependency. |

## Architecture

### Components and responsibilities

1. **`CausalModel.load_data()` — row filter seam (new, small).**
   New optional constructor/attribute `participant_ids: set | None`. When set,
   `load_data()` filters the dataframe to `df["Participant ID"].isin(participant_ids)`
   immediately after `pd.read_excel`, before feature selection and normalization.
   Everything downstream (rescale, z-score, fit, bootstrap, save) is unchanged.
   This is the only change to `CausalModel`.

2. **`FoldCausalArtifacts` — per-fold discovery + cache (new, in `cross_validation.py`).**
   Given a fold's training participant-ID set and the run's `--causal_config`,
   returns the path to a `*_robust_model_data.npy` fitted on those rows.
   - Cache dir: `causal_outputs/cv_folds/<key>/`
   - `key = sha1(b"\n".join(sorted(str(pid) for pid in train_ids))
             + causal_config_file_bytes
             + str(n_sampling) + SET).hexdigest()[:16]`
   - Hit: `<key>/models/lingam_causal_model_robust_model_data.npy` exists →
     return it.
   - Miss: acquire `<key>/.lock` (atomic create; if held and younger than
     `CV_CAUSAL_LOCK_TIMEOUT` seconds, poll; if stale, steal). Re-check for the
     artifact after acquiring (another worker may have finished). Then build:
     ```
     cm = CausalModel(causal_config_path)
     cm.config["data"]["output_dir"] = str(cache_dir)
     cm.config["model"]["name"] = "lingam_causal_model"
     cm.config["visualization"]["save_graphs"] = False
     cm.config["model"]["bootstrap"]["n_jobs"] = cv_causal_jobs   # default 1
     cm._setup_directories()
     cm.participant_ids = train_ids
     cm.run_complete_analysis()
     ```
     Release lock.
   - `n_jobs` defaults to 1 in CV to avoid nested joblib oversubscription when
     several sweep workers each run a CV; overridable via `--cv_causal_jobs`.

3. **Fold-local interaction normalization (new, in `cross_validation.py`).**
   The CV dataset is still built once with the **global** stats (current
   behavior) — this gives a known, invertible affine transform. `run()` keeps the
   loaded global stats lookup `{col: (g_mean, g_std)}` on `self`.
   In `run_fold`, before constructing `fold_train_data` / `fold_val_data`, for
   each interaction type in `{RATING, VOICE_QUALITY, SUBJECTIVE_FEELING}` and each
   of its component columns:
   ```
   x_fold = (x_global * g_std + g_mean - f_mean) / f_std
   ```
   applied to columns `2:` of both the train and val interaction arrays. `f_*`
   from the fold artifact's sibling `lingam_causal_model_normalization_stats.csv`.
   `BFI_PERCEPTION` (fixed-scale MinMax, data-independent) is left untouched.
   Guard: if any column is absent from the global lookup **or** the fold lookup,
   skip that interaction type (leave on global normalization) and log a warning —
   the round-trip can't be done safely.

4. **Wiring into `FoldTrainer` (no change to `FoldTrainer` / `CausalMixin`).**
   `run_fold` builds `fold_args = copy.copy(self.args)` and sets
   `fold_args.causal_path = <cache_dir>/models/lingam_causal_model_robust_model_data.npy`,
   then `FoldTrainer(fold_args, ...)`. `Trainer._setup_causal_paths()` reads
   `self.args.causal_path`; `CausalMixin` resolves the sibling
   `_normalization_stats.csv` for the SEM head automatically.

5. **`--cv_global_causal` escape hatch.**
   When set, skip components 2–4 entirely; behavior is exactly today's (all folds
   share `args.causal_path`). Used for A/B comparison and regression testing.

### Data flow (one CV run, per-fold causal ON)

```
load_data()  ── builds dataset once, global stats, keeps self._global_norm_lookup
    │
split_interactions()  ── K (train_interactions, val_interactions) by user
    │
for each fold:
    train_ids  = { idx_to_user[int(r[0])] for r in train_interactions["RATING"] }
    art_path   = FoldCausalArtifacts.get(train_ids, causal_config, n_sampling)   # cache or build
    f_lookup   = read <art dir>/…_normalization_stats.csv
    train/val interactions ── renormalize RATING/VQ/SF columns via (g_*, f_*)
    fold_args  = copy(args); fold_args.causal_path = art_path
    FoldTrainer(fold_args, …).train()
```

### Metadata addition

`CrossValidationTrainer.load_data()` adds `idx_to_user` to `metadata` (from
`dataset.idx_to_user`, or inverted from `user_to_idx` on the cached-load path).

### New config / CLI

`configs/config.py`:
- `CV_CAUSAL_JOBS = 1`
- `CV_CAUSAL_LOCK_TIMEOUT = 1800` (seconds; stale-lock steal threshold)
- `CV_CAUSAL_MIN_FOLD_ROWS = 100` (warn below this)

Per-fold discovery is on whenever `--cv` is combined with `--causal`/`--ace`;
`--cv_global_causal` (`store_true`, default off) is the only switch.

`utility/parser.py` (`add_cv_arguments`):
- `--cv_global_causal` (`store_true`) — force single shared causal artifact
- `--cv_causal_jobs` (`int`, default `CV_CAUSAL_JOBS`) — joblib workers for
  per-fold bootstrap

### Outputs / bookkeeping

- `cv_config.json` gains `per_fold_causal: bool` and `fold_causal_keys: [<key>, …]`.
- Each `fold_N/` gets `causal_adjacency.csv` (copy of the fold's robust adjacency)
  for traceability.
- Cache dirs under `causal_outputs/cv_folds/` persist across runs (git-ignored).

## Cost

Per unique `(fold train-ID set, causal_config, n_sampling)`: one full
discovery (fit + 5000-sample bootstrap + evaluate + save; viz skipped).
K folds ⇒ K discoveries per CV run at worst. A recommender sweep with a fixed
causal config and fixed CV args reuses the cache across every trial ⇒ K
discoveries total for the whole sweep. A causal-config sweep pays
K × (unique causal configs).

## Testing

**Unit**
- Cache key: stable across dict ordering of `train_ids`; changes when config
  bytes / `n_sampling` / SET change.
- Normalization round-trip: `g→raw→fold→` inverse recovers raw within 1e-9;
  `f_* == g_*` ⇒ identity.
- Fold-ID extraction: IDs derived from `train_interactions` equal the splitter's
  `train_users` mapped through `idx_to_user`.
- Guard path: a stats CSV missing a column ⇒ that interaction type is left on
  global normalization and a warning is logged.

**Integration**
- 2-fold `--cv --causal` on `SET=25`: each `fold_N/causal_adjacency.csv` differs;
  run completes; `cv_results.json` written.
- Concurrent: two processes racing the same cold cache key ⇒ exactly one builds,
  both return the same path, no corrupt artifact.

**Regression**
- `--cv --causal --cv_global_causal` reproduces the pre-change CV metrics on a
  fixed seed (byte-identical `cv_summary.csv` within float tolerance).

## Out of scope

- `train.py` single-split path.
- Per-fold re-discovery of user/item **features** (already fold-independent by
  design — features come from questionnaire data, not interactions).
- Changing `n_sampling` or the bootstrap/robustness logic.
- Any change to `run_causal_discovery.py` or the offline artifact layout.

## Risks

- **Nested parallelism under sweeps.** Mitigated by `--cv_causal_jobs` default 1
  and the cache (discovery runs at most K times per sweep).
- **Stale lock after a crash.** Mitigated by the `CV_CAUSAL_LOCK_TIMEOUT` steal.
- **Tiny fold train sets.** ~24 users × ~34 items ≈ 800 rows for 5-fold; LiNGAM on
  15 vars is fine. Add a warning if a fold's row count drops below a threshold
  (e.g. 100).
- **Global artifact dependency.** The inversion needs the global stats CSV, i.e.
  the offline `run_causal_discovery.py` must have been run once. This is already a
  documented prerequisite of the pipeline; the guard degrades to global
  normalization if stats are missing.

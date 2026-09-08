# Per-fold causal discovery in cross-validation

**Date:** 2026-09-04 (rev. 2026-09-08 — explicit precompute step)
**Status:** implemented in `8fda238` (commit also fixes a pre-existing int64
truncation of normalized interaction targets in the dataloader)
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

Split the dataset into K folds **first**, run DirectLiNGAM + bootstrap once per
fold on that fold's **training** participants (→ K artifact sets
`A_train_fold_1 … A_train_fold_K`, separate files), then have every CV / sweep
training run consume the artifact matching the fold it is currently on. Both the
SEM head and the interaction-target normalization become fully fold-local. The
K discoveries are a one-time precompute; the sweep afterwards does no causal
fitting at all.

## Decisions (locked)

| Question | Decision |
|---|---|
| Structure | **Explicit precompute:** `run_causal_discovery.py --cv` fits all K folds up front and writes K artifact sets; CV/sweep runs only read them. |
| Normalization coupling | Fully fold-local: SEM head **and** interaction targets re-normalized with fold-train stats. |
| Run paths affected | CV only. `train.py` single-split unchanged. |
| Bootstrap cost | `n_sampling: 5000` per fold, same as offline. Exactly K fits, once. |
| Default vs opt-in | Per-fold artifacts are used automatically by any `--cv` run with `--causal`/`--ace`; `--cv_global_causal` opts out. |
| Missing artifacts | CV run auto-builds the missing fold (atomic `O_CREAT|O_EXCL` lockfile, no dep) and logs the recommended precompute command. `run_sweep.py` / `run_optuna.py` run the precompute once before fanning out, so the fallback rarely triggers. |
| Fold-split source of truth | One shared function (`CrossValidationSplitter.split_participant_ids(df)`) called by both the precompute entrypoint and `cross_validation.py`. |

## Architecture

### Components and responsibilities

1. **`CausalModel.load_data()` — row filter seam (new, small).**
   New optional constructor/attribute `participant_ids: set | None`. When set,
   `load_data()` filters the dataframe to `df["Participant ID"].isin(participant_ids)`
   immediately after `pd.read_excel`, before feature selection and normalization.
   Everything downstream (rescale, z-score, fit, bootstrap, save) is unchanged.
   This is the only change to `CausalModel`.

2. **Shared fold split (new, in `cross_validation.py`).**
   Extract the KFold/StratifiedKFold call out of `split_interactions` into a
   private helper `_fold_index_splits(users_array, activity_values | None)` →
   `list[(train_idx, val_idx)]` (seed `RANDOM_SEED`, `np.digitize` on activity
   terciles when stratified). `split_interactions` keeps its exact current
   behavior by delegating to it.
   New public `CrossValidationSplitter.split_participant_ids(df)`:
   - builds `user_to_idx` from `df["Participant ID"].unique()` (same
     first-appearance order the dataloader uses), `users = list(set(indices))`,
     activity = per-user `RATING`-row count — **identical construction to
     `split_interactions`**;
   - calls the same `_fold_index_splits` helper;
   - returns `list[set[participant_id]]` (K **training**-ID sets) by mapping
     indices back through `idx_to_user`.
   A unit test asserts `split_participant_ids(df)` and `split_interactions` agree
   fold-for-fold. This keeps the fold boundaries **identical to
   `sweep_20260904_173056`** (no re-partitioning), so per-fold results stay
   comparable to that sweep.

3. **`run_causal_discovery.py --cv` — the precompute entrypoint (new mode).**
   New flags: `--cv`, `--n_folds` (default `CV_CONFIG["n_folds"]`), `--stratified`
   / `--no-stratified`, `--causal_jobs` (default `CV_CAUSAL_JOBS`).
   Steps:
   1. Read `DATABASE_PATH` (`database.xlsx`); `split_participant_ids(df)` → K
      training-ID sets.
   2. Compute `key` (below). Target dir `causal_fold_cache/<key>/`.
   3. For each fold `i` (skip if `fold_i/models/…_robust_model_data.npy` already
      exists):
      ```
      cm = CausalModel(causal_config_path)
      cm.config["data"]["output_dir"]              = "<cache>/<key>/fold_i/"
      cm.config["model"]["name"]                   = "lingam_causal_model"
      cm.config["visualization"]["save_graphs"]    = False
      cm.config["model"]["bootstrap"]["n_jobs"]    = causal_jobs
      cm._setup_directories()
      cm.participant_ids = train_ids_i     # component 1 seam
      cm.run_complete_analysis()
      ```
      Warn if a fold's filtered row count < `CV_CAUSAL_MIN_FOLD_ROWS`.
   4. Write `<cache>/<key>/folds_manifest.json`:
      `{key, set, n_folds, stratified, seed, n_sampling, causal_config_sha,
        folds: [{idx, n_train_participants, train_participant_ids, artifact_path,
                 causal_order, n_edges}]}`.
   The standalone `run_causal_discovery.py` (no `--cv`) is byte-for-byte
   unchanged — it still writes the global artifact to `causal_outputs/`.

   **Cache key & location.** Location `checkpoints/causal_fold_cache/<key>/`
   (fixed path, *not* `causal_outputs/`, *not* per-sweep — so one precompute
   serves every later CV run and every sweep).
   `key = sha1(SET + "|" + str(n_folds) + "|" + str(stratified) + "|"
              + str(RANDOM_SEED) + "|" + causal_config_file_bytes).hexdigest()[:16]`.
   `causal_config_file_bytes` already covers `n_sampling`, prior knowledge,
   feature lists, thresholds, etc. `folds_manifest.json` records the actual
   per-fold ID lists; the CV run asserts its own split matches them.

4. **Sweep launchers run the precompute once (`run_sweep.py`, `run_optuna.py`).**
   Before spawning workers: if the grid/search space contains any `--causal` /
   `--ace` config, invoke the precompute for `(n_folds, stratified)` taken from
   `CV_COMMON_ARGS` unless `--skip_causal_precompute` is passed. One serial
   K-fold discovery up front; the 8 workers then never touch causal fitting and
   never contend on a lock.

5. **CV-run consumption (`cross_validation.py`).**
   `CrossValidationTrainer.run()`:
   - `load_data()` adds `idx_to_user` to `metadata` (from `dataset.idx_to_user`,
     or inverted from the cached `user_to_idx`).
   - After `split_interactions`, derive each fold's training participant IDs
     directly from the split it just produced:
     `train_ids_i = { idx_to_user[int(r[0])] for r in train_interactions_i["RATING"] }`.
   - Compute `key`; `cache = checkpoints/causal_fold_cache/<key>/`.
   - If `cache/folds_manifest.json` exists, assert `train_ids_i` equals the
     manifest's `folds[i].train_participant_ids` (guards against a drifted
     splitter or a stale cache). Mismatch → hard error naming both sets.
   - Per fold `i`, resolve `art_i = cache/fold_i/models/lingam_causal_model_robust_model_data.npy`.
     If missing → build that one fold in place (`CausalModel(participant_ids=train_ids_i)`)
     under `cache/fold_i/.lock` (atomic `O_CREAT|O_EXCL`, stale-steal after
     `CV_CAUSAL_LOCK_TIMEOUT`), then log the full
     `run_causal_discovery.py --cv --n_folds K …` command that would precompute
     all folds at once.
   - Copy `fold_i` artifact set into `<cv_dir>/fold_{i+1}/causal/` (self-contained
     checkpoint) and set `fold_args.causal_path` to that copy.

6. **Fold-local interaction normalization (`cross_validation.py`).**
   The CV dataset is still built once with the **global** stats — an invertible
   affine transform. `run()` keeps `self._global_norm_lookup = {col: (g_mean, g_std)}`.
   In `run_fold`, before building `fold_train_data` / `fold_val_data`, for each
   interaction type in `{RATING, VOICE_QUALITY, SUBJECTIVE_FEELING}` and each
   component column:
   ```
   x_fold = (x_global * g_std + g_mean - f_mean) / f_std
   ```
   on columns `2:` of both the train and val arrays. `f_*` from
   `fold_{i+1}/causal/lingam_causal_model_normalization_stats.csv`.
   `BFI_PERCEPTION` (fixed-scale MinMax) is untouched. Guard: any column absent
   from the global **or** fold lookup ⇒ that interaction type stays on global
   normalization + warning.

7. **Wiring into `FoldTrainer` (no change to `FoldTrainer` / `CausalMixin`).**
   `fold_args = copy.copy(self.args)`;
   `fold_args.causal_path = <cv_dir>/fold_{i+1}/causal/lingam_causal_model_robust_model_data.npy`;
   `FoldTrainer(fold_args, …)`. `Trainer._setup_causal_paths()` reads it;
   `CausalMixin` resolves the sibling `_normalization_stats.csv` automatically.

8. **`--cv_global_causal` escape hatch.**
   When set, skip components 5–7; all folds share `args.causal_path` (today's
   behavior). For A/B comparison and regression testing.

### Data flow

**Precompute (once):**
```
database.xlsx ── split_participant_ids(df) ── [ids_1, …, ids_K]
    │
for i in 1..K:
    CausalModel(participant_ids=ids_i).run_complete_analysis()
        → checkpoints/causal_fold_cache/<key>/fold_i/{models/…_robust_model_data.npy,
                                                      …_normalization_stats.csv}
    │
    → checkpoints/causal_fold_cache/<key>/folds_manifest.json
```

**Each CV run / sweep trial:**
```
load_data() ── dataset built once, global stats → self._global_norm_lookup
split_interactions() ── K (train, val) by user;  key → cache dir; assert manifest split matches
for each fold i:
    art_i    = cache/fold_i/…_robust_model_data.npy      (build-in-place only if missing)
    copy     → cv_dir/fold_{i+1}/causal/
    f_lookup = read cv_dir/fold_{i+1}/causal/…_normalization_stats.csv
    train/val interactions ── renormalize RATING/VQ/SF via (g_*, f_*)
    fold_args = copy(args); fold_args.causal_path = cv_dir/fold_{i+1}/causal/…_robust_model_data.npy
    FoldTrainer(fold_args, …).train()   ── SEM head built from fold i's DAG
```

### New config / CLI

`configs/config.py`:
- `CV_CAUSAL_JOBS = 1` — joblib workers for the per-fold bootstrap
- `CV_CAUSAL_LOCK_TIMEOUT = 1800` (seconds; stale-lock steal threshold)
- `CV_CAUSAL_MIN_FOLD_ROWS = 100` (warn below this)
- `CV_CAUSAL_CACHE_DIR = "checkpoints/causal_fold_cache"` (relative to Models/)

`run_causal_discovery.py` argparse:
- `--cv` — precompute mode
- `--n_folds` (int, default `CV_CONFIG["n_folds"]`)
- `--stratified` / `--no-stratified` (default `CV_CONFIG["stratified"]`)
- `--causal_jobs` (int, default `CV_CAUSAL_JOBS`)

`utility/parser.py` (`add_cv_arguments`):
- `--cv_global_causal` (`store_true`, default off) — force the single shared
  `--causal_path` artifact for all folds
- `--cv_causal_jobs` (int, default `CV_CAUSAL_JOBS`) — passed through when a CV
  run has to build a missing fold in place

`scripts/sweep/run_sweep.py`, `run_optuna.py`:
- `--skip_causal_precompute` (`store_true`) — don't run the up-front precompute

### Outputs / bookkeeping

- `checkpoints/causal_fold_cache/<key>/fold_i/` — the K artifact sets (robust
  model data + normalization stats; `save_graphs` off).
- `checkpoints/causal_fold_cache/<key>/folds_manifest.json` — key, split
  params, and per-fold `{train_participant_ids, artifact_path, causal_order,
  n_edges}`.
- Each CV run copies its folds' artifact sets into `<cv_dir>/fold_{i}/causal/`
  (self-contained checkpoint) and records `per_fold_causal: true` + the resolved
  `causal_fold_key` in `cv_config.json`.
- `causal_outputs/` is never written by any of this. `checkpoints/` is already
  git-ignored (`Experiment3/Workspace/Models/checkpoints/`).

## Cost

The precompute is **exactly K DirectLiNGAM + 5000-sample bootstrap fits**, run
once, serially (`CV_CAUSAL_JOBS=1` by default to stay friendly under an
already-parallel sweep; raise `--causal_jobs` for a standalone precompute). At
K=5 that is ~5× the current single offline discovery — tens of minutes, one time.
Every subsequent CV run and every one of the sweep's 99 trials then does **zero**
causal fitting: it reads `fold_i/…_robust_model_data.npy` and copies it
(sub-second). Re-running the precompute is a no-op once `fold_i/` exists (per-fold
skip). A different `causal_config` / `n_folds` / `n_sampling` / seed / SET yields
a new `<key>` and its own K fits.

## Testing

**Unit**
- `split_participant_ids(df)`: deterministic across calls/processes; the K
  training sets are disjoint in their *validation* complements and cover all
  participants; matches `split_interactions`' user split mapped through
  `idx_to_user`.
- Cache key: unchanged for the same split params; changes on any of SET,
  n_folds, stratified, seed, n_sampling, causal_config bytes.
- Normalization round-trip: `g → raw → fold` recovers raw within 1e-9;
  `f_* == g_*` ⇒ identity transform.
- Manifest guard: a `folds_manifest.json` whose `train_participant_ids` disagree
  with a freshly computed split ⇒ hard error.
- Column guard: a fold stats CSV missing a column ⇒ that interaction type stays
  on global normalization + warning.

**Integration**
- `run_causal_discovery.py --cv --n_folds 2` on `SET=25`: writes
  `causal_fold_cache/<key>/fold_{1,2}/…` + manifest; the two robust adjacency
  matrices differ; `causal_outputs/` mtime unchanged.
- 2-fold `--cv --causal` on `SET=25` after that precompute: each fold's SEM head
  loads its own `fold_i` artifact (assert `causal_path` per fold); run completes;
  `cv_config.json` has `per_fold_causal: true`.
- Missing-fold fallback: delete `fold_2/`, run CV, assert it rebuilds exactly
  that fold under a lock and finishes.

**Regression**
- `--cv --causal --cv_global_causal` reproduces the pre-change CV metrics on a
  fixed seed (`cv_summary.csv` within float tolerance).

## Invariants

- `causal_outputs/` is read-only for the CV/sweep path. Per-fold artifacts live
  only under `checkpoints/causal_fold_cache/` and `<cv_dir>/fold_*/causal/`. The
  offline global artifact is used solely as the source of the *global*
  normalization stats for the inversion (component 6).
- The fold split is computed by one function; the precompute and the CV run must
  produce identical folds or the CV run aborts.
- `run_causal_discovery.py` without `--cv` is behaviourally unchanged.

## Out of scope

- `train.py` single-split path.
- Per-fold re-discovery of user/item **features** (fold-independent by design —
  they come from questionnaire data, not interactions).
- Changing `n_sampling` or the bootstrap/robustness logic.
- The offline global artifact layout under `causal_outputs/`.

## Risks

- **Splitter drift.** If `split_interactions` and `split_participant_ids` diverge,
  folds mismatch. Mitigated by making the latter the single implementation and by
  the manifest assertion in the CV run.
- **Precompute skipped, sweep cold.** Falls back to per-fold build-in-place under
  a lock — correct but serialises the first trial's folds across workers.
  `run_sweep.py` runs the precompute up front to avoid this.
- **Stale lock after a crash.** `CV_CAUSAL_LOCK_TIMEOUT` steal.
- **Tiny fold train sets.** ~24 participants for 5-fold ⇒ ~800 rows; LiNGAM on 15
  vars is fine. Warn below `CV_CAUSAL_MIN_FOLD_ROWS`.
- **Global stats dependency for the inversion.** Needs the offline global
  `_normalization_stats.csv`. Already a pipeline prerequisite; the column guard
  degrades to global normalization per-interaction-type if absent.

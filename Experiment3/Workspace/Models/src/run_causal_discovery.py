"""
Entry point to run the LiNGAM causal discovery pipeline.

Run from the `Models/` project root:

    # Global model (fitted on the whole dataset) -> causal_outputs/
    python src/run_causal_discovery.py
    python src/run_causal_discovery.py --config src/configs/causal_config.yaml

    # Per-fold models for k-fold CV: fits DirectLiNGAM + bootstrap once per fold
    # on that fold's TRAINING participants, into
    # checkpoints/causal_fold_cache/<key>/fold_*/ . Consumed automatically by
    # `python src/main.py train --cv --causal` (unless --cv_global_causal).
    python src/run_causal_discovery.py --cv --n_folds 5 --stratified
"""

import argparse
import logging

import pandas as pd
from configs.config import CV_CAUSAL_JOBS, CV_CONFIG, DATABASE_PATH
from models.CausalModel import CausalModel

logger = logging.getLogger(__name__)


def _run_global(config_path: str) -> None:
    """Original behaviour: one model fitted on the entire dataset."""
    causal_model = CausalModel(config_path)
    causal_model.run_complete_analysis()
    causal_model.print_model_info()

    if causal_model.config["inference"]["enable_inference"]:
        example_input = {
            "Difference BFI-TIPI O": 0.5,
            "Difference BFI-TIPI C": -0.2,
            "Difference BFI-TIPI E": 0.1,
            "Difference BFI-TIPI A": 0.3,
            "Difference BFI-TIPI N": -0.1,
        }
        predictions = causal_model.predict(example_input)
        logger.info(f"Example prediction: {predictions}")


def _run_cv(config_path: str, n_folds: int, stratified: bool, causal_jobs: int) -> None:
    """Precompute one causal artifact per CV fold (training participants only)."""
    from cross_validation import CrossValidationSplitter
    from utility.cv_causal import (
        build_fold_artifact,
        causal_fold_key,
        fold_artifact_npy,
        fold_cache_dir,
        write_folds_manifest,
    )

    df = pd.read_excel(DATABASE_PATH)
    splitter = CrossValidationSplitter(n_folds=n_folds, stratified=stratified)
    train_ids_by_fold = splitter.split_participant_ids(df)

    key = causal_fold_key(n_folds, stratified, config_path)
    cache = fold_cache_dir(key)
    logger.info("Per-fold causal discovery: key=%s -> %s", key, cache)

    records = []
    for i, train_ids in enumerate(train_ids_by_fold):
        npy = fold_artifact_npy(key, i)
        if npy.exists():
            logger.info("fold %d/%d: artifact exists, skipping (%s)", i + 1, n_folds, npy)
            # still record it in the manifest
            records.append(
                {
                    "idx": i,
                    "n_train_participants": len(train_ids),
                    "n_train_rows": None,
                    "train_participant_ids": sorted(str(p) for p in train_ids),
                    "artifact_path": str(npy),
                    "causal_order": None,
                    "n_edges": None,
                }
            )
            continue
        logger.info(
            "fold %d/%d: fitting DirectLiNGAM + bootstrap on %d training participants",
            i + 1,
            n_folds,
            len(train_ids),
        )
        records.append(
            build_fold_artifact(i, train_ids, key, config_path, n_jobs=causal_jobs)
        )

    manifest = write_folds_manifest(key, n_folds, stratified, records, config_path)
    logger.info("Wrote %s", manifest)
    print(f"\nPer-fold causal artifacts ready under: {cache}")
    print("Run the sweep / CV as usual — `--cv --causal` will pick them up automatically.")


def main() -> None:
    """Load data, fit the LiNGAM model(s), run bootstrap analysis, and save outputs."""
    parser = argparse.ArgumentParser(description="Run LiNGAM causal discovery")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/causal_config.yaml",
        help="Path to the causal model YAML config",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Precompute one model per CV fold (training participants only)",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=CV_CONFIG["n_folds"],
        help="Number of CV folds (--cv mode)",
    )
    parser.add_argument(
        "--stratified",
        action=argparse.BooleanOptionalAction,
        default=CV_CONFIG["stratified"],
        help="Stratify the fold split by participant activity (--cv mode)",
    )
    parser.add_argument(
        "--causal_jobs",
        type=int,
        default=CV_CAUSAL_JOBS,
        help="joblib workers for each fold's bootstrap (--cv mode)",
    )
    args = parser.parse_args()

    if args.cv:
        _run_cv(args.config, args.n_folds, args.stratified, args.causal_jobs)
    else:
        _run_global(args.config)


if __name__ == "__main__":
    main()

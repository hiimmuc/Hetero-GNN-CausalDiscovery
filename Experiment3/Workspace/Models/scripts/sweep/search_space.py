"""Optuna search space for the GHCF ablation search.

Mirrors grid.py's axes (interaction_set x causal_mode x bound_bfi x
freeze_sem x loss x adj_type x gnn_type) as conditional categorical
suggestions, plus continuous hyperparameters that grid.py held fixed
(lr, weight_decay, dropout, embed_size) and a continuous ACE mix weight
in place of grid.py's three fixed splits.
"""

import optuna

from grid import CV_COMMON_ARGS, INTERACTION_SETS

LOSSES = ["mse", "hybrid"]
ADJ_TYPES = ["pre", "norm", "mean", "plain"]
GNN_TYPES = ["gcn", "sage", "gat"]
CAUSAL_MODES = ["none", "causal", "ace_learnable", "ace_fixed"]
EMBED_SIZES = [16, 32, 64, 128]


def _causal_cli_args(trial: optuna.Trial, causal_mode: str) -> list[str]:
    if causal_mode == "none":
        return []
    if causal_mode == "causal":
        args = ["--causal"]
    elif causal_mode == "ace_learnable":
        args = ["--ace", "--ace_weight_mode", "learnable"]
    else:
        w_causal = trial.suggest_float("ace_w_causal", 0.0, 1.0)
        args = [
            "--ace",
            "--ace_weight_mode",
            "fixed",
            "--ace_w_mlp",
            str(1.0 - w_causal),
            "--ace_w_causal",
            str(w_causal),
        ]
    freeze_sem = trial.suggest_categorical("freeze_sem", [True, False])
    args.append("--freeze_sem" if freeze_sem else "--no-freeze_sem")
    return args


def suggest_params(trial: optuna.Trial) -> dict:
    """Sample one trial's config, returning both the config dict (for the
    manifest/dataframe) and the cli_args to pass to `main.py train --cv`."""
    interaction_set = trial.suggest_categorical("interaction_set", list(INTERACTION_SETS))
    interactions = INTERACTION_SETS[interaction_set]
    has_bfi = "BFI_PERCEPTION" in interactions

    causal_mode = trial.suggest_categorical("causal_mode", CAUSAL_MODES) if has_bfi else "none"
    bound_bfi = trial.suggest_categorical("bound_bfi", [True, False]) if has_bfi else None
    causal_cli_args = _causal_cli_args(trial, causal_mode) if has_bfi else []

    loss = trial.suggest_categorical("loss", LOSSES)
    adj_type = trial.suggest_categorical("adj_type", ADJ_TYPES)
    gnn_type = trial.suggest_categorical("gnn_type", GNN_TYPES)

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.6)
    embed_size = trial.suggest_categorical("embed_size", EMBED_SIZES)

    cli_args = (
        ["--interactions", *interactions]
        + CV_COMMON_ARGS
        + [
            "--loss",
            loss,
            "--adj_type",
            adj_type,
            "--gnn_type",
            gnn_type,
            "--lr",
            str(lr),
            "--decay",
            str(weight_decay),
            "--dropout",
            str(dropout),
            "--embed_size",
            str(embed_size),
        ]
        + causal_cli_args
        + ["--run_tag", f"trial{trial.number}"]
    )
    if bound_bfi is not None:
        cli_args.append("--bound_bfi" if bound_bfi else "--no-bound_bfi")

    config = {
        "interaction_set": interaction_set,
        "causal_mode": causal_mode,
        "bound_bfi": bound_bfi,
        "loss": loss,
        "adj_type": adj_type,
        "gnn_type": gnn_type,
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "embed_size": embed_size,
    }
    return {"config": config, "cli_args": cli_args}

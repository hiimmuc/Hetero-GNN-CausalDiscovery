"""Generates the full pruned-factorial ablation grid for the GHCF sweep.

Axes: interaction_set x causal_mode x bound_bfi x freeze_sem x loss x adj_type
x gnn_type. bound_bfi/freeze_sem collapse to a single "n/a" value where the
underlying flag has no effect (see the plan doc for the pruning rationale).
"""

from typing import Dict, List, Optional

INTERACTION_SETS = {
    "default": ["RATING", "BFI_PERCEPTION", "VOICE_QUALITY", "SUBJECTIVE_FEELING"],
    "single": ["RATING"],
    "multiple": ["RATING", "BFI_PERCEPTION"],
}

CAUSAL_MODES = [
    "none",
    "causal",
    "ace_learnable",
    "ace_fixed_5050",
    "ace_fixed_7030",
    "ace_fixed_3070",
]

ACE_FIXED_SPLITS = {
    "ace_fixed_5050": (0.5, 0.5),
    "ace_fixed_7030": (0.7, 0.3),
    "ace_fixed_3070": (0.3, 0.7),
}

LOSSES = ["mse", "hybrid"]
ADJ_TYPES = ["pre", "norm", "mean", "plain"]
GNN_TYPES = ["gcn", "sage", "gat"]

CV_COMMON_ARGS = [
    "--cv",
    "--n_folds", "5",
    "--stratified",
    "--cv_epochs", "500",
    "--patience", "50",
]


def _causal_modes_for(interaction_set: str) -> List[str]:
    if "BFI_PERCEPTION" not in INTERACTION_SETS[interaction_set]:
        return ["none"]
    return CAUSAL_MODES


def _bound_bfi_values(interaction_set: str) -> List[Optional[bool]]:
    if "BFI_PERCEPTION" not in INTERACTION_SETS[interaction_set]:
        return [None]
    return [True, False]


def _freeze_sem_values(causal_mode: str) -> List[Optional[bool]]:
    if causal_mode == "none":
        return [None]
    return [True, False]


def _causal_cli_args(causal_mode: str, freeze_sem: Optional[bool]) -> List[str]:
    if causal_mode == "none":
        return []
    if causal_mode == "causal":
        args = ["--causal"]
    elif causal_mode == "ace_learnable":
        args = ["--ace", "--ace_weight_mode", "learnable"]
    else:
        w_mlp, w_causal = ACE_FIXED_SPLITS[causal_mode]
        args = [
            "--ace", "--ace_weight_mode", "fixed",
            "--ace_w_mlp", str(w_mlp), "--ace_w_causal", str(w_causal),
        ]
    args.append("--freeze_sem" if freeze_sem else "--no-freeze_sem")
    return args


def _tag(interaction_set, causal_mode, bound_bfi, freeze_sem, loss, adj_type, gnn_type) -> str:
    parts = [interaction_set, causal_mode]
    if bound_bfi is not None:
        parts.append(f"bb{int(bound_bfi)}")
    if freeze_sem is not None:
        parts.append(f"fs{int(freeze_sem)}")
    parts += [loss, adj_type, gnn_type]
    return "_".join(parts)


def build_grid() -> List[Dict]:
    grid = []
    for interaction_set, interactions in INTERACTION_SETS.items():
        for causal_mode in _causal_modes_for(interaction_set):
            for bound_bfi in _bound_bfi_values(interaction_set):
                for freeze_sem in _freeze_sem_values(causal_mode):
                    for loss in LOSSES:
                        for adj_type in ADJ_TYPES:
                            for gnn_type in GNN_TYPES:
                                tag = _tag(
                                    interaction_set, causal_mode, bound_bfi,
                                    freeze_sem, loss, adj_type, gnn_type,
                                )
                                cli_args = (
                                    ["--interactions", *interactions]
                                    + CV_COMMON_ARGS
                                    + ["--loss", loss, "--adj_type", adj_type, "--gnn_type", gnn_type]
                                    + _causal_cli_args(causal_mode, freeze_sem)
                                    + ["--run_tag", tag]
                                )
                                if bound_bfi is not None:
                                    cli_args.append("--bound_bfi" if bound_bfi else "--no-bound_bfi")
                                grid.append({
                                    "tag": tag,
                                    "interaction_set": interaction_set,
                                    "causal_mode": causal_mode,
                                    "bound_bfi": bound_bfi,
                                    "freeze_sem": freeze_sem,
                                    "loss": loss,
                                    "adj_type": adj_type,
                                    "gnn_type": gnn_type,
                                    "cli_args": cli_args,
                                })
    return grid


if __name__ == "__main__":
    grid = build_grid()
    tags = [g["tag"] for g in grid]
    print(f"Total configs: {len(grid)}")
    print(f"Unique tags: {len(set(tags))}")
    assert len(tags) == len(set(tags)), "duplicate tags detected"

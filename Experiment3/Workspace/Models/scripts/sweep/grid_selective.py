"""Narrowed ablation grid for the full-data retrain sweep.

Prior sweeps (checkpoints/sweep_full, 1080 trials; checkpoints/optuna_full35,
199 trials) were run on the 35-participant set with 5 users excluded. This
grid targets a retrain on the full data, restricted to the axes/values that
matched-pair analysis of sweep_full actually supported:

- interaction_set: all three kept (default/single/multiple) so the
  default-vs-single and default-vs-multiple comparisons can be re-verified
  on the new data, not just multiple.
- causal_mode: dropped the three ace_fixed_* splits -- none of them beat
  `none` in a matched-pair comparison (same loss/adj_type/gnn_type/bound_bfi,
  only causal_mode/freeze_sem differing) on sweep_full. Kept `causal` (CE)
  and `ace_learnable` as a single spot-check of the ACE architecture without
  the fixed-split fan.
- bound_bfi: fixed True. bound_bfi=False matched-pair delta was negative
  (-0.006 vs +0.020 for True) -- not a free axis, an active loss.
- freeze_sem: kept both True/False. Frozen SEM underperformed trainable SEM
  in every matched breakdown, but re-verifying that on new data is cheap
  (it's a free binary axis under causal_mode != none) and worth confirming
  rather than assuming.
- loss: fixed to mse (mildly ahead of hybrid across breakdowns, and fixing
  it here trades a weak signal for a 2x grid reduction).
- adj_type: dropped "mean" (kept plain/pre/norm). "mean" was never a clear
  standout in the matched breakdowns; dropping the weakest-appearing value
  buys another ~25% cut. gnn_type kept at all three values -- no consistent
  winner there, and it's cheap to keep full.

Total: 99 configs (vs sweep_full's ~1140), reusing grid.py's INTERACTION_SETS
and CV_COMMON_ARGS for consistency with the earlier sweep's schema.
"""

from grid import CV_COMMON_ARGS, INTERACTION_SETS

CAUSAL_MODES = ["none", "causal", "ace_learnable"]
LOSSES = ["mse"]
ADJ_TYPES = ["plain", "pre", "norm"]
GNN_TYPES = ["gcn", "sage", "gat"]


def _causal_modes_for(interaction_set: str) -> list[str]:
    if "BFI_PERCEPTION" not in INTERACTION_SETS[interaction_set]:
        return ["none"]
    return CAUSAL_MODES


def _freeze_sem_values(causal_mode: str) -> list[bool | None]:
    if causal_mode == "none":
        return [None]
    return [True, False]


def _causal_cli_args(causal_mode: str, freeze_sem: bool | None) -> list[str]:
    if causal_mode == "none":
        return []
    if causal_mode == "causal":
        args = ["--causal"]
    else:  # ace_learnable
        args = ["--ace", "--ace_weight_mode", "learnable"]
    args.append("--freeze_sem" if freeze_sem else "--no-freeze_sem")
    return args


def _tag(interaction_set, causal_mode, freeze_sem, loss, adj_type, gnn_type) -> str:
    parts = [interaction_set, causal_mode]
    if freeze_sem is not None:
        parts.append(f"fs{int(freeze_sem)}")
    parts += [loss, adj_type, gnn_type]
    return "_".join(parts)


def build_grid() -> list[dict]:
    grid = []
    for interaction_set, interactions in INTERACTION_SETS.items():
        has_bfi = "BFI_PERCEPTION" in interactions
        for causal_mode in _causal_modes_for(interaction_set):
            for freeze_sem in _freeze_sem_values(causal_mode):
                for loss in LOSSES:
                    for adj_type in ADJ_TYPES:
                        for gnn_type in GNN_TYPES:
                            tag = _tag(
                                interaction_set, causal_mode, freeze_sem, loss, adj_type, gnn_type
                            )
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
                                ]
                                + _causal_cli_args(causal_mode, freeze_sem)
                                + ["--run_tag", tag]
                            )
                            if has_bfi:
                                cli_args.append("--bound_bfi")
                            grid.append(
                                {
                                    "tag": tag,
                                    "interaction_set": interaction_set,
                                    "causal_mode": causal_mode,
                                    "bound_bfi": True if has_bfi else None,
                                    "freeze_sem": freeze_sem,
                                    "loss": loss,
                                    "adj_type": adj_type,
                                    "gnn_type": gnn_type,
                                    "cli_args": cli_args,
                                }
                            )
    return grid


if __name__ == "__main__":
    grid = build_grid()
    tags = [g["tag"] for g in grid]
    print(f"Total configs: {len(grid)}")
    print(f"Unique tags: {len(set(tags))}")
    assert len(tags) == len(set(tags)), "duplicate tags detected"

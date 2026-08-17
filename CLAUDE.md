# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Scope

This repo is the workspace for **Experiment 3** of an eHMI (external Human-Machine
Interface) voice-perception study. It contains experimental stimuli, raw data,
audio-feature extraction, statistical analysis, and — the main code — a machine-learning
pipeline that models how listeners perceive synthesized eHMI voices.

Most engineering work happens in `Experiment3/Workspace/Models/`. Its `src/README.md`
is the authoritative, detailed reference for the modeling code; read it before working
there. The other workspace areas (`ExpDataAnalyze/`, `FeatureProcessing/`) are mostly
notebooks and one-off scripts.

`Experiment3/Workspace/Models/archive/GHCF/` is a **git submodule** pointing at the
upstream author implementation (`chenchongthu/GHCF`). Treat it as read-only reference;
do not edit it as part of normal work.

## Environment

Conda environment `eHMI` (Python 3.10), defined in `environment.yml` at the repo root:

```bash
conda env create -f environment.yml   # first time
conda activate eHMI
```

The modeling code has a narrower dependency set installable on its own:

```bash
pip install -r Experiment3/Workspace/Models/src/requirements.txt
```

There is no test suite and no configured linter in this repo.

## Running the Models

**Critical working-directory convention:** all model commands run from
`Experiment3/Workspace/Models/` (the directory one level above `src/`). Data and output
paths in `src/configs/config.py` are resolved relative to that directory, so running
from elsewhere breaks path resolution.

```bash
cd Experiment3/Workspace/Models

# Recommender training (single split)
python src/main.py train
python src/main.py train --interactions RATING BFI_PERCEPTION
python src/main.py train --causal          # GHCF_CE: non-BFI predicted via causal SEM head
python src/main.py train --ace             # GHCF_ACE: blend GNN + causal head for non-BFI

# K-fold cross-validation (adds --cv to the train subcommand)
python src/main.py train --cv --n_folds 5 --stratified --cv_epochs 200

# Evaluate a saved run
python src/main.py evaluate --model_dir checkpoints/<run_dir>

# Offline causal discovery (produces artifacts consumed by --causal / --ace)
python src/run_causal_discovery.py

# Aggregate/plot CV results (or re-analyze an existing CV run)
python src/cross_validation.py --analyze checkpoints/<cv_run_dir>
```

`--causal` and `--ace` are mutually exclusive. Both require causal artifacts from
`run_causal_discovery.py`, and require `NORMALIZE_INTERACTIONS` (default on) so targets
share the SEM head's z-scored output space.

## Architecture: the Two-Stage Causal-Recommender Pipeline

The defining idea of this codebase is that a **causal DAG is discovered offline, then
embedded inside the GNN as a differentiable head**, so ranking losses back-propagate
through the causal structure. Understanding this requires reading several files together
(`models/CausalModel.py`, `models/CausalHead.py`, `models/CausalMixin.py`, and the GHCF
variants).

1. **Stage 1 — causal discovery (`models/CausalModel.py`, `run_causal_discovery.py`).**
   A LiNGAM model (config: `src/configs/causal_config.yaml`) learns a causal graph from
   personality **BFI-TIPI differences** (exogenous) → perceptual mediators (MOS-X2 voice
   quality, SQ subjective quality) → **Rating** (sink), with prior knowledge and
   bootstrap robustness. It writes an adjacency matrix and normalization stats to
   `causal_outputs/`.

2. **Stage 2 — recommender (`models/GHCF*.py`, `main.py`/`train.py`).** A Graph
   Heterogeneous Collaborative Filtering GNN operates over a user-item graph with four
   heterogeneous relations: `RATING`, `BFI_PERCEPTION`, `VOICE_QUALITY`,
   `SUBJECTIVE_FEELING`. `CausalHead.SEMHead` wraps the discovered DAG as a torch module
   (linear SEM, weights initialized from LiNGAM coefficients and masked to the DAG,
   evaluated in causal order). `CausalMixin` computes BFI-TIPI differences from the
   graph's BFI predictions and pushes them through the SEM head — entirely inside
   autograd — so rating/quality/feeling losses train the GNN through the causal path.

### Model variants (all in `models/`)

- **`GHCF`** — base multi-relational GNN; one prediction head per interaction type.
- **`GHCF_CE`** (`--causal`) — GNN trained on `BFI_PERCEPTION` only; every other task is
  predicted purely by the causal SEM head.
- **`GHCF_ACE`** (`--ace`) — GNN trained on all tasks; for non-BFI tasks it blends the
  GNN head with the causal head (`pred = w_mlp·pred_gnn + w_causal·pred_causal`), with a
  per-type learnable gate (`--ace_weight_mode learnable`) or fixed weights
  (`--ace_weight_mode fixed --ace_w_mlp … --ace_w_causal …`).
- **`HeteroGNN`** — the per-relation GCN/GraphSAGE/GAT message-passing layers
  (`--gnn_type`) and relation/node embeddings.

### Configuration is centralized

`src/configs/config.py` is the **single source of truth** for all defaults. The argument
parser (`src/utility/parser.py`) imports its defaults from that file; any CLI flag then
overrides the corresponding config value. To change a default globally, edit `config.py`;
to override for one run, pass a flag. Defaults are deliberately tuned compact (32-dim
embeddings, single GNN layer, strong dropout) because the dataset is small (~30 users ×
~50 items) and overfits easily.

### Data flow

`data/database.xlsx` (and `database_similarity.xlsx` for causal discovery) holds user
features (demographics, TIPI/BFI personality), item features (eHMI gender, content type,
character), and recorded interactions. `utility/dataloader.py` loads and normalizes it;
preprocessed interaction tensors are cached under `data/processed_*`. Training writes a
timestamped run directory under `checkpoints/` (`best_state.pth`, `last_state.pth`,
TensorBoard logs, JSON config snapshot). Evaluation reports Precision@K, Recall@K,
NDCG@K, Hit@K, and AUC.

## Git

`Experiment3` is committed at the repo root alongside `manuscripts/` and
`environment.yml`. Remember the GHCF submodule when committing — changes there are a
separate concern from the main project.

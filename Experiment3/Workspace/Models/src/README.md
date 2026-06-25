# GHCF eHMI Recommendation — Source Code

PyTorch implementation of a Graph Heterogeneous Collaborative Filtering (GHCF)
recommender for external Human-Machine Interface (eHMI) voices, combined with a
LiNGAM-based causal discovery pipeline. The system jointly models multiple
interaction types (voice rating, perceived personality, perceived voice
quality, and subjective feeling) over a heterogeneous user-item graph, and can
optionally route non-personality predictions through a differentiable causal
structural-equation model (SEM).

All commands below assume the `Models/` directory as the working directory
(one level above this `src/` folder), because data and output paths in
`configs/config.py` are resolved relative to it.

## Directory Layout

```text
src/
├── main.py                 # Unified CLI entry point (train / evaluate)
├── train.py                # Trainer class: training loop, early stopping, checkpointing
├── evaluate.py             # Evaluator class: load a checkpoint and report metrics
├── cross_validation.py     # K-fold (optionally stratified) cross-validation framework
├── run_cv.py               # Backwards-compatible wrapper for cross_validation.main
├── run_causal_discovery.py # Entry point for the LiNGAM causal discovery pipeline
├── analyze_cv.py           # Aggregate and plot cross-validation results
├── configs/
│   ├── config.py           # Single source of truth for all defaults
│   └── causal_config.yaml  # LiNGAM causal discovery configuration
├── models/
│   ├── GHCF.py             # Base multi-relational GNN recommender
│   ├── GHCF_CE.py          # GNN on BFI only; other tasks via the causal SEM head
│   ├── GHCF_ACE.py         # GNN on all tasks; ensembles GNN + causal head for non-BFI
│   ├── HeteroGNN.py        # Heterogeneous GCN/GraphSAGE/GAT layers + relation embeddings
│   ├── CausalModel.py      # Offline LiNGAM causal discovery (structure learning)
│   ├── CausalHead.py       # Differentiable SEM head built from the discovered DAG
│   └── CausalMixin.py      # Glue between the GNN and the SEM head
└── utility/
    ├── dataloader.py       # eHMIDataset, data loading and preprocessing
    ├── batch_processing.py # Batch construction for training/evaluation
    ├── metrics.py          # Precision@K, Recall@K, NDCG@K, Hit@K, AUC
    ├── gradnorm.py         # GradNorm dynamic multi-task loss balancing
    ├── helper.py           # Seeding, path resolution, normalization-stats loading
    └── plotting.py         # Plotting helpers
```

## Pipeline Overview

The project runs in two stages:

1. **Causal discovery (offline, optional).** `run_causal_discovery.py` fits a
   LiNGAM model on the survey data to learn a causal DAG from BFI-TIPI
   personality differences through mediator variables (voice quality, subjective
   quality) to the final rating. The discovered adjacency matrix and
   normalization statistics are written to `causal_outputs/`.

2. **Recommender training and evaluation.** `main.py` builds the heterogeneous
   graph, trains a GHCF variant, and evaluates ranking metrics. The causal
   variants load the artifacts from stage 1 and embed the DAG as a
   differentiable SEM head so that rating/quality/feeling losses back-propagate
   into the GNN through the causal structure.

## Models

| Module | Class | Description |
| --- | --- | --- |
| `models/GHCF.py` | `GHCF` | Base model. Multi-relational embedding propagation over the user-item graph, followed by a multi-task prediction module for every interaction type. |
| `models/GHCF_CE.py` | `GHCF_CE` | Builds the GNN on `BFI_PERCEPTION` only. Graph BFI predictions are converted to BFI-TIPI differences and propagated through the SEM head to predict all other interactions. The full path is differentiable. |
| `models/GHCF_ACE.py` | `GHCF_ACE` | Adaptive Causal Ensemble. Builds the GNN on all interaction types, and for non-BFI predictions blends the GNN head with the causal SEM head: `pred = w_mlp * pred_gnn + w_causal * pred_causal`. Weights are a per-type learnable gate (`--ace_weight_mode learnable`) or fixed hyperparameters (`--ace_weight_mode fixed`). |
| `models/HeteroGNN.py` | `HeteroGCNLayer`, `HeteroGraphSAGELayer`, `HeteroGATLayer`, `MultiRelationEmbedding` | Per-relation message passing layers and relation/node embeddings. Layer type is selected with `--gnn_type`. |
| `models/CausalModel.py` | `CausalModel` | Offline LiNGAM structure learning with prior knowledge, bootstrap robustness analysis, graph visualization, and artifact export. |
| `models/CausalHead.py` | `SEMHead` | Linear SEM constrained to the discovered DAG. Weights are initialized from the LiNGAM coefficients and masked by the DAG structure; equations are evaluated in causal order so mediated paths propagate correctly. Operates in the same z-scored space LiNGAM was fitted on. |
| `models/CausalMixin.py` | `CausalMixin` | Owns the `SEMHead`, computes BFI-TIPI differences from graph BFI predictions, and produces rating/quality/feeling predictions inside autograd. |

## Configuration

`configs/config.py` is the single source of truth. The argument parser
(`utility/parser.py`) reads its defaults from this file, and any command-line
flag overrides the corresponding config value. Key groups:

- **Data columns and scales** — `USER_FEATURES`, `ITEM_FEATURES`, the TIPI/BFI
  personality columns, MOS-X2 voice-quality columns, and SQ subjective-quality
  columns, each with its measurement scale.
- **`INTERACTION_TYPES`** — the four heterogeneous relations and their source
  columns: `RATING`, `BFI_PERCEPTION`, `VOICE_QUALITY`, `SUBJECTIVE_FEELING`.
- **`GRAPH_CONFIG`** — embedding dimension (32), layer sizes, dropout, GNN type,
  attention heads, per-task loss weights, and GradNorm settings. Tuned compact
  on purpose: the dataset is small (about 30 users x 50 items), so the defaults
  favor a single small GNN layer with strong dropout to limit overfitting.
- **Training settings** — `BATCH_SIZE`, `NUM_EPOCHS` (300), `LEARNING_RATE`,
  `WEIGHT_DECAY`, `EARLY_STOP_PATIENCE`, `EVAL_INTERVAL`, `RANDOM_SEED`,
  `TEST_SIZE`.
- **Loss mode** — `LOSS_MODE` (`mse` or `hybrid`). `mse` regresses rating
  magnitude only; `hybrid` adds a BPR pairwise-ranking term (`BPR_LAMBDA`,
  `BPR_NUM_NEG`) so training also optimizes the item ordering that ranking
  metrics evaluate. BFI/VQ/SF tasks always use MSE.
- **`NORMALIZE_INTERACTIONS`** — normalizes interaction targets (BFI MinMax;
  RATING/VQ/SF z-scored against the shared causal stats). Required by the causal
  models for consistent losses.
- **Causal paths** — `CAUSAL_MODEL_PATH`, `CAUSAL_CONFIG_PATH`.
- **`ADJ_TYPE`** — adjacency normalization. `pre` matches the reference GHCF
  (symmetric `D^-1/2 A D^-1/2`, no self-loop). Alternatives: `norm`, `mean`,
  `plain`.
- **`CV_CONFIG`** — number of folds (5), stratification, epochs per fold.

## Command-Line Interface

`main.py` exposes two subcommands, `train` and `evaluate`.

### Train

```bash
# Train with all defaults from configs/config.py
python src/main.py train

# Train on a subset of interaction types
python src/main.py train --interactions RATING BFI_PERCEPTION

# Override hyperparameters (these win over config.py)
python src/main.py train --embed_size 128 --lr 0.0001 --epoch 500

# Use the causal SEM head for non-BFI predictions
python src/main.py train --causal

# Use the adaptive causal ensemble (GNN + causal head, learnable blend)
python src/main.py train --ace
# ... or a fixed manual blend
python src/main.py train --ace --ace_weight_mode fixed --ace_w_mlp 0.7 --ace_w_causal 0.3

# Hybrid ranking loss with GradNorm multi-task balancing
python src/main.py train --loss hybrid --bpr_lambda 1.0 --use_gradnorm
```

Selected `train` arguments:

| Flag | Default (config) | Description |
| --- | --- | --- |
| `--interactions` | all four types | Interaction types to include in the graph. |
| `--lr` | `LEARNING_RATE` | Learning rate. |
| `--batch_size` | `BATCH_SIZE` | Batch size. |
| `--epoch` | `NUM_EPOCHS` | Maximum epochs. |
| `--patience` | `EARLY_STOP_PATIENCE` | Early-stopping patience (epochs without improvement). |
| `--decay` | `WEIGHT_DECAY` | Weight decay (L2). |
| `--embed_size` | `GRAPH_CONFIG.embedding_dim` | Embedding dimension. |
| `--gnn_type` | `GRAPH_CONFIG.gnn_type` | `gcn`, `sage`, or `gat`. |
| `--n_heads` | `GRAPH_CONFIG.n_heads` | Attention heads (used only for `gat`). |
| `--loss` | `LOSS_MODE` | RATING loss mode: `mse` or `hybrid`. |
| `--bpr_lambda` | `BPR_LAMBDA` | BPR ranking term weight in hybrid mode. |
| `--use_gradnorm` | off | Enable GradNorm dynamic loss balancing. |
| `--gradnorm_alpha` | `GRAPH_CONFIG.gradnorm_alpha` | GradNorm restoring force. |
| `--causal` | off | Predict non-BFI interactions via the causal SEM head only (`GHCF_CE`). |
| `--ace` | off | Adaptive Causal Ensemble: blend GNN and causal head for non-BFI (`GHCF_ACE`). |
| `--ace_weight_mode` | `ACE_WEIGHT_MODE` | ACE mixing: `learnable` (per-type gate) or `fixed`. |
| `--ace_w_mlp` / `--ace_w_causal` | config | Fixed-mode GNN/causal weights (applied to all non-BFI types). |
| `--causal_path` | `CAUSAL_MODEL_PATH` | Path to the fitted causal model artifact. |
| `--causal_config` | `CAUSAL_CONFIG_PATH` | Path to the causal YAML config. |
| `--normalize_interactions` | `NORMALIZE_INTERACTIONS` | Toggle target normalization (`--no-normalize_interactions` disables). |
| `--save_flag` | `1` | Save checkpoints. |

### Cross-Validation

Add `--cv` to the `train` command to run k-fold cross-validation instead of a
single train/test split:

```bash
# 5-fold stratified cross-validation
python src/main.py train --cv --n_folds 5 --stratified --cv_epochs 200
```

| Flag | Default (config) | Description |
| --- | --- | --- |
| `--cv` | off | Run k-fold CV instead of a single split. |
| `--n_folds` | `CV_CONFIG.n_folds` | Number of folds. |
| `--stratified` | `CV_CONFIG.stratified` | Stratify users by interaction count. |
| `--cv_epochs` | `CV_CONFIG.cv_epochs` | Epochs per fold. |
| `--adj_type` | `ADJ_TYPE` | Adjacency normalization type. |

`run_cv.py` is a thin backwards-compatible wrapper; prefer `main.py train --cv`
or running `cross_validation.py` directly.

### Evaluate

```bash
python src/main.py evaluate --model_dir checkpoints/model_20241218_120000
python src/main.py evaluate --model_dir <dir> --checkpoint last_state.pth
python src/main.py evaluate --model_dir <dir> --k_values "[3,5,10]"
```

| Flag | Default | Description |
| --- | --- | --- |
| `--model_dir` | required | Directory containing the trained model. |
| `--checkpoint` | `best_state.pth` | `best_state.pth` or `last_state.pth`. |
| `--k_values` | `[3,5,10]` | Top-K cutoffs for ranking metrics. |
| `--test_flag` | `part` | `part` or `full` test evaluation. |

### Causal Discovery

```bash
python src/run_causal_discovery.py
python src/run_causal_discovery.py --config src/configs/causal_config.yaml
```

This fits the LiNGAM model, runs bootstrap analysis, prints the system of
equations (standard and robust), and writes the adjacency matrices,
normalization statistics, figures, and evaluation results into the directory
configured in `causal_config.yaml` (`causal_outputs/` by default). The
recommender's causal variants then consume these artifacts.

`causal_config.yaml` defines the input file, output directory, BFI-TIPI
difference rescaling, the feature partition (exogenous inputs, mediators,
output), the LiNGAM algorithm (`DirectLiNGAM` or `ICALiNGAM`), bootstrap
parameters, prior knowledge, and visualization/output formats.

### Analyze Cross-Validation Results

```bash
python src/analyze_cv.py <cv_results_dir>
```

Loads `cv_results.json` and produces aggregate statistics and per-fold metric
plots.

## Data and Outputs

- **Input** — `data/database.xlsx` (and `database_similarity.xlsx` for causal
  discovery) with user features (demographics, TIPI/BFI personality), item
  features (eHMI gender, content type, character), and the recorded
  interactions. Preprocessed interaction tensors are cached under
  `data/processed_*`.
- **Training output** — per-run directory under `checkpoints/` containing model
  checkpoints (`best_state.pth`, `last_state.pth`), TensorBoard logs, and a JSON
  config snapshot.
- **Evaluation output** — Precision@K, Recall@K, NDCG@K, Hit@K, and AUC.
- **Causal output** — adjacency matrices, normalization statistics, causal-graph
  figures, and model-fit/evaluation spreadsheets under `causal_outputs/`.

## Installation

```bash
pip install -r src/requirements.txt
```

Requirements include PyTorch and PyTorch Geometric (with `torch-scatter` and
`torch-sparse`), pandas/numpy/scipy/scikit-learn, openpyxl, tqdm, tensorboard,
matplotlib/seaborn, and the causal-discovery stack (`lingam`, `joblib`,
`graphviz`, `pyyaml`).

## Reference

The GHCF architecture is adapted from Chen et al., "Graph Heterogeneous
Multi-Relational Recommendation" (AAAI 2021). The original reference
implementation is kept under `../archive/GHCF/`.

"""
Configuration settings for the model, training, and evaluation
"""

from pathlib import Path

import torch

# Directory paths
# Repo layout: this file is <Models>/src/configs/config.py, so parents[2] is the
# Models/ working directory that holds data/ and causal_outputs/. Anchoring to it
# (absolute) makes the derived paths independent of the current working directory.
BASE_DIR = Path(__file__).resolve().parents[2]
SET = "35"  # Set to "25" for the 25-participant subset, or "35" for all participants
DATA_DIR = BASE_DIR / "data" / SET
DATA_INTERACTIONS_DIR = DATA_DIR / "interactions"

# Data settings
DATABASE_PATH = DATA_DIR / "database.xlsx"
TIPI_COLS = [
    "TIPI-O",
    "TIPI-C",
    "TIPI-E",
    "TIPI-A",
    "TIPI-N",
]  # TIPI personality traits (1-14 scale)
TIPI_SCALE = (1, 14)  # TIPI scale range

BFI_COLS = ["BFI-O", "BFI-C", "BFI-E", "BFI-A", "BFI-N"]  # BFI personality traits (1-10 scale)
BFI_SCALE = (1, 10)  # BFI scale range

MOS_X2_COLS = [
    "MOS-X2 Q1",
    "MOS-X2 Q2",
    "MOS-X2 Q3",
    "MOS-X2 Q4",
]
MOS_X2_SCALE = (0, 10)  # MOS-X2 scale range (0-10)

SQ_COLS = [
    "SQ1",
    "SQ2",
    "SQ3",
    "SQ4",
    "SQ5",
]
SQ_SCALE = (1, 5)  # Subjective Quality scale range (1-5)

# Features
USER_FEATURES = ["Participant ID", "Participant Gender"] + TIPI_COLS
ITEM_FEATURES = ["eHM ID", "eHMI Gender", "eHMI Content Type", "eHMI Character"]

# Define heterogeneous interaction types
INTERACTION_TYPES = {
    "RATING": {
        "columns": ["Rating score"],
        "scale": (0, 10),
    },
    "BFI_PERCEPTION": {
        "columns": BFI_COLS,
        "scale": BFI_SCALE,
    },
    "VOICE_QUALITY": {
        "columns": MOS_X2_COLS,
        "scale": MOS_X2_SCALE,
    },
    "SUBJECTIVE_FEELING": {
        "columns": SQ_COLS,
        "scale": SQ_SCALE,
    },
}

# Graph model settings
# Dataset is small (30 users × 50 items = 900 rows) so keep the model compact
# to avoid overfitting: single GNN layer, 32-dim embeddings, strong dropout.
GRAPH_CONFIG = {
    "embedding_dim": 32,  # 64 → 32; 80 nodes × 32 dims is already expressive
    "layer_sizes": [32],  # two 64-layers → single 32-layer (less capacity, less over-fit)
    "dropout_rates": [0.4],  # 0.1 → 0.4; stronger regularization for small data
    "gnn_type": "gcn",  # Type of GNN layer ('gcn', 'sage', 'gat')
    "n_heads": 4,  # Number of attention heads (only used for 'gat')
    "interaction_weights": {  # Weights for loss computation
        "RATING": 1.0,
        "BFI_PERCEPTION": 0.8,
        "VOICE_QUALITY": 0.7,
        "SUBJECTIVE_FEELING": 0.6,
    },
    "use_gradnorm": False,  # Whether to use GradNorm for dynamic weight balancing
    "gradnorm_alpha": 1.5,  # GradNorm restoring force hyperparameter (typically 0.12-3.0)
}

# Model architecture settings
MODEL_CONFIG = {
    "user_feature_dim": len(USER_FEATURES) - 1,  # Gender (1) + TIPI scores (5)
    "item_feature_dim": len(ITEM_FEATURES) - 1,  # Gender, Content Type, Character (encoded)
}

# Training settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 300  # 1000 → 300; early stopping handles real termination
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4  # 1e-5 → 1e-4; stronger L2 for small dataset
EARLY_STOP_PATIENCE = 20  # 100 → 50; small data converges (or overfits) fast
EVAL_INTERVAL = 10  # Evaluate on the test set every N epochs
RANDOM_SEED = 42
TEST_SIZE = 0.2
RATING_THRESHOLD = 0.0  # Threshold for positive ratings (after normalization)

# Loss mode for the RATING task (BFI/VOICE_QUALITY/SUBJECTIVE_FEELING are always
# MSE — pairwise ranking is meaningless for multi-dim perception scores):
#   - "mse":    rating magnitude regression only (original behavior)
#   - "hybrid": MSE + BPR_LAMBDA * BPR pairwise ranking, so training also
#               optimizes the item ordering that recall/nDCG@k evaluate while
#               keeping rating magnitude (needed by the causal SEM head).
LOSS_MODE = "hybrid"
BPR_LAMBDA = 1.0  # weight of the BPR ranking term in hybrid mode (tune)
BPR_NUM_NEG = 1  # negatives sampled per positive rating for BPR

# Whether to normalize interaction *targets*. Decoupled from feature
# normalization (utility.dataloader always applies it via normalize_features) on purpose:
#   - BFI_PERCEPTION: MinMax to [0, 1] (BFI scale)
#   - RATING / VOICE_QUALITY / SUBJECTIVE_FEELING: z-scored with the causal
#     model's normalization_stats.csv when available, so the targets share the
#     SEM head's output space exactly (single source of truth). Falls back to
#     per-split z-score only if those stats are missing.
# The causal models (GHCF-CE / GHCF-RB-causal) require this to be True for
# their rating/quality/feeling losses to be computed in a consistent space.
NORMALIZE_INTERACTIONS = True

# Causal model settings (paths relative to the Models/ working directory)
CAUSAL_MODEL_PATH = "causal_outputs/models/lingam_causal_model_robust_model_data.npy"
CAUSAL_CONFIG_PATH = "src/configs/causal_config.yaml"

# Whether the differentiable SEM head's coefficients are fine-tuned during
# training. The DAG structure (mask) is always fixed; this only controls whether
# the (masked) weights are learnable. Freezing (--freeze_sem) keeps them at the
# discovered LiNGAM values so the causal head stays a pure structural prior.
FREEZE_SEM = False

# Squash the BFI perception head through a sigmoid so predictions stay in [0, 1].
# BFI targets are MinMax-scaled to [0, 1] and the BFI prediction is the input to
# the causal SEM head (whose rescaled BFI-TIPI difference assumes BFI in [0, 1]),
# so bounding makes that invariant exact. RATING/VQ/SF heads are z-scored and
# stay unbounded regardless.
BOUND_BFI_OUTPUT = True

# GHCF-ACE: how to combine the GNN head and the causal SEM head for the
# non-BFI predictions (RATING / VOICE_QUALITY / SUBJECTIVE_FEELING):
#   pred = w_mlp * pred_gnn + w_causal * pred_causal
#   - "learnable": per-type convex gate, w_causal = sigmoid(logit) (init 0.5/0.5)
#   - "fixed":     per-type hyperparameters from ACE_FIXED_WEIGHTS (manual tuning)
ACE_WEIGHT_MODE = "learnable"
ACE_FIXED_WEIGHTS = {
    "RATING": {"w_mlp": 0.5, "w_causal": 0.5},
    "VOICE_QUALITY": {"w_mlp": 0.5, "w_causal": 0.5},
    "SUBJECTIVE_FEELING": {"w_mlp": 0.5, "w_causal": 0.5},
}


# Adjacency matrix settings
# 'pre' matches the reference GHCF: symmetric D^-1/2 A D^-1/2 with NO self-loop
# (self-info comes from the layer self-transform + layer-0 term). Other options:
# 'norm' (symmetric with self-loop), 'mean' (row-normalized), 'plain' (raw values).
ADJ_TYPE = "pre"

# Cross-validation settings
CV_CONFIG = {
    "n_folds": 5,  # 5-fold: 24 train / 6 val users — reasonable for 30 users
    "stratified": True,
    "cv_epochs": 200,  # 100 → 200; early stopping terminates early anyway
    "random_state": RANDOM_SEED,
}

# Per-fold causal discovery in CV (see
# docs/superpowers/specs/2026-09-04-per-fold-causal-discovery-cv-design.md).
# `run_causal_discovery.py --cv` fits DirectLiNGAM + bootstrap once per fold on
# that fold's training participants and caches the K artifact sets here; CV /
# sweep runs with --causal/--ace then load the artifact matching the current
# fold instead of the single global one. `--cv_global_causal` restores the old
# shared-artifact behaviour.
CV_CAUSAL_CACHE_DIR = "checkpoints/causal_fold_cache"  # relative to Models/
CV_CAUSAL_JOBS = 1  # joblib workers for the per-fold bootstrap (1 = safe under a sweep)
CV_CAUSAL_LOCK_TIMEOUT = 1800  # seconds before a fold build lock is treated as stale
CV_CAUSAL_MIN_FOLD_ROWS = 100  # warn if a fold's filtered training set is smaller

"""
Configuration settings for the model, training, and evaluation
"""

from pathlib import Path

import torch

# Directory paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
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

LABEL_COLS = [
    "Label-O",
    "Label-C",
    "Label-E",
    "Label-A",
    "Label-N",
]  # Labels for BFI traits (0-1 scale)
LABEL_SCALE = (0, 1)  # Label scale range

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
    "layer_sizes": [32],  # two 64-layers → single 32-layer (less capacity, less overfit)
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
    "gradnorm_update_freq": 1,  # How often to update GradNorm weights (every N steps)
}

# Model architecture settings
MODEL_CONFIG = {
    "user_feature_dim": len(USER_FEATURES) - 1,  # Gender (1) + TIPI scores (5)
    "item_feature_dim": len(ITEM_FEATURES) - 1,  # Gender, Content Type, Character (encoded)
    "normalize_features": True,
    "feature_norm_method": "mixed",  # 'mixed', 'minmax', or 'zscore'
}

# Training settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 300  # 1000 → 300; early stopping handles real termination
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4  # 1e-5 → 1e-4; stronger L2 for small dataset
EARLY_STOP_PATIENCE = 50  # 100 → 50; small data converges (or overfits) fast
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
LOSS_MODE = "mse"
BPR_LAMBDA = 1.0  # weight of the BPR ranking term in hybrid mode (tune)
BPR_NUM_NEG = 1  # negatives sampled per positive rating for BPR

# Whether to normalize interaction *targets*. Decoupled from feature
# normalization (MODEL_CONFIG["normalize_features"]) on purpose:
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


# Models: GHCF eHMI Recommendation

This directory contains PyTorch code for the Graph Heterogeneous Collaborative Filtering (GHCF) model, used for eHMI voice recommendation and personality prediction.

## Main Scripts

- `main.py`: Entry point for training and evaluation
- `train.py`: Model training logic
- `evaluate.py`: Model evaluation logic

## Core Models

- `models/GHCF.py`: Main multi-relational GNN model
- `models/GHCF_Causal.py`: GHCF with causal prediction
- `models/GHCF_Refine.py`: GHCF with refined causal logic
- `models/HeteroGNN.py`: Heterogeneous GNN layers

## Usage

Install requirements, then run:

```bash
python main.py train           # Train model
python main.py evaluate        # Evaluate model
```

Add options for embedding size, learning rate, epochs, and interaction types as needed.

Run Causal process, start from `Models/` folder:

```bash
python src/models/CausalModel.py
```

## Features

- Multi-relational graph neural network (GNN)
- Handles multiple interaction types (RATING, BFI_PERCEPTION, VOICE_QUALITY, SUBJECTIVE_FEELING)
- Supports causal and refined prediction modes
- Configurable model parameters

## Architecture

Node and relation embeddings → Heterogeneous GNN layers → Multi-task prediction

**Project Structure**:

```text
src/
├── main.py                # Unified entry point
├── train.py               # Trainer class with modular training logic
├── evaluate.py            # Evaluator class for model assessment
├── utility/parser.py      # Unified argument parsing for train/evaluate
├── models/                # Causal Models, GHCF and HeteroGNN implementations
└── checkpoints/           # Model outputs and logs
```

## Data & Output

-   **Input**: `database.xlsx` with user features (demographics, Big Five), item features (gender, content), and interactions
-   **Training Output**: Model checkpoints, TensorBoard logs, configuration JSON
-   **Evaluation Output**: Precision@K, Recall@K, NDCG@K, Hit@K, AUC metrics

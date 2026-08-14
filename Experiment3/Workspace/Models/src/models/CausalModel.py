"""
Causal Model for analyzing causal relationships using LiNGAM algorithm.

This module provides a comprehensive causal analysis framework including:
- Data preprocessing and normalization
- Causal structure learning with prior knowledge
- Bootstrap analysis for robustness
- Visualization of causal graphs
- Model evaluation and inference capabilities
- Prediction of both intermediate and output features
"""

import logging
import os
from pathlib import Path

import graphviz
import lingam
import numpy as np
import pandas as pd
import yaml
from lingam.bootstrap import BootstrapResult
from lingam.utils import (
    evaluate_model_fit,
    make_prior_knowledge,
    print_causal_directions,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set numpy options for better display
np.set_printoptions(precision=3, suppress=True)


def _build_lingam_model(algorithm: str, prior_knowledge: np.ndarray | None):
    """Construct an (unfitted) LiNGAM estimator from an algorithm name.

    Defined at module level so it can be pickled and sent to joblib workers.
    """
    if algorithm == "DirectLiNGAM":
        return lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
    if algorithm == "ICALiNGAM":
        return lingam.ICALiNGAM()
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _bootstrap_chunk(
    data: np.ndarray,
    n_sampling: int,
    seed: int,
    algorithm: str,
    prior_knowledge: np.ndarray | None,
):
    """Run a slice of the bootstrap in a worker process.

    Returns the per-sample adjacency matrices and total-effect matrices for
    this chunk so the parent can concatenate them into a single result.
    """
    np.random.seed(seed)
    model = _build_lingam_model(algorithm, prior_knowledge)
    result = model.bootstrap(data, n_sampling=n_sampling)
    return np.asarray(result.adjacency_matrices_), np.asarray(result.total_effects_)


def _split_counts(total: int, n_parts: int) -> list:
    """Split ``total`` samples into ``n_parts`` near-equal positive chunks."""
    n_parts = max(1, min(n_parts, total))
    base, remainder = divmod(total, n_parts)
    return [base + (1 if i < remainder else 0) for i in range(n_parts)]


class CausalModel:
    """
    A comprehensive causal modeling class using DirectLiNGAM algorithm.

    This class handles the complete pipeline from data loading to inference,
    including bootstrap analysis, visualization, and model persistence.
    """

    def __init__(self, config_path: str):
        """
        Initialize the CausalModel with configuration.

        Parameters:
        config_path (str): Path to the YAML configuration file.
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.data = None
        self.features = None
        self.adjacency_matrix = None
        self.bootstrap_result = None
        self.probability_matrix = None
        self._prior_knowledge = None

        # Set random seed for reproducibility
        np.random.seed(self.config["model"]["random_seed"])

        # Create output directories
        self._setup_directories()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def _setup_directories(self) -> None:
        """Create necessary output directories."""
        output_dir = Path(self.config["data"]["output_dir"])
        directories = [
            output_dir / "figures" / "causal_graphs",
            output_dir / "figures" / "prior_knowledge",
            output_dir / "models",
            output_dir / "results",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """
        Load and prepare data for causal analysis.

        Returns:
        pd.DataFrame: Preprocessed data ready for modeling.
        """
        # Load data from Excel file
        data_path = self.config["data"]["input_file"]
        df = pd.read_excel(data_path)
        logger.info(f"Data loaded from {data_path}, shape: {df.shape}")

        # Recompute BFI-TIPI difference inputs from a common scale (see method)
        preprocessing = self.config.get("preprocessing", {})
        if preprocessing.get("rescale_bfi_tipi_difference", False):
            df = self._rescale_bfi_tipi_difference(df)

        # Extract features based on configuration
        input_feats = self.config["features"]["input_features"]
        intermediate_feats = self.config["features"]["intermediate_features"]
        output_feats = self.config["features"]["output_features"]

        self.features = input_feats + intermediate_feats + output_feats
        self.data = df[self.features].copy()

        # Normalize data if specified
        if self.config["model"]["normalize_data"]:
            # Calculate and store statistics before normalization
            self.normalization_stats = pd.DataFrame(
                {
                    "feature": self.data.columns,
                    "mean": self.data.mean(),
                    "std": self.data.std(),
                    "min": self.data.min(),
                    "max": self.data.max(),
                }
            )

            # Apply z-score normalization
            self.data = self.data.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

            # Export normalization statistics to CSV
            output_dir = Path(self.config["data"]["output_dir"]) / "results"
            stats_path = output_dir / "normalization_statistics.csv"
            self.normalization_stats.to_csv(stats_path, index=False)

            logger.info("Data normalized using z-score normalization")
            logger.info(f"Normalization statistics exported to: {stats_path}")

            # Print normalization statistics using the dedicated method
            self.print_normalization_stats()

        logger.info(f"Features selected: {len(self.features)} variables")
        return self.data

    def _rescale_bfi_tipi_difference(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recompute ``Difference BFI-TIPI <trait>`` inputs on a common scale.

        The raw Excel column subtracts BFI (1-10 scale) from TIPI (1-14 scale)
        directly, so the difference mixes two incompatible units. The GNN
        SEMHead, on the other hand, differences a MinMax-normalized BFI
        prediction ([0, 1]) against MinMax-normalized TIPI features ([0, 1]).

        To keep the causal model and the SEMHead on the same input scale, this
        rescales each trait's BFI and TIPI to [0, 1] (MinMax over the declared
        questionnaire range) *before* differencing:

            diff = (BFI - bfi_min) / (bfi_max - bfi_min)
                 - (TIPI - tipi_min) / (tipi_max - tipi_min)

        Args:
            df: Raw dataframe loaded from the input Excel file.

        Returns:
            The dataframe with the difference columns overwritten in place.
        """
        from configs.config import BFI_SCALE, TIPI_SCALE
        from utility.helper import minmax_scale

        cfg = self.config["preprocessing"]
        bfi_min, bfi_max = cfg.get("bfi_scale", list(BFI_SCALE))
        tipi_min, tipi_max = cfg.get("tipi_scale", list(TIPI_SCALE))
        input_feats = self.config["features"]["input_features"]

        recomputed = []
        for feat in input_feats:
            trait = feat.split()[-1]  # "Difference BFI-TIPI O" -> "O"
            bfi_col, tipi_col = f"BFI-{trait}", f"TIPI-{trait}"
            if bfi_col not in df.columns or tipi_col not in df.columns:
                logger.warning(
                    f"Cannot rescale '{feat}': missing '{bfi_col}' or '{tipi_col}'; "
                    f"keeping the original column"
                )
                continue
            bfi_scaled = minmax_scale(df[bfi_col], bfi_min, bfi_max)
            tipi_scaled = minmax_scale(df[tipi_col], tipi_min, tipi_max)
            df[feat] = bfi_scaled - tipi_scaled
            recomputed.append(feat)

        logger.info(
            f"Rescaled BFI-TIPI differences to a common [0, 1] scale before "
            f"differencing: {recomputed}"
        )
        return df

    def _create_prior_knowledge(self) -> np.ndarray:
        """
        Create prior knowledge matrix based on feature groups.

        Returns:
        np.ndarray: Prior knowledge matrix for the model.
        """
        n_variables = len(self.features)
        input_feats = self.config["features"]["input_features"]
        output_feats = self.config["features"]["output_features"]

        # Define exogenous (input) and sink (output) variables
        exogenous_vars = list(range(len(input_feats)))
        sink_vars = list(range(n_variables - len(output_feats), n_variables))

        prior_knowledge = make_prior_knowledge(
            n_variables=n_variables, sink_variables=sink_vars, exogenous_variables=exogenous_vars
        )

        logger.info(
            f"Prior knowledge created: {len(exogenous_vars)} exogenous, {len(sink_vars)} sink variables"
        )
        return prior_knowledge

    def _visualize_prior_knowledge(self, prior_knowledge: np.ndarray) -> None:
        """Visualize prior knowledge graph."""
        if not self.config["visualization"]["save_graphs"]:
            return

        graph = self._make_prior_knowledge_graph(prior_knowledge)
        output_path = (
            Path(self.config["data"]["output_dir"])
            / "figures"
            / "prior_knowledge"
            / "prior_knowledge"
        )

        # Save Graphviz DOT file
        dot_path = str(output_path) + ".dot"
        with open(dot_path, "w") as f:
            f.write(graph.source)
        logger.info(f"Prior knowledge DOT file saved: {dot_path}")

        # Save rendered formats
        for fmt in self.config["visualization"]["formats"]:
            graph.render(str(output_path), format=fmt, cleanup=True)

        logger.info("Prior knowledge graph saved")

    def _make_prior_knowledge_graph(self, prior_knowledge_matrix: np.ndarray) -> graphviz.Digraph:
        """Create graphviz representation of prior knowledge."""
        d = graphviz.Digraph(engine="dot")

        labels = [f"x{i}" for i in range(prior_knowledge_matrix.shape[0])]
        for label in labels:
            d.node(label, label)

        dirs = np.where(prior_knowledge_matrix > 0)
        for to, from_ in zip(dirs[0], dirs[1]):
            d.edge(labels[from_], labels[to])

        dirs = np.where(prior_knowledge_matrix < 0)
        for to, from_ in zip(dirs[0], dirs[1]):
            if to != from_:
                d.edge(labels[from_], labels[to], style="dashed")

        return d

    def fit_model(self) -> None:
        """
        Fit the DirectLiNGAM model to the data.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Create prior knowledge if specified
        self._prior_knowledge = None
        if self.config["model"]["prior_knowledge"]["use_prior"]:
            self._prior_knowledge = self._create_prior_knowledge()
            self._visualize_prior_knowledge(self._prior_knowledge)

        # Initialize and fit model
        algorithm = self.config["model"]["algorithm"]
        self.model = _build_lingam_model(algorithm, self._prior_knowledge)

        self.model.fit(self.data)
        self.adjacency_matrix = self.model.adjacency_matrix_

        logger.info(f"Model fitted using {algorithm}")
        logger.info(f"Causal order: {[int(i) for i in self.model.causal_order_]}")

    def run_bootstrap_analysis(self) -> None:
        """
        Perform bootstrap analysis for robust causal inference.
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit_model() first.")

        bootstrap_config = self.config["model"]["bootstrap"]
        n_sampling = bootstrap_config["n_sampling"]
        n_jobs = bootstrap_config.get("n_jobs", -1)

        self.bootstrap_result = self._run_parallel_bootstrap(n_sampling, n_jobs)
        # save the results as npy
        np.save(
            Path(self.config["data"]["output_dir"]) / "models" / "bootstrap_result.npy",
            self.bootstrap_result,
        )

        # Get causal direction counts
        min_effect = bootstrap_config["min_causal_effect"]
        cdc = self.bootstrap_result.get_causal_direction_counts(
            min_causal_effect=min_effect, split_by_causal_effect_sign=True
        )

        logger.info("Causal direction counts:")
        print_causal_directions(cdc, n_sampling)

        # Calculate median effects with probability filtering
        self._calculate_robust_effects()

    def _run_parallel_bootstrap(self, n_sampling: int, n_jobs: int) -> BootstrapResult:
        """Run the LiNGAM bootstrap across processes with joblib.

        Each worker bootstraps an independent slice of the resamples (seeded
        distinctly for reproducibility) and returns its per-sample adjacency
        and total-effect matrices. The slices are concatenated into a single
        ``BootstrapResult`` that is API-compatible with the serial path.

        Args:
            n_sampling: Total number of bootstrap resamples.
            n_jobs: Number of parallel workers (-1 uses all cores).

        Returns:
            Combined bootstrap result over all resamples.
        """
        from joblib import Parallel, delayed, effective_n_jobs

        algorithm = self.config["model"]["algorithm"]
        prior_knowledge = getattr(self, "_prior_knowledge", None)
        base_seed = int(self.config["model"]["random_seed"])
        data = np.asarray(self.data)

        n_workers = effective_n_jobs(n_jobs)
        chunks = _split_counts(n_sampling, n_workers)

        logger.info(
            f"Running bootstrap analysis with {n_sampling} samples "
            f"across {len(chunks)} worker(s) (n_jobs={n_jobs})..."
        )

        results = Parallel(n_jobs=n_jobs)(
            delayed(_bootstrap_chunk)(data, count, base_seed + i, algorithm, prior_knowledge)
            for i, count in enumerate(chunks)
        )

        adjacency_matrices = np.concatenate([adj for adj, _ in results], axis=0)
        total_effects = np.concatenate([eff for _, eff in results], axis=0)

        return BootstrapResult(adjacency_matrices, total_effects)

    def _calculate_robust_effects(self) -> np.ndarray:
        """Calculate robust causal effects using bootstrap results."""
        bootstrap_config = self.config["model"]["bootstrap"]
        prob_threshold = bootstrap_config["prob_lower_limit"]
        min_effect = bootstrap_config["min_causal_effect"]

        # Get probability matrix
        self.probability_matrix = self.bootstrap_result.get_probabilities(
            min_causal_effect=min_effect
        )

        # Get adjacency matrices from bootstrap
        adjacency_matrices = self.bootstrap_result.adjacency_matrices_

        # Replace zeros with NaN for median calculation
        adjacency_matrices_nan = adjacency_matrices.copy()
        adjacency_matrices_nan[adjacency_matrices_nan == 0.0] = np.nan

        # Calculate median effects
        median_effects = np.nanmedian(adjacency_matrices_nan, axis=0)
        median_effects = np.nan_to_num(median_effects)

        # Apply probability threshold
        significant_mask = np.abs(self.probability_matrix) > prob_threshold
        self.robust_adjacency_matrix = median_effects * significant_mask

        logger.info("Robust causal effects calculated")
        return self.robust_adjacency_matrix

    def visualize_causal_graph(self, use_robust: bool = True) -> None:
        """
        Create and save causal graph visualizations.

        Parameters:
        use_robust (bool): Whether to use robust adjacency matrix from bootstrap.
        """
        if not self.config["visualization"]["save_graphs"]:
            return

        matrix = self.robust_adjacency_matrix if use_robust else self.adjacency_matrix
        if matrix is None:
            logger.warning("No adjacency matrix available for visualization")
            return

        # Create causal graph
        graph = self._make_causal_graph(matrix, self.probability_matrix)

        # Add visual styling for feature groups
        self._style_causal_graph(graph)

        # Save graph
        model_name = self.config["model"]["name"]
        suffix = "_robust" if use_robust else "_direct"
        output_path = (
            Path(self.config["data"]["output_dir"])
            / "figures"
            / "causal_graphs"
            / f"{model_name}{suffix}"
        )

        # Save Graphviz DOT file
        dot_path = str(output_path) + ".dot"
        with open(dot_path, "w") as f:
            f.write(graph.source)
        logger.info(f"Causal graph DOT file saved: {dot_path}")

        # Save rendered formats
        for fmt in self.config["visualization"]["formats"]:
            graph.render(str(output_path), format=fmt, cleanup=True)

        logger.info(f"Causal graph saved: {model_name}{suffix}")

    def _make_causal_graph(
        self, adjacency_matrix: np.ndarray, probability_matrix: np.ndarray
    ) -> graphviz.Digraph:
        """
        Create a causal graph using Graphviz with effect sizes and probabilities.
        """
        d = graphviz.Digraph(engine="dot")
        d.attr("graph", rankdir="TB", splines="true")
        d.attr("node", style="filled", fillcolor="white")

        # Get lower limit for visualization
        lower_limit = self.config["visualization"].get("lower_limit", 0.01)
        if isinstance(lower_limit, str):
            lower_limit = float(lower_limit)

        # Add nodes
        labels = (
            self.features if self.features else [f"x{i}" for i in range(adjacency_matrix.shape[0])]
        )

        for label in labels:
            d.node(label, label)

        # Add edges with both effect size and probability
        n_variables = adjacency_matrix.shape[0]
        for i in range(n_variables):
            for j in range(n_variables):
                coef = adjacency_matrix[i, j]
                prob = abs(probability_matrix[i, j]) if probability_matrix is not None else None

                if abs(coef) > lower_limit:
                    # Format edge label with both coefficient and probability
                    edge_label = (
                        f"{coef:.3f}\\n({int(prob*100)}%)" if prob is not None else f"{coef:.3f}"
                    )

                    d.edge(
                        labels[j],
                        labels[i],
                        label=edge_label,
                        fontsize="14",
                    )

        return d

    def _style_causal_graph(self, graph: graphviz.Digraph) -> None:
        """Add visual styling to causal graph based on feature groups."""
        input_feats = self.config["features"]["input_features"]
        output_feats = self.config["features"]["output_features"]

        # Style input features
        with graph.subgraph(name="cluster_input") as s:
            s.attr(rank="same")
            s.attr(style="filled", color="lightblue", label="Input Features")
            for feat in input_feats:
                s.node(feat, style="filled", fillcolor="lightblue")

        # Style output features
        with graph.subgraph(name="cluster_output") as s:
            s.attr(rank="same")
            s.attr(style="filled", color="lightgreen", label="Output Features")
            for feat in output_feats:
                s.node(feat, style="filled", fillcolor="lightgreen")

    def evaluate_model(self) -> dict[str, float]:
        """
        Evaluate model fit using various metrics.

        Returns:
        Dict[str, float]: Dictionary containing evaluation metrics.
        """
        if self.adjacency_matrix is None:
            raise ValueError("Model not fitted. Call fit_model() first.")

        # Evaluate direct model fit
        direct_fit = evaluate_model_fit(self.adjacency_matrix, self.data)

        # Evaluate robust model fit if available
        robust_fit = None
        if hasattr(self, "robust_adjacency_matrix"):
            robust_fit = evaluate_model_fit(self.robust_adjacency_matrix, self.data)

        evaluation = {"direct_model_fit": direct_fit, "robust_model_fit": robust_fit}

        # Save evaluation results
        if self.config["evaluation"]["save_effects"]:
            results_path = Path(self.config["data"]["output_dir"]) / "results"
            pd.DataFrame([evaluation]).to_csv(results_path / "model_evaluation.csv", index=False)

        logger.info("Model evaluation completed")
        return evaluation

    def save_model(self) -> None:
        """Save trained model and adjacency matrices."""
        if self.adjacency_matrix is None:
            raise ValueError("Model not fitted. Call fit_model() first.")

        output_dir = Path(self.config["data"]["output_dir"]) / "models"
        model_name = self.config["model"]["name"]

        # Save adjacency matrices and causal order
        if self.config["evaluation"]["matrix_format"] == "npy":
            # Save adjacency matrix with causal order and features
            model_data = {
                "adjacency_matrix": self.adjacency_matrix,
                "causal_order": self.model.causal_order_ if self.model is not None else None,
                "features": self.features,
            }
            np.save(output_dir / f"{model_name}_direct_model_data.npy", model_data)

            # Also save adjacency matrix separately for backward compatibility
            np.save(output_dir / f"{model_name}_adjacency_matrix.npy", self.adjacency_matrix)

            # Save causal order separately
            if self.model is not None and hasattr(self.model, "causal_order_"):
                np.save(output_dir / f"{model_name}_causal_order.npy", self.model.causal_order_)

            if hasattr(self, "robust_adjacency_matrix"):
                robust_model_data = {
                    "adjacency_matrix": self.robust_adjacency_matrix,
                    "causal_order": self.model.causal_order_ if self.model is not None else None,
                    "features": self.features,
                }
                np.save(output_dir / f"{model_name}_robust_model_data.npy", robust_model_data)

                # Also save robust adjacency matrix separately for backward compatibility
                np.save(
                    output_dir / f"{model_name}_robust_adjacency_matrix.npy",
                    self.robust_adjacency_matrix,
                )
        else:
            pd.DataFrame(self.adjacency_matrix, index=self.features, columns=self.features).to_csv(
                output_dir / f"{model_name}_adjacency_matrix.csv"
            )
            if hasattr(self, "robust_adjacency_matrix"):
                pd.DataFrame(
                    self.robust_adjacency_matrix, index=self.features, columns=self.features
                ).to_csv(output_dir / f"{model_name}_robust_adjacency_matrix.csv")

            # Save causal order as CSV
            if self.model is not None and hasattr(self.model, "causal_order_"):
                causal_order_df = pd.DataFrame(
                    {
                        "causal_order_index": range(len(self.model.causal_order_)),
                        "feature_index": self.model.causal_order_,
                        "feature_name": (
                            [self.features[i] for i in self.model.causal_order_]
                            if self.features
                            else None
                        ),
                    }
                )
                causal_order_df.to_csv(output_dir / f"{model_name}_causal_order.csv", index=False)

        # Save causal effects if bootstrap was performed
        if self.bootstrap_result is not None:
            min_effect = self.config["model"]["bootstrap"]["min_causal_effect"]
            causal_effects = self.bootstrap_result.get_total_causal_effects(
                min_causal_effect=min_effect
            )

            effects_df = pd.DataFrame(causal_effects)
            effects_df["from"] = effects_df["from"].apply(lambda x: self.features[x])
            effects_df["to"] = effects_df["to"].apply(lambda x: self.features[x])
            effects_df = effects_df.sort_values("effect", ascending=False)
            effects_df.to_csv(output_dir / f"{model_name}_causal_effects.csv", index=False)

        # Save normalization statistics if available
        if hasattr(self, "normalization_stats") and self.normalization_stats is not None:
            stats_path = output_dir / f"{model_name}_normalization_stats.csv"
            self.normalization_stats.to_csv(stats_path, index=False)
            logger.info(f"Normalization statistics saved to {stats_path}")

        logger.info(f"Model saved to {output_dir}")

    def load_model(self, model_path: str | None = None) -> None:
        """
        Load a previously saved model.

        Parameters:
        model_path (str, optional): Path to the saved model. If None, uses config path.
        """
        if model_path is None:
            model_path = self.config["inference"]["model_path"]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Try to load complete model data first
        try:
            model_data = np.load(model_path, allow_pickle=True).item()
            if isinstance(model_data, dict) and "adjacency_matrix" in model_data:
                self.adjacency_matrix = model_data["adjacency_matrix"]

                # Load causal order if available
                if "causal_order" in model_data and model_data["causal_order"] is not None:
                    # Create a mock model object to store causal order
                    if self.model is None:
                        self.model = type("MockModel", (), {})()
                    self.model.causal_order_ = model_data["causal_order"]

                # Load features if available
                if "features" in model_data and model_data["features"] is not None:
                    self.features = model_data["features"]

                logger.info(f"Complete model data loaded from {model_path}")
            else:
                # Fallback to loading just adjacency matrix
                self.adjacency_matrix = model_data
                logger.info(f"Adjacency matrix loaded from {model_path}")
        except (ValueError, AttributeError):
            # Fallback to loading just adjacency matrix
            self.adjacency_matrix = np.load(model_path)
            logger.info(f"Adjacency matrix loaded from {model_path}")

        # Try to load causal order from separate file if not loaded yet
        if self.model is None or not hasattr(self.model, "causal_order_"):
            causal_order_path = (
                str(model_path)
                .replace("_model_data.npy", "_causal_order.npy")
                .replace("_adjacency_matrix.npy", "_causal_order.npy")
            )
            if os.path.exists(causal_order_path):
                try:
                    causal_order = np.load(causal_order_path)
                    if self.model is None:
                        self.model = type("MockModel", (), {})()
                    self.model.causal_order_ = causal_order
                    logger.info(f"Causal order loaded from {causal_order_path}")
                except Exception as e:
                    logger.warning(f"Could not load causal order from {causal_order_path}: {e}")

        # Try to load normalization statistics from separate file
        normalization_stats_path = (
            str(model_path)
            .replace("_model_data.npy", "_normalization_stats.csv")
            .replace("_adjacency_matrix.npy", "_normalization_stats.csv")
        )
        if os.path.exists(normalization_stats_path):
            try:
                self.normalization_stats = pd.read_csv(normalization_stats_path)
                logger.info(f"Normalization statistics loaded from {normalization_stats_path}")
            except Exception as e:
                logger.warning(
                    f"Could not load normalization statistics from {normalization_stats_path}: {e}"
                )

    def predict(self, input_values: dict[str, float]) -> dict[str, float]:
        """
        Perform causal inference given input values.

        Parameters:
        input_values (Dict[str, float]): Dictionary of input feature values.

        Returns:
        Dict[str, float]: Predicted intermediate and output feature values.
        """
        if not self.config["inference"]["enable_inference"]:
            raise ValueError("Inference is disabled in configuration")

        if self.adjacency_matrix is None:
            raise ValueError("No model available for inference. Train or load a model first.")

        # Create input vector
        input_vector = np.zeros(len(self.features))
        for feature, value in input_values.items():
            if feature in self.features:
                idx = self.features.index(feature)
                input_vector[idx] = value
            else:
                logger.warning(f"Feature '{feature}' not found in model features")

        # Propagate through the SEM in causal order so mediated paths
        # (input -> intermediate -> output) contribute to downstream variables
        input_indices = {
            self.features.index(feature) for feature in input_values if feature in self.features
        }
        causal_order = self.get_causal_order()
        order = (
            [int(v) for v in causal_order]
            if causal_order is not None
            else list(range(len(self.features)))
        )

        output_vector = input_vector.copy()
        for v in order:
            if v in input_indices:
                continue
            output_vector[v] = self.adjacency_matrix[v, :] @ output_vector

        # Extract intermediate and output features for predictions
        intermediate_feats = self.config["features"]["intermediate_features"]
        output_feats = self.config["features"]["output_features"]
        target_features = intermediate_feats + output_feats
        predictions = {}

        for feat in target_features:
            if feat in self.features:
                idx = self.features.index(feat)
                predictions[feat] = output_vector[idx]

        return predictions

    def run_complete_analysis(self) -> None:
        """Run the complete causal analysis pipeline."""
        logger.info("Starting complete causal analysis pipeline...")

        # Load and prepare data
        self.load_data()

        # Fit model
        self.fit_model()

        # Run bootstrap analysis
        self.run_bootstrap_analysis()

        # Create visualizations
        self.visualize_causal_graph(use_robust=False)
        self.visualize_causal_graph(use_robust=True)

        # Evaluate model
        self.evaluate_model()

        # Save results
        self.save_model()

        logger.info("Complete causal analysis pipeline finished successfully!")

    def get_causal_order(self) -> np.ndarray | None:
        """
        Get the causal order from the fitted model.

        Returns:
        Optional[np.ndarray]: Causal order array if available, None otherwise.
        """
        if self.model is not None and hasattr(self.model, "causal_order_"):
            return self.model.causal_order_
        return None

    def get_causal_order_features(self) -> list | None:
        """
        Get feature names in causal order.

        Returns:
        Optional[list]: List of feature names in causal order if available, None otherwise.
        """
        causal_order = self.get_causal_order()
        if causal_order is not None and self.features is not None:
            return [self.features[i] for i in causal_order]
        return None

    def print_model_info(self) -> None:
        """Print comprehensive model information including causal order."""
        if self.adjacency_matrix is None:
            logger.info("No model fitted or loaded.")
            return

        logger.info("=== Model Information ===")
        logger.info(f"Number of features: {len(self.features) if self.features else 'Unknown'}")

        if self.features:
            logger.info(f"Features: {self.features}")

        causal_order = self.get_causal_order()
        if causal_order is not None:
            logger.info(f"Causal order (indices): {causal_order}")

            ordered_features = self.get_causal_order_features()
            if ordered_features:
                logger.info(f"Causal order (features): {ordered_features}")
        else:
            logger.info("Causal order: Not available")

        logger.info(f"Adjacency matrix shape: {self.adjacency_matrix.shape}")

        if hasattr(self, "robust_adjacency_matrix") and self.robust_adjacency_matrix is not None:
            logger.info(f"Robust adjacency matrix available: {self.robust_adjacency_matrix.shape}")

        if self.bootstrap_result is not None:
            logger.info("Bootstrap analysis results available")

        if hasattr(self, "normalization_stats") and self.normalization_stats is not None:
            logger.info("Normalization statistics available")
            logger.info(f"Number of normalized features: {len(self.normalization_stats)}")
        else:
            logger.info("Normalization statistics: Not available")

        # Print system of equations for both direct and robust models
        print("\n" + "=" * 60)
        self.print_system_of_equations(use_robust=False)

        if hasattr(self, "robust_adjacency_matrix") and self.robust_adjacency_matrix is not None:
            self.print_system_of_equations(use_robust=True)

    def get_normalization_stats(self) -> pd.DataFrame | None:
        """
        Get normalization statistics if available.

        Returns:
        Optional[pd.DataFrame]: DataFrame with normalization statistics or None if not available.
        """
        return getattr(self, "normalization_stats", None)

    def export_normalization_stats(self, output_path: str | None = None) -> str:
        """
        Export normalization statistics to CSV file.

        Parameters:
        output_path (str, optional): Custom path for output file. If None, uses default path.

        Returns:
        str: Path to the exported file.
        """
        if not hasattr(self, "normalization_stats"):
            raise ValueError(
                "No normalization statistics available. Data must be normalized first."
            )

        if output_path is None:
            output_dir = Path(self.config["data"]["output_dir"]) / "results"
            output_path = output_dir / "normalization_statistics.csv"

        self.normalization_stats.to_csv(output_path, index=False)
        logger.info(f"Normalization statistics exported to: {output_path}")
        return str(output_path)

    def print_normalization_stats(self) -> None:
        """Print normalization statistics in a formatted table."""
        if not hasattr(self, "normalization_stats"):
            logger.warning("No normalization statistics available.")
            return

        print("\nNormalization Statistics:")
        print("=" * 90)
        print(f"{'Feature':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Range':<15}")
        print("-" * 90)

        for _, row in self.normalization_stats.iterrows():
            range_str = f"[{row['min']:.3f}, {row['max']:.3f}]"
            print(
                f"{row['feature']:<25} {row['mean']:<10.3f} {row['std']:<10.3f} "
                f"{row['min']:<10.3f} {row['max']:<10.3f} {range_str:<15}"
            )
        print("=" * 90)

    def print_system_of_equations(self, use_robust: bool = False) -> None:
        """
        Print the system of equations derived from the causal order and adjacency matrix.

        Parameters:
        use_robust (bool): Whether to use robust adjacency matrix from bootstrap analysis.
        """
        if self.adjacency_matrix is None:
            logger.warning("No adjacency matrix available. Train or load a model first.")
            return

        if self.features is None:
            logger.warning("No feature names available.")
            return

        # Choose which matrix to use
        matrix = (
            self.robust_adjacency_matrix
            if use_robust and hasattr(self, "robust_adjacency_matrix")
            else self.adjacency_matrix
        )
        matrix_type = (
            "Robust" if use_robust and hasattr(self, "robust_adjacency_matrix") else "Direct"
        )

        print(f"\n=== {matrix_type} LinGAM System of Equations ===")

        # Print causal order
        causal_order = self.get_causal_order()
        if causal_order is not None:
            # Convert to list if it's a numpy array, otherwise use as is
            causal_order_list = (
                causal_order.tolist() if hasattr(causal_order, "tolist") else list(causal_order)
            )
            print("\nCausal Order (indices):", causal_order_list)
            causal_order_named = self.get_causal_order_features()
            if causal_order_named:
                print("Causal Order (features):", causal_order_named)
        else:
            print("\nCausal Order: Not available (using feature order)")

        print(f"\nSystem of Equations ({matrix_type} LinGAM):")
        print("=" * 80)

        # Use causal order if available, otherwise use feature order
        order_to_use = (
            causal_order if causal_order is not None else list(range(len(self.features)))
        )

        # For each variable in the order
        for position, node_idx in enumerate(order_to_use):
            node_name = self.features[node_idx]
            equation = f"  {node_name} = "

            # Find coefficients of predecessors (parents) in the adjacency matrix
            parents = np.where(np.abs(matrix[node_idx, :]) > 1e-6)[
                0
            ]  # Use small threshold to avoid numerical noise

            if len(parents) == 0:
                equation += "ε_{" + node_name + "} (Exogenous variable)"
            else:
                terms = []
                for parent_idx in parents:
                    parent_name = self.features[parent_idx]
                    coef = matrix[node_idx, parent_idx]

                    # Format coefficient for better readability
                    if abs(coef) < 0.001:
                        coef_str = f"{coef:.6f}"
                    elif abs(coef) < 0.01:
                        coef_str = f"{coef:.5f}"
                    else:
                        coef_str = f"{coef:.4f}"

                    # Add sign handling
                    if coef >= 0 and len(terms) > 0:
                        terms.append(f"+ {coef_str} × {parent_name}")
                    else:
                        terms.append(f"{coef_str} × {parent_name}")

                if terms:
                    equation += " ".join(terms)

                equation += " + ε_{" + node_name + "}"

            print(equation)

        print("=" * 80)

        # Print additional information about errors/noise terms
        print("\nWhere:")
        print("  ε_{variable} represents the error/noise term for each variable")
        print("  × denotes multiplication")
        print("  Coefficients represent the causal effect strength")
        print("  Variables are ordered according to the causal structure")

        # Print model fit information if available
        if hasattr(self, "bootstrap_result") and self.bootstrap_result is not None:
            print(f"\nModel Type: {matrix_type} LinGAM with Bootstrap Analysis")
            if (
                use_robust
                and hasattr(self, "probability_matrix")
                and self.probability_matrix is not None
            ):
                min_prob = self.config["model"]["bootstrap"]["prob_lower_limit"]
                print(f"Only effects with probability > {min_prob} are included")
        else:
            print(f"\nModel Type: {matrix_type} LinGAM")

        print()

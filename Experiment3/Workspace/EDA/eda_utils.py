"""Loading and plotting helpers for the Experiments/DATA/ exploratory analysis.

Mirrors the loading conventions of ../ExpDataAnalyze/preprocess_data.py, but loads
every raw file (including the participants preprocess_data.py excludes from the
modeling pipeline) so the EDA can audit that exclusion decision instead of assuming it.
"""

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "Experiments" / "DATA"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Response files preprocess_data.py drops when building the model's database.xlsx.
EXCLUDED_FROM_MODEL = ["C3", "C15", "C25", "C30", "C31"]


def load_participants() -> pd.DataFrame:
    """Participants.xlsx: demographics + TIPI personality (test 1 and test 2)."""
    return pd.read_excel(DATA_DIR / "Metadata" / "Participants.xlsx")


def load_condition() -> pd.DataFrame:
    """Condition.xlsx: 31 interactions x 35 participants, cell = assigned eHMI condition code."""
    return pd.read_excel(DATA_DIR / "Metadata" / "Condition.xlsx")


def load_ehmi() -> pd.DataFrame:
    """eHMI.xlsx: the 51 eHMI condition codes (50 experimental + 99.T control) and their metadata."""
    return pd.read_excel(DATA_DIR / "Metadata" / "eHMI.xlsx")


def load_all_responses() -> dict[str, pd.DataFrame]:
    """Load every QuestionnaireResponses/C*.xlsx file, keyed by participant code (e.g. 'C0')."""
    files = sorted(
        glob.glob(str(DATA_DIR / "QuestionnaireResponses" / "C*.xlsx")),
        key=lambda p: int(Path(p).stem[1:]),
    )
    return {Path(f).stem: pd.read_excel(f) for f in files}


def responses_long(responses: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-participant response frames into one long table.

    Adds `participant` and `excluded_from_model` columns so downstream analysis can
    compare included vs. excluded participants without a second loading pass.
    """
    frames = []
    for participant, df in responses.items():
        df = df.copy()
        df.insert(0, "participant", participant)
        df.insert(1, "excluded_from_model", participant in EXCLUDED_FROM_MODEL)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column NaN count and percentage, sorted worst-first."""
    n = len(df)
    counts = df.isna().sum()
    return (
        pd.DataFrame({"n_missing": counts, "pct_missing": (counts / n * 100).round(2)})
        .sort_values("pct_missing", ascending=False)
    )


def savefig(fig: plt.Figure, name: str) -> Path:
    """Save a figure to output/figures/<name>.png."""
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def save_table(df: pd.DataFrame, name: str) -> Path:
    """Save a DataFrame to output/files/<name>.csv."""
    files_dir = OUTPUT_DIR / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    path = files_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    return path

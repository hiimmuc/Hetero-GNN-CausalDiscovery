"""Builds the Models-ready database.xlsx / database_similarity.xlsx files from raw
Experiments/DATA/.

Consolidates what the archived Archive/ExpDataAnalyze/preprocess_data.py and the
precomputed audio features in Archive/FeatureProcessing/ produce, but joins
participant response rows to their eHMI condition explicitly via
(TestID, Interaction) -> Condition.xlsx, instead of assuming response rows and
metadata rows line up positionally. Audio-acoustic columns (formants, pitch,
intensity, ...) are dropped: nothing in Models/src reads them, only eHMI
Gender/Content Type/Character and the Label-* trait-intensity scores are
consumed downstream.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "Experiments" / "DATA"
MODELS_DATA_DIR = Path(__file__).resolve().parent.parent / "Models" / "data"

TRAITS = ["O", "C", "E", "A", "N"]
CONTROL_CONDITION = "99.T"

# Response TestIDs preprocess_data.py flags as low quality (see EDA/eda.ipynb
# Section 5: C3 has a non-numeric stray value, C31 has 25/28 cells missing;
# C15/C25/C30 have similar missingness/straight-lining issues).
EXCLUDED_TESTIDS = {"C3", "C15", "C25", "C30", "C31"}

# The two dataset variants Models/src/configs/config.py's SET switches between:
# "35" = the full 35-participant study; "25" = the original 25-session pilot
# cohort (TestID C0-C24), with the same data-quality exclusions applied.
SET_TESTIDS = {
    "35": [f"C{i}" for i in range(35) if f"C{i}" not in EXCLUDED_TESTIDS],
    "25": [f"C{i}" for i in range(25) if f"C{i}" not in EXCLUDED_TESTIDS],
}


def normalize_scale(values, min_val, max_val):
    """Normalize values from [min_val, max_val] to [-1, 1]."""
    return -1 + 2 * (np.asarray(values, dtype=float) - min_val) / (max_val - min_val)


def cosine_similarity(v1, v2):
    v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    return v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def euclidean_distance(v1, v2):
    v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    return np.linalg.norm(v1 - v2)


def load_participants() -> pd.DataFrame:
    """Participants.xlsx, dropping the trailing "Total" summary row and any
    participant missing TIPI scores."""
    df = pd.read_excel(DATA_DIR / "Metadata" / "Participants.xlsx")
    df = df[df["Participant ID"] != "Total"]
    tipi_cols = [f"TIPI {sess} {t}" for sess in (1, 2) for t in TRAITS]
    return df.dropna(subset=tipi_cols).set_index("TestID")


def load_condition() -> pd.DataFrame:
    """Condition.xlsx: 31 interactions x per-participant assigned eHMI condition code."""
    return pd.read_excel(DATA_DIR / "Metadata" / "Condition.xlsx")


def load_ehmi_lookup() -> pd.DataFrame:
    """eHMI_metadata_full.xlsx keyed by Condition code.

    A handful of conditions have duplicate rows (repeated audio takes for the
    same character/content); they carry identical eHMI-level metadata, so keep
    the first and drop the rest.
    """
    cols = [
        "Condition",
        "eHMI Gender",
        "eHMI Content Type",
        "eHMI Character",
        "Label-E",
        "Label-A",
        "Label-C",
        "Label-N",
        "Label-O",
    ]
    df = pd.read_excel(DATA_DIR / "Metadata" / "eHMI_metadata_full.xlsx")[cols]
    return df.drop_duplicates(subset="Condition", keep="first").set_index("Condition")


def load_responses(testids) -> dict:
    """QuestionnaireResponses/C*.xlsx for the given TestIDs, keyed by TestID."""
    return {
        testid: pd.read_excel(DATA_DIR / "QuestionnaireResponses" / f"{testid}.xlsx")
        for testid in testids
    }


def build_database(testids, participants, condition, ehmi_lookup, responses) -> pd.DataFrame:
    """One row per valid (participant, non-control interaction), joined explicitly
    via (TestID, Interaction) -> Condition.xlsx -> eHMI metadata."""
    mos_cols = [f"MOS-X2 Q{i}" for i in range(1, 5)]
    bfi_cols = [f"BFI-{t}" for t in TRAITS]

    rows = []
    for testid in testids:
        participant = participants.loc[testid]
        resp = responses[testid].dropna(subset=mos_cols + bfi_cols)

        for _, r in resp.iterrows():
            interaction = int(r["Interaction"])
            code = condition.loc[interaction - 1, testid]
            if code == CONTROL_CONDITION or code not in ehmi_lookup.index:
                continue
            ehmi = ehmi_lookup.loc[code]

            row = {
                "Participant ID": participant["Participant ID"],
                "Participant Gender": participant["Gender"],
                "Participant nationality": participant["Nationality"],
                "eHMI ID": code,
                "eHMI Gender": ehmi["eHMI Gender"],
                "eHMI Content Type": ehmi["eHMI Content Type"],
                "eHMI Character": ehmi["eHMI Character"],
            }
            for t in TRAITS:
                row[f"TIPI {t}-Before"] = participant[f"TIPI 1 {t}"]
                row[f"TIPI {t}-After"] = participant[f"TIPI 2 {t}"]
                row[f"TIPI-{t}"] = participant[f"TIPI 2 {t}"]
                row[f"Label-{t}"] = ehmi[f"Label-{t}"]
                row[f"BFI-{t}"] = float(r[f"BFI-{t}"])
            for i in range(1, 5):
                row[f"MOS-X2 Q{i}"] = float(r[f"MOS-X2 Q{i}"])
            for i in range(1, 6):
                row[f"SQ{i}"] = float(r[f"SQ {i}"])
            row["Rating score"] = float(r["Overall Score"])

            rows.append(row)

    return pd.DataFrame(rows)


def add_similarity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add BFI-TIPI-Label similarity/distance features consumed by causal discovery."""
    df = df.copy()
    normalized_tipi = pd.DataFrame(
        {t: normalize_scale(df[f"TIPI {t}-After"], 1, 14) for t in TRAITS}, index=df.index
    )
    normalized_bfi = pd.DataFrame(
        {t: normalize_scale(df[f"BFI-{t}"], 1, 10) for t in TRAITS}, index=df.index
    )
    normalized_label = pd.DataFrame(
        {t: normalize_scale(df[f"Label-{t}"], 0, 1) for t in TRAITS}, index=df.index
    )

    df["similarity_tipi_bfi"] = [
        cosine_similarity(normalized_tipi.iloc[i], normalized_bfi.iloc[i]) for i in range(len(df))
    ]
    df["similarity_tipi_label"] = [
        cosine_similarity(normalized_tipi.iloc[i], normalized_label.iloc[i])
        for i in range(len(df))
    ]
    df["similarity_bfi_label"] = [
        cosine_similarity(normalized_bfi.iloc[i], normalized_label.iloc[i]) for i in range(len(df))
    ]
    df["l2_distance_tipi_bfi"] = [
        euclidean_distance(normalized_tipi.iloc[i], normalized_bfi.iloc[i]) for i in range(len(df))
    ]

    for t in TRAITS:
        diff = normalized_bfi[t] - normalized_tipi[t]
        df[f"Difference BFI-TIPI {t}"] = diff
        df[f"Difference BFI-TIPI {t} (abs)"] = np.abs(diff)

    mean_rating = df["Rating score"].mean()
    df["Preferred Type"] = np.where(df["Rating score"] >= mean_rating, "High", "Low")
    return df


def build_set(set_name: str) -> pd.DataFrame:
    """Build and save database.xlsx / database_similarity.xlsx for one SET variant."""
    testids = SET_TESTIDS[set_name]
    print(f"\n=== SET {set_name}: {len(testids)} participants ===")

    participants = load_participants()
    condition = load_condition()
    ehmi_lookup = load_ehmi_lookup()
    responses = load_responses(testids)

    database = build_database(testids, participants, condition, ehmi_lookup, responses)
    database_similarity = add_similarity_features(database)

    out_dir = MODELS_DATA_DIR / set_name
    out_dir.mkdir(parents=True, exist_ok=True)
    database.to_excel(out_dir / "database.xlsx", index=False)
    database_similarity.to_excel(out_dir / "database_similarity.xlsx", index=False)

    print(f"database.xlsx: {database.shape} -> {out_dir / 'database.xlsx'}")
    print(f"database_similarity.xlsx: {database_similarity.shape}")
    return database_similarity


def main():
    for set_name in ("35", "25"):
        build_set(set_name)


if __name__ == "__main__":
    main()

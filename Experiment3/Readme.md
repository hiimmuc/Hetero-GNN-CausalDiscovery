# Experiment 3

eHMI (external Human-Machine Interface) voice study. This repository holds the
experimental materials and the data-analysis and modeling workspace for an
experiment on how synthesized eHMI voices are perceived: their rating, the
personality attributed to them, their perceived voice quality, and the
listener's subjective feeling. The modeling workspace pairs a graph
recommender with a causal-discovery pipeline to relate listener personality to
these perceptions.

## Directory Structure

```text
Experiment3/
├── Experiments/                       # Experimental setup and collected data
│   ├── 2025_Nam_eHMI-Voices-Contents/ # eHMI voice stimuli and design materials
│   └── DATA/                          # Raw collected experimental data
└── Workspace/                         # Analysis, processing, and modeling tools
    ├── EDA/                           # Exploratory analysis of the raw DATA/
    ├── FeatureEngineering/            # Builds the Models-ready database files
    ├── Models/                        # GHCF recommender + causal discovery
    ├── Archive/                       # Superseded preprocessing/feature-extraction work
    └── Docs/                          # Documentation and research materials
```

## Experiments

Materials and raw data from running the study.

- **`2025_Nam_eHMI-Voices-Contents/`** — the eHMI voice stimuli, content
  designs, and presentation materials used in the experiment.
- **`DATA/`** — the raw data collected from participants (survey responses and
  per-trial measurements).

## Workspace

### EDA/

Exploratory analysis of the raw `Experiments/DATA/` (participants, condition
assignment, eHMI metadata, questionnaire responses) — data quality, exclusion
audit, and distribution checks on the inputs `FeatureEngineering/` consumes.

- `eda.ipynb` — the main narrative notebook.
- `eda_utils.py` — loading/plotting helpers reused by the notebook.
- `output/` — generated figures and summary tables.

### FeatureEngineering/

Builds the two Models-ready files, `database.xlsx` and
`database_similarity.xlsx`, directly from raw `Experiments/DATA/` (participant
TIPI/BFI, condition assignment, eHMI metadata + audio labels, questionnaire
responses), for both dataset variants (`SET = "35"` or `"25"` in
`Models/src/configs/config.py`).

- `feature_engineering.py` — the full pipeline: joins each response row to its
  eHMI condition explicitly via `(TestID, Interaction) -> Condition.xlsx`
  (rather than assuming response and metadata rows line up positionally, as
  the archived `Archive/ExpDataAnalyze/preprocess_data.py` did), drops the raw
  audio-acoustic columns nothing downstream reads, and adds the BFI-TIPI-Label
  similarity/difference features the causal-discovery pipeline needs.
- Run with `python feature_engineering.py` from this directory; it writes
  straight into `Models/data/35/` and `Models/data/25/`. Re-run it whenever
  the raw `Experiments/DATA/` changes — note this also invalidates any
  `Models/data/<SET>/processed_*/` tensor cache, which must be deleted so the
  next training run rebuilds from the new database.

### Models/

The machine-learning workspace, combining a recommender with causal analysis.
See `Workspace/Models/src/README.md` for full details.

- **Recommender** — a Graph Heterogeneous Collaborative Filtering (GHCF) model
  that jointly predicts voice rating, perceived personality (BFI), perceived
  voice quality, and subjective feeling over a heterogeneous user-item graph.
- **Causal discovery** — a LiNGAM pipeline that learns a causal DAG from
  personality differences through perceptual mediators to the final rating. The
  discovered structure can be embedded into the recommender as a differentiable
  structural-equation head.
- **Subdirectories** — `src/` (source code), `data/` (input spreadsheets and
  cached tensors), `checkpoints/` (trained models and logs),
  `causal_outputs/` (causal artifacts and figures), and `archive/` (legacy data
  and the original GHCF reference implementation).

### Archive/

Superseded preprocessing and audio-feature-extraction work, kept for
reference only — nothing here is read by `EDA/`, `FeatureEngineering/`, or
`Models/src`.

- **`ExpDataAnalyze/`** — the original `preprocess_data.py`/`dataProcessing.ipynb`
  pipeline that `FeatureEngineering/feature_engineering.py` now replaces (it
  joined response rows to conditions positionally rather than by key), plus
  downstream statistical analysis (ANOVA, LinGAM, regression) of the resulting
  database that duplicates or predates `Models/`'s causal-discovery stage.
- **`FeatureProcessing/`** — the MFCC/mel-spectrogram/embedding audio-feature
  extraction and the Big-5-label modeling that produced the `Label-*` columns
  now already baked into `Experiments/DATA/Metadata/eHMI_metadata_*.xlsx`.
  Re-run this only if the label-generation step itself needs to change.

### Docs/

Documentation and research materials: model prompts, experimental protocols,
and methodology notes.

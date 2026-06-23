# Experiment 3

## Directory Structure

```text
Experiment3/
├── Experiments/           # Experimental setup and data collection
│   ├── 2025_Nam_eHMI-Voices-Contents/  # eHMI design
│   └── DATA/              # Collected experimental data
└── Workspace/             # Analysis and processing tools
    ├── ExpDataAnalyze/    # Data processing and analysis (ignore)
    ├── FeatureProcessing/ # Audio feature extraction (ignore)
    ├── Models/            # Machine learning models (GHCF and Causal Discovery)
    └── Docs/              # Documentation
```

## Workspace Directory

### ExpDataAnalyze/

**Purpose**: data processing and statistical analysis

- **Scripts**:
    - `preprocess_data.py`: data preprocessing
    - `dataProcessing.ipynb`: analysis and visualization
- **Data Organization**: Structured by experimental phases (data/, data_25/)
- **Outputs**: Statistical analysis results and visualizations

### FeatureProcessing/

**Purpose**: Advanced audio feature extraction and analysis

- **Feature Extraction Scripts**:
    - `feature_extraction_mfcc.py`: Mel-frequency cepstral coefficients
    - `feature_extraction_mel.py`: Mel spectrogram features
    - `feature_extraction_emb.py`: Deep learning embeddings
- **Analysis Notebooks**:
    - `feat_extract.ipynb`: feature extraction
    - `feat_analyze.ipynb`: Feature analysis and modeling

### Models/

**Purpose**: Recommendation model and Causal Discovery

- **Checkpoints**: Trained model parameters and states
- **Causal Analysis**: Cause-effect relationship modeling
- **Source Code**: Model architectures and training scripts
- **Outputs**: Prediction results and model performance metrics

### Docs/

**Purpose**: Documentation and research materials

- Model prompts and experimental protocols
- Research documentation and methodology

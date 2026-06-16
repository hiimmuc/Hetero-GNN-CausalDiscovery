# Experiment 3

## Directory Structure

```text
Experiment3/
├── Experiments/           # Experimental setup and data collection
│   ├── 2025_Nam_eHMI-Voices-Contents/  # eHMI design
│   └── DATA/              # Collected experimental data
└── Workspace/             # Analysis and processing tools
    ├── ExpDataAnalyze/    # Data processing and analysis
    ├── FeatureProcessing/ # Audio feature extraction
    ├── Models/            # Machine learning models
    ├── VoiceGeneration/   # Speech synthesis
    ├── utils/             # Utility functions
    ├── Archive/           # Historical data
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

### VoiceGeneration/

**Purpose**: Text-to-speech synthesis and audio processing

- **Scripts**:
    - `tts.py`: Azure-based speech synthesis
    - `audio2video.py`: Audio-to-video conversion
- **Notebooks**:
    - `speech_tuning.ipynb`: Voice parameter optimization

### utils/

**Purpose**: Shared utility functions and tools

- `augment.py`: Audio data augmentation
- `pca_utils.py`: Principal component analysis utilities
- `praat_feature_extraction.py`: Advanced acoustic feature extraction

### Archive/

**Purpose**: Historical data and deprecated models

- Previous experiment versions and legacy code
- Model backups and experimental iterations

### Docs/

**Purpose**: Documentation and research materials

- Model prompts and experimental protocols
- Research documentation and methodology

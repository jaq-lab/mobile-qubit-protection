# Data Processing Notebooks

This directory contains notebooks that process raw experimental data and convert it into processed data files for figure generation.

## Purpose

Each notebook in this directory:
1. **Loads raw experimental data** from the `/data` folder using UUIDs (via `common_scripts.raw_data_loader`)
2. **Processes the data** through analysis steps such as:
   - Fitting decay curves (stretched exponentials, etc.)
   - Extracting physical parameters (T2, coherence times, exponents, frequencies)
   - Envelope extraction and signal processing
   - Correlation function calculations
   - Spectrum analysis
3. **Saves processed data** to the `/processed_data` folder using `common_scripts.data_saver`

## Notebooks

### Main Text Figures

- **`fig23_process_stationary.ipynb`**
  - Processes stationary qubit measurements
  - Extracts T2 coherence times and decay parameters
  - Output: `/processed_data/fig23/`

- **`fig4_process_shuttling.ipynb`**
  - Processes shuttling experiments (high and low field)
  - Fits spatial correlation functions
  - Extracts shuttling-dependent coherence times
  - Output: `/processed_data/fig4/`

- **`fig5b_process_LZSM.ipynb`**
  - Processes Landau-Zener-Stückelberg-Majorana (LZSM) interference data
  - Analyzes driven qubit spectrum
  - Output: `/processed_data/fig5/fig5b_floquet_spectrum.*`

- **`fig5c_process_Rabi.ipynb`**
  - Processes Rabi oscillation measurements
  - Extracts driven qubit coherence parameters
  - Output: `/processed_data/fig5/fig5c_driven_coherence.*`

### Appendix Numerical Simulations

- **`appendix_B_two_point_DC.ipynb`**
  - Numerical simulations for Appendix B
  - Two-point correlation DC model calculations
  - Output: `/processed_data/` (appendix-specific)

- **`appendix_D_coarse_model.ipynb`**
  - Numerical simulations for Appendix D
  - Coarse-grained model calculations
  - Output: `/processed_data/` (appendix-specific)

- **`appendix_G_LZSM.ipynb`**
  - Numerical simulations for Appendix G
  - LZSM model calculations and comparisons
  - Output: `/processed_data/` (appendix-specific)

## Output Format

All processed data is saved to `/processed_data/` with the following structure:

```
processed_data/
├── fig23/
│   ├── fig23_high_75mV.pkl      # Processed data
│   ├── fig23_high_75mV.json     # Metadata
│   ├── fig23_low.pkl
│   ├── fig23_low.json
│   └── fig23_latest.pkl          # Symlink to latest version
├── fig4/
│   ├── shuttling_high.pkl
│   ├── shuttling_high.json
│   ├── shuttling_low.pkl
│   ├── shuttling_low.json
│   └── fig4_latest.pkl
└── fig5/
    ├── fig5b_floquet_spectrum.pkl
    ├── fig5b_floquet_spectrum.json
    ├── fig5c_driven_coherence.pkl
    ├── fig5c_driven_coherence.json
    └── fig5_latest.pkl
```

Each processed data file includes:
- **PKL files**: NumPy arrays, dictionaries, and processed data structures
- **JSON files**: Metadata including:
  - UUIDs of source data
  - Processing parameters
  - Fit results and extracted parameters
  - Timestamps and version information

## Usage

### Running a Processing Notebook

1. Open the desired notebook (e.g., `fig4_process_shuttling.ipynb`)
2. Ensure the required UUIDs are specified in the notebook
3. Run all cells to:
   - Load raw data
   - Process and analyze
   - Save results to `/processed_data/`
4. Verify output files in the corresponding `/processed_data/figX/` directory

### Common Functions

All notebooks use shared utilities from `common_scripts/`:

```python
from common_scripts.raw_data_loader import load_raw_data_by_uuid
from common_scripts.data_processor import (
    process_shuttling_dataset,
    fit_stretched_exponential,
    extract_envelope,
    # ... other processing functions
)
from common_scripts.data_saver import save_figure_data
```

## Dependencies

- `common_scripts/raw_data_loader.py` - Loading raw HDF5 data
- `common_scripts/data_processor.py` - Data processing and fitting functions
- `common_scripts/data_saver.py` - Saving processed data with versioning
- `common_scripts/correlation_fun.py` - Correlation function calculations
- `common_scripts/floquet_system_st.py` - Floquet analysis (for driven systems)

## Notes

- Processed data files are versioned and timestamped
- The `*_latest.pkl` symlinks always point to the most recent version
- Metadata JSON files provide full traceability back to source UUIDs
- If reprocessing is needed, simply re-run the notebook (it will create new timestamped files)

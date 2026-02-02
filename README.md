
# Coherence Protection for Mobile Spin Qubits in Silicon

<div align="center">

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10834811.svg)](https://doi.org/10.5281/zenodo.10834811)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/your-username/mobile-qubit-protection/graphs/commit-activity)

**Official repository for the data analysis pipeline associated with the manuscript.**

</div>

---

## 📖 About The Project

This repository contains the complete data analysis pipeline used to demonstrate systematic noise mitigation during spin shuttling in a linear $^{28}$Si/SiGe quantum dot device.

Mobile spin qubit architectures promise flexible connectivity for quantum error correction, but preserving coherence during transport is a major challenge. This codebase supports the findings that **noise mitigation strategies**—specifically micromagnet design optimization, motional narrowing, and dressed-state driving—can extend the coherence time of mobile qubits by nearly an order of magnitude, reaching over $30~\mu s$.

The pipeline handles raw HDF5 experimental data, performs curve fitting (Ramsey, Echo, Rabi), extracts coherence parameters ($T_2^*$, $\alpha$), and generates the publication figures.

## 👥 Authors

**J. A. Krzywda**$^{1,2}$, **Y. Matsumoto**$^1$, **M. De Smet**$^1$, **L. Tryputen**$^1$, **S.L. de Snoo**$^1$, **S.V. Amitonov**$^1$, **E. van Nieuwenburg**$^2$, **G. Scappucci**$^1$, and **L.M.K. Vandersypen**$^1$.

$^1$ *QuTech and Kavli Institute of Nanoscience, Delft University of Technology, The Netherlands* $^2$ *Applied Quantum Algorithms, Leiden University, The Netherlands*

## 📂 Repository Structure

```text
mobile-qubit-protection/
├── code/
│   ├── common_scripts/          # Shared utilities and experimental infrastructure
│   ├── processed_data/          # Processed data files (JSON/PKL format)
│   ├── figures/                 # Generated publication figures (PDF)
│   ├── get_processed_data/      # Stage 1: Notebooks for data processing
│   └── get_plots/               # Stage 2: Notebooks for figure generation
└── README.md

```

## 🚀 Getting Started

### Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/your-username/mobile-qubit-protection.git](https://github.com/your-username/mobile-qubit-protection.git)
cd mobile-qubit-protection

```


2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


4. **Start Jupyter:**
```bash
jupyter notebook

```



### 📦 Downloading Raw Data

**Note:** Raw experimental data is not hosted on GitHub due to size constraints.

1. Visit the **Zenodo Repository**: [10.5281/zenodo.10834811](https://doi.org/10.5281/zenodo.10834811).
2. Download the raw data archive.
3. Extract the data. The notebooks utilize `raw_data_loader.py` which expects the path to these HDF5 files. You may need to update the path in `common_scripts` depending on where you extract the files.

## ⚙️ Data Pipeline Overview

The analysis follows a linear two-stage pipeline:

```text
┌─────────────────────────────────────┐
│ 1. RAW EXPERIMENTAL DATA (Zenodo)   │
│    - HDF5 files organized by UUID   │
│    - Loaded via: raw_data_loader.py │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 2. PROCESSING (get_processed_data/) │
│    - Fit decay curves (Ramsey/Echo) │
│    - Extract T2* and Alpha          │
│    - Saves .pkl files to processed/ │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 3. PLOTTING (get_plots/)            │
│    - Load .pkl processed data       │
│    - Generate PDF figures           │
│    - Saved to: code/figures/        │
└─────────────────────────────────────┘

```

### Stage 1: Data Processing

Located in `code/get_processed_data/`. These notebooks require the raw HDF5 files.

* **`fig4_process_shuttling.ipynb`**: Analyzes coherence during periodic shuttling to demonstrate motional narrowing.
* **`fig5c_process_Rabi.ipynb`**: Extracts Rabi decay times () for the dressed-state shuttling protocol.
* **`appendix_*.ipynb`**: Contains numerical simulations for the theoretical models (e.g., Two-point DC model, Coarse-grained model).

### Stage 2: Figure Generation

Located in `code/get_plots/`. These notebooks utilize the `.pkl` files generated in Stage 1.

* **`main_text_figures.ipynb`**: Reproduces Figures 2, 3, 4, and 5 from the manuscript.
* **`misc_figures.ipynb`**: Generates supplementary data plots.

## 📚 Citation

If you use this code or data in your research, please cite:

```bibtex
@article{Krzywda2026Protection,
  title = {Coherence Protection for Mobile Spin Qubits in Silicon},
  author = {Krzywda, J. A. and Matsumoto, Y. and De Smet, M. and Tryputen, L. and de Snoo, S.L. and Amitonov, S.V. and van Nieuwenburg, E. and Scappucci, G. and Vandersypen, L.M.K.},
  journal = {arXiv preprint},
  year = {2026},
  url = {[https://doi.org/10.5281/zenodo.10834811](https://doi.org/10.5281/zenodo.10834811)}
}

```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

```

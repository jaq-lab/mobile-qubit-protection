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

[cite_start]This repository contains the complete data analysis pipeline used to demonstrate systematic noise mitigation during spin shuttling in a linear 28Si/SiGe quantum dot device[cite: 9].

[cite_start]Mobile spin qubit architectures promise flexible connectivity for quantum error correction, but preserving coherence during transport is a major challenge[cite: 7]. [cite_start]This codebase supports the findings that **noise mitigation strategies**—specifically micromagnet design optimization, motional narrowing, and dressed-state driving—can extend the coherence time of mobile qubits by nearly an order of magnitude, reaching over $30~\mu s$[cite: 10, 11].

The pipeline handles raw HDF5 experimental data, performs curve fitting (Ramsey, Echo, Rabi), extracts coherence parameters ($T_2^*$, $\alpha$), and generates the publication figures.

## 👥 Authors

**J. A. Krzywda**$^{1,2}$, **Y. Matsumoto**$^1$, **M. De Smet**$^1$, **L. Tryputen**$^1$, **S.L. de Snoo**$^1$, **S.V. Amitonov**$^1$, **E. van Nieuwenburg**$^2$, **G. Scappucci**$^1$, and **L.M.K. [cite_start]Vandersypen**$^1$[cite: 2, 3].

[cite_start]$^1$ *QuTech and Kavli Institute of Nanoscience, Delft University of Technology, The Netherlands* [cite: 4]
[cite_start]$^2$ *Applied Quantum Algorithms, Leiden University, The Netherlands* [cite: 3]

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

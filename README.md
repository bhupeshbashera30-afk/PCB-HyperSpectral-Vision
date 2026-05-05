# ⬡ PCB HyperSpectral Vision — Component Classification

<div align="center">

**Advanced PCB component classification using GaborMamba and CNN2D_AttGCN deep learning models on hyperspectral imagery.**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 📋 Overview

This project implements a **hyperspectral imaging (HSI)** pipeline for automated **PCB (Printed Circuit Board) component classification**. It leverages two state-of-the-art deep learning architectures to classify pixel regions of PCB images into four component categories:

| Class | Label | Description |
|:-----:|:------|:------------|
| 0 | **Others** | Background substrate, traces, solder mask, non-component regions |
| 1 | **IC** | Integrated Circuits — QFP, BGA, SOP and other semiconductor packages |
| 2 | **Capacitor** | Ceramic, electrolytic, and tantalum capacitors |
| 3 | **Connector** | Pin headers, edge connectors, USB ports, and I/O interfaces |

---

## 🧠 Model Architectures

### 1. GaborMamba

A Gabor-enhanced temporal encoder that combines fixed-orientation Gabor filters with a dilated Conv1D backbone for spectral-spatial feature extraction.

| Property | Value |
|:---------|:------|
| Parameters | 221,774 |
| Gabor Filters | 6 orientations |
| State Dimension | 128 |
| Dropout | 0.4 |

**Pipeline:** `GaborLayer → TokenGen → FeatureGate → Conv1D Encoder → BatchNorm → Classifier`

### 2. CNN2D_AttGCN

A dual-branch architecture fusing CNN2D spatial features with Contextual Transformer (CoT) Attention and Graph Convolutional Network (GloRe) reasoning.

| Property | Value |
|:---------|:------|
| Alpha Fusion | 0.50 |
| Graph Engine | GloRe GCN |
| Attention | CoT (Contextual Transformer) |
| Dropout | 0.3 – 0.4 |

**Pipeline:** `Conv2D ×4 → CoT Attention → GloRe GCN → GAP → α-Fusion → Classifier`

---

## ⚙️ Processing Pipeline

```
┌─────────────┐    ┌──────────┐    ┌────────────┐    ┌─────────────────┐    ┌───────────────┐
│  HSI Capture │───▶│ PCA → 15D│───▶│ 8×8 Patches│───▶│ Model Inference │───▶│ Component Map │
│ 224 bands    │    │ Reduction│    │ Extraction  │    │ GaborMamba /    │    │ Pixel-level   │
│              │    │          │    │             │    │ CNN2D_AttGCN    │    │ Classification│
└─────────────┘    └──────────┘    └────────────┘    └─────────────────┘    └───────────────┘
```

1. **HSI Capture** — 224-band hyperspectral imaging cube acquisition
2. **PCA Reduction** — Dimensionality reduction from 224 → 15 principal components
3. **Patch Extraction** — 8×8 spatial patch generation for local context
4. **Model Inference** — Classification via GaborMamba and/or CNN2D_AttGCN
5. **Component Map** — Pixel-level classification output with labeled regions

---

## 📁 Project Structure

```
Minor/
├── PCB_HyperSpectral_Imaging.ipynb   # Main training & evaluation notebook
├── README.md                          # This file
├── Test Data/
│   └── PCB.jpg                        # Sample test PCB image
└── webapp/
    ├── index.html                     # Frontend — main page
    ├── style.css                      # Styling with glassmorphism & animations
    ├── script.js                      # Client-side logic & API interaction
    ├── output_gabormamba.png           # Sample GaborMamba output
    ├── output_cnn2d.png               # Sample CNN2D_AttGCN output
    └── output_result.jpg              # Sample combined result
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- NumPy, scikit-learn, matplotlib

### Running the Web App

The web application provides an interactive interface for uploading PCB images and running classification:

```bash
# Navigate to the webapp directory
cd webapp

# Serve with any static file server
npx serve .
```

Then open [http://localhost:3000](http://localhost:3000) in your browser.

### Training (Jupyter Notebook)

Open `PCB_HyperSpectral_Imaging.ipynb` in Jupyter to:
- Preprocess hyperspectral data with PCA
- Train GaborMamba and CNN2D_AttGCN models
- Evaluate classification accuracy
- Export model weights for the web backend

---

## 🖥️ Web Interface Features

- **Drag & drop** PCB image upload
- **Dual model selection** — run GaborMamba, CNN2D_AttGCN, or both
- **Real-time progress** tracking during inference
- **Visual results** — classification heatmaps and per-class component maps
- **Modern UI** — glassmorphism design with smooth animations

---

## 📊 Results

The system produces pixel-level classification maps showing detected components overlaid on the original PCB image. Each model generates its own output, allowing comparative analysis between the two architectures.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**PCB HyperSpectral Vision** · Built with GaborMamba & CNN2D_AttGCN

</div>

# ⬡ PCB HyperSpectral Vision & Recycling-Aware Deep Compression

<div align="center">

**State-of-the-Art PCB Component Classification, Saliency-Guided Deep Compressed Sensing (SSANet, IEEE GRSL 2025), and Automated E-Waste Material Recovery System.**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![IEEE GRSL](https://img.shields.io/badge/IEEE%20GRSL-2025-00629B?style=for-the-badge)](https://doi.org/10.1109/LGRS.2025.3624612)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 📋 Overview

This repository provides an **industrial-grade hyperspectral imaging (HSI) pipeline** for automated **Printed Circuit Board (PCB) component classification and recycling-aware deep compression**. 

Designed for high-speed conveyor belts and pushbroom line-scan cameras (400–1000 nm, 224 spectral bands), the system integrates:
1. **SSANet (IEEE GRSL 2025)**: A multiscale spatial-spectral attention network with $3\times 1$ non-square convolutions that compresses massive line-scan data by **up to 100× (1% sampling rate)** while eliminating patch-stitching grid seam artifacts.
2. **Saliency-Guided Adaptive Sampling**: Employs an attention engine to compress FR4 substrate at 1% rate (saving 99% data) while preserving high-priority components at 10%–20% rate.
3. **Dual-Backbone Classification**: **GaborMamba** (Gabor filter bank + dilated Conv1D) and **CNN2D_AttGCN** (CoT Attention + GloRe Graph Convolution).
4. **Reconstruction Quality Audit**: Mathematical verification of reconstructed cubes using **PSNR**, **SSIM**, and **SAM (Spectral Angle Mapper)**.
5. **Component Inventory & Segmentation Analytics**: Real-time pixel-level identification of ICs, Capacitors, and Connectors with >99.8% accuracy.

---

## 🏗️ System Architecture Flowchart

```
                       ┌────────────────────────┐
                       │      Image Upload      │
                       └───────────┬────────────┘
                                   │
                       ┌───────────▼────────────┐
                       │   PCB Data Ingestion   │ (Raw 224-Band Cube, 400-1000 nm)
                       └───────────┬────────────┘
                                   │
                       ┌───────────▼────────────┐
                       │Calibration & Preprocess│ (Dark/White Radiometric Calibration)
                       └───────────┬────────────┘
                                   │
                       ┌───────────▼────────────┐
                       │ PCA / Channel Attn (CA)│ (224 Bands -> 15 Components)
                       └───────────┬────────────┘
                                   │
                       ┌───────────▼────────────┐
                       │ Spatial Patch Extractor│ (8x8 Context Window Slicing, WS=8)
                       └───────────┬────────────┘
                                   │
                       ┌───────────▼────────────┐
                       │Spatio-Spectral Backbone│ (GaborMamba / CNN2D_AttGCN)
                       └───────────┬────────────┘
                                   │
                       ┌───────────▼────────────┐
                       │Component Saliency Eng. │ (Component vs. Background Masks)
                       └─────┬────────────┬─────┘
                             │            │
            ┌────────────────┘            └────────────────┐
            ▼                                              ▼
┌───────────────────────────────┐              ┌───────────────────────────────┐
│ Important Regions (Low Comp.) │              │Unimportant Regions (High Comp)│
│ • ICs, Capacitors, Connectors │              │ • FR4 Substrate, Empty Traces │
│ • SSANet 10%–20% Rate         │              │ • SSANet 1% Rate (100x Comp.) │
└───────────────┬───────────────┘              └───────────────┬───────────────┘
                │                                              │
                └──────────────────────┬───────────────────────┘
                                       │
                       ┌───────────────▼───────────────┐
                       │ Compressed Latent Rep. z = E(X)│ (0.30M Params, 0.036 GFLOPs)
                       └───────┬───────────────┬───────┘
                               │               │
       ┌───────────────────────┘               └───────────────────────┐
       ▼                                                               ▼
┌───────────────────────────────┐              ┌───────────────────────────────┐
│  Image Synthesis Decoder D(z) │              │  Component Classification Head│
│ • 16 SSFE Blocks with Dense   │              │ • Softmax: Others, IC,        │
│   Residuals (MRDF) + MMF      │              │   Capacitor, Connector        │
│ • Eliminates Grid Seams       │              │ • High Accuracy on Compressed │
└───────────────┬───────────────┘              └───────────────┬───────────────┘
                │                                              │
┌───────────────▼───────────────┐              ┌───────────────▼───────────────┐
│  Reconstruction Quality Audit │              │ Component Inventory Analytics │
│ • PSNR (>51 dB)               │              │ • Spatial Area Footprint      │
│ • SSIM (>0.99)                │              │ • Component Count per Class   │
│ • SAM (<0.26 deg Distortion)  │              │ • Zero Boundary Seam Artifacts│
└───────────────┬───────────────┘              └───────────────┬───────────────┘
                │                                              │
                └──────────────────────┬───────────────────────┘
                                       │
                       ┌───────────────▼───────────────┐
                       │ Live Labeled Component Map &  │
                       │ Analytics Dashboard           │
                       └───────────────────────────────┘
```

---

## 🧠 Core Technologies & Innovations

### 1. SSANet Compressed Sensing (IEEE GRSL 2025)
* **MMF Module**: Fuses **Multiscale Spatial Attention (SA)** ($3\times3, 5\times5, 7\times7$ convolutions) with **Channel Attention (CA)** (spatial hybrid pooling + MLP) through learnable gating $\lambda$.
* **3×1 Non-Square Convolutions**: Specifically designed for line-scan pushbroom cameras to compress narrow stripes without spatial collapse.
* **Fidelity**: Achieves $\text{SAM} \le 1.62^\circ$ and $\text{PSNR} > 35.6\text{ dB}$ at extreme **1% sampling rate** (100× data reduction).

### 2. GaborMamba
* Gabor filter bank (6 fixed orientations) + Token generation + Causal dilated Conv1D temporal backbone.
* Total parameters: 221,774.

### 3. CNN2D_AttGCN
* Dual-branch architecture fusing 4-layer CNN2D spatial features with Contextual Transformer (CoT) attention and GloRe Graph Convolution.

---

## 🚀 Quick Start

### 1. Run the End-to-End Pipeline CLI
Execute the full pipeline on sample PCB imagery:

```bash
# Run with Saliency-Guided Adaptive Rate (default)
python run_pipeline.py --image "Test Data/PCB.jpg" --rate adaptive

# Run with 1% Extreme Compressed Sensing (100x compression)
python run_pipeline.py --image "Test Data/PCB.jpg" --rate 1%
```

**Output Summary:**
```text
[+] PIPELINE SUMMARY
    - Effective Rate       : 6.96% (14.4x compression)
    - Bandwidth Savings    : 93.04%
    - Reconstruction PSNR  : 51.16 dB
    - Spectral Angle (SAM) : 0.251 deg (PASS < 1.62 deg)
    - Grid Artifacts       : ELIMINATED (MMF Multi-Scale Attention Active)
    - Component Counts     : IC=39, Caps=579, Connectors=1356
    - Substrate Isolation  : 99.9% FR4 accuracy (74.8% board surface)
```

### 2. Launch the Interactive Web Dashboard
```bash
# Serve the webapp
cd webapp
python -m http.server 8080
```
Open **[http://localhost:8080](http://localhost:8080)** to explore:
* Interactive flow diagram navigator
* Variable sampling rate sliders & Saliency engine preview
* Before / After artifact elimination viewer (SSANet vs. Baseline DCSN grid seams)
* Live telemetry with component inventory breakdown and classification metrics.

---

## 📁 Repository Structure

```text
Major/
├── FLOW_DIAGRAM.md                                 # Detailed flowchart documentation
├── pcb_project_flow_diagram.png                    # Complete visual flow architecture
├── PCB_HyperSpectral_Vision_Complete_Defense_Report.pdf  # Comprehensive 8-page Defense Report & Viva Guide
├── README.md                                       # Main project guide
├── run_pipeline.py                                 # End-to-end pipeline runner
├── PCB_HyperSpectral_Imaging.ipynb                 # Original training & evaluation notebook
├── PCB_HyperSpectral_Imaging_SSANet_Test.ipynb     # SSANet validation & comparative evaluation notebook
├── build_complete_defense_report.py                # ReportLab script compiling complete defense report
├── models/
│   ├── ssanet_compression.py                       # SSANet architecture, MMF, & SAM metrics (IEEE GRSL 2025)
│   └── recycling_pipeline.py                       # End-to-end component vision & compression pipeline
├── Research Paper/
│   └── Multiscale_SpatialSpectral_Attention_Network_for_Hyperspectral_Image_Compressed_Sensing.pdf
├── Test Data/
│   └── PCB.jpg                                     # Test PCB sample image
└── webapp/
    ├── index.html                                  # Interactive dashboard
    ├── style.css                                   # Modern dark glassmorphic styling
    ├── script.js                                   # UI logic, telemetry, & animations
    ├── output_reconstructed_ssanet.jpg
    ├── output_reconstructed_baseline_grid.jpg
    ├── output_saliency.jpg
    ├── output_component_map.jpg
    └── pipeline_results.json
```

---

## 📄 References & Citations

1. S. Dong, J. Xiao, Z. Zhang, H. Li, and L. Liao, *"Multiscale Spatial–Spectral Attention Network for Hyperspectral Image Compressed Sensing,"* **IEEE Geoscience and Remote Sensing Letters**, vol. 22, 2025, Art no. 5510505.
2. C.-C. Hsu et al., *"DCSN: Deep Compressed Sensing Network for Efficient Hyperspectral Data Transmission of Miniaturized Satellite,"* **IEEE TGRS**, 2021.
3. X. Zhou et al., *"BTC-Net: Efficient Bit-Level Tensor Data Compression Network for Hyperspectral Image,"* **IEEE TGRS**, 2024.

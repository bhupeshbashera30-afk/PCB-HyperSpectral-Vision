# PCB HyperSpectral Vision & Recycling-Aware Deep Compression Flow Diagram

## End-to-End System Flowchart

```mermaid
flowchart TD
    %% Top Level
    UPLOAD["Image Upload"]

    %% Main Ingestion & Preprocessing
    INGEST["PCB Data Ingestion<br/><small>Raw RGB Image / 224-Band Hyperspectral Cube (400–1000 nm)</small>"]
    CALIB["Calibration & Preprocessing<br/><small>Dark/White Reference Radiometric Calibration & Rescaling</small>"]
    PCA["PCA Dimensionality Reduction<br/><small>Extracts Top 15 Principal Components (224 Bands -> 15 Bands)</small>"]
    PATCH["Spatial Patch Extractor<br/><small>8×8 Context Window Slicing (WS=8, Stride=1 with Border Padding)</small>"]
    BACKBONE["Spatio-Spectral Feature Backbone<br/><small>GaborMamba (Gabor Filters + Conv1D) / CNN2D_AttGCN (CoT + GloRe)</small>"]
    SALIENCY["Recycling Importance Attention Engine<br/><small>Saliency Map Generation: Identifies High-Value vs. Low-Value Regions</small>"]

    UPLOAD --> INGEST
    INGEST --> CALIB
    CALIB --> PCA
    PCA --> PATCH
    PATCH --> BACKBONE
    BACKBONE --> SALIENCY

    %% Saliency-Guided Adaptive Compression Split
    COMP_HIGH["Important Regions (Low Compression)<br/><small>ICs, Capacitors, Connectors (Fine Quantization)</small>"]
    COMP_LOW["Unimportant Regions (High Compression)<br/><small>FR4 Substrate, Non-Functional Traces (Coarse Rate)</small>"]

    SALIENCY --> COMP_HIGH
    SALIENCY --> COMP_LOW

    %% Latent Code Bottleneck
    LATENT["Compressed Latent Representation (z)<br/><small>Compact Feature Code (Storage / Transmission Bottleneck)</small>"]

    COMP_HIGH --> LATENT
    COMP_LOW --> LATENT

    %% Parallel Execution: Reconstruction Audit (Left) vs Classification Head (Right)
    RECON["Image Synthesis Decoder D(z)<br/><small>Reconstructs Spatial-Spectral PCB Image Cube</small>"]
    AUDIT["Reconstruction Quality Audit<br/><small>PSNR / SSIM / Spectral Angle Mapper (SAM) Verification</small>"]

    TASK["Component Classification Head<br/><small>Softmax: Others (0), IC (1), Capacitor (2), Connector (3)</small>"]
    INVENTORY["Component Inventory & Analytics<br/><small>Pixel Footprints, Component Counts & Classification Metrics</small>"]

    LATENT --> RECON
    RECON --> AUDIT

    LATENT --> TASK
    TASK --> INVENTORY

    %% Final Unified Output & Feedback Loop
    OUTPUT["Live Labeled Component Map & Analytics<br/><small>Pixel-Level Segmentation Masks, Component Counts, Confidence & Metrics</small>"]

    AUDIT --> OUTPUT
    INVENTORY --> OUTPUT
    OUTPUT --> UPLOAD

    %% Styling
    classDef default fill:#f8fafc,stroke:#94a3b8,stroke-width:1.2px,color:#0f172a;
    classDef topbox fill:#f8fafc,stroke:#94a3b8,stroke-width:1.4px,color:#0f172a;
    classDef outputbox fill:#ecfdf5,stroke:#10b981,stroke-width:1.4px,color:#065f46;

    class UPLOAD topbox;
    class OUTPUT outputbox;
```

---

## Generated Artifacts

* 🖼️ **High-Resolution Vector Diagram**: [pcb_project_flow_diagram.png](file:///c:/Users/sanub/OneDrive/Desktop/Major/pcb_project_flow_diagram.png)
* 📄 **Clean 1-Page PDF**: [PCB_Recycling_Flow_Architecture.pdf](file:///c:/Users/sanub/OneDrive/Desktop/Major/PCB_Recycling_Flow_Architecture.pdf)

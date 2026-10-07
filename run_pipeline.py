"""
Command-line runner for the PCB HyperSpectral Vision & Recycling-Aware Deep Compression Pipeline.
Executes the full pipeline according to pcb_project_flow_diagram.png and SSANet (IEEE GRSL 2025).

Usage:
    python run_pipeline.py [--image PATH] [--rate RATE] [--output DIR]

Example:
    python run_pipeline.py --image "Test Data/PCB.jpg" --rate adaptive
"""

import argparse
import sys
import os
import json

# Ensure models can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from models.recycling_pipeline import PCBRecyclingPipeline

def main():
    parser = argparse.ArgumentParser(description="PCB HyperSpectral Vision & Recycling Deep Compression")
    parser.add_argument("--image", type=str, default="Test Data/PCB.jpg", help="Path to input PCB image")
    parser.add_argument("--rate", type=str, default="adaptive", choices=["adaptive", "1%", "5%", "10%", "20%"],
                        help="Compression sampling rate (SSANet)")
    parser.add_argument("--output", type=str, default="webapp", help="Output directory for results & visual maps")
    args = parser.parse_args()

    print("=" * 75)
    print(" ⬡ PCB HyperSpectral Vision & Saliency-Guided Deep Compression")
    print("   Architecture: SSANet (IEEE GRSL 2025) + GaborMamba + CNN2D_AttGCN")
    print("=" * 75)
    print(f"[*] Input image      : {args.image}")
    print(f"[*] Compression mode : {args.rate}")
    print(f"[*] Output directory : {args.output}")
    print("-" * 75)

    pipeline = PCBRecyclingPipeline()
    results = pipeline.run(args.image, compression_mode=args.rate, output_dir=args.output)

    print("\n[+] PIPELINE SUMMARY")
    print(f"    - Execution Time       : {results['execution_time_ms']} ms")
    print(f"    - Effective Rate       : {results['compression_latent']['effective_sampling_rate']}% ({results['compression_latent']['compression_ratio']} compression)")
    print(f"    - Bandwidth Savings    : {results['compression_latent']['bandwidth_savings_pct']}%")
    print(f"    - Reconstruction PSNR  : {results['audit_metrics']['ssanet']['psnr_db']} dB")
    print(f"    - Spectral Angle (SAM) : {results['audit_metrics']['ssanet']['sam_deg']} deg (Distortion < 1.62 deg)")
    print(f"    - Grid Artifacts       : {results['audit_metrics']['ssanet']['grid_artifacts']}")
    print(f"    - Component Counts     : IC={results['component_counts']['IC']}, Caps={results['component_counts']['Capacitor']}, Connectors={results['component_counts']['Connector']}")
    print("=" * 75)
    print(f"[✓] All visual maps and JSON report saved to '{args.output}/'")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compute CKA scores (diagonal + full matrix) for encoder models.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from Similarity_utils import (
    load_layer_representations,
    compute_cka_diagonal,
    compute_cka_full_matrix,
    save_results
)

# ============================================================
# CONFIGURATION
# ============================================================

REP_ROOT = "/rep-root/rep"
OUTPUT_DIR = "/output-root/cka_results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Just add model names here - script will auto-detect num_layers
MODELS = [
    # "mbert",
    # "hing_mbert", 
    # "xlm_roberta_base",
    # "xlm_roberta_large",
    # "hing_mbert_mixed",
    # "hing_roberta",
    # "hing_roberta_mixed",
    # "mbert_trilingual_aligned",
    # "xlmr_trilingual_aligned",
    # "xlmr_ablation",
    # "mbert_ablation",
    "mbert_ablation_main",
    "xlmr_ablation_main"
]

LANGUAGE_PAIRS = [
    ("en", "cm"),
    ("en", "hi"),
    ("hi", "cm")
]

# ============================================================
# PLOTTING
# ============================================================

def plot_diagonal(scores, title, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(scores)), scores, marker='o', linewidth=2)
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('CKA', fontsize=14)
    plt.title(title, fontsize=14)
    plt.ylim([0, 1])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path + ".png", dpi=300)
    plt.savefig(save_path + ".pdf")
    plt.close()


def plot_heatmap(matrix, title, save_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(label='CKA')
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('Layer', fontsize=14)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path + ".png", dpi=300)
    plt.savefig(save_path + ".pdf")
    plt.close()


# ============================================================
# MAIN
# ============================================================

print(f"Device: {DEVICE}\n")
print("="*70)
print("COMPUTING CKA SCORES")
print("="*70)

for model_name in MODELS:
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")
    
    model_path = os.path.join(REP_ROOT, model_name)
    
    # Auto-detect number of layers
    en_dir = os.path.join(model_path, "en")
    if not os.path.exists(en_dir):
        print(f"ERROR: {en_dir} not found")
        continue
    
    num_layers = len([f for f in os.listdir(en_dir) if f.startswith("layer_") and f.endswith(".npy")])
    print(f"Detected {num_layers} layers")
    
    # Load all language representations
    try:
        en_reps = load_layer_representations(os.path.join(model_path, "en"), num_layers)
        hi_reps = load_layer_representations(os.path.join(model_path, "hi"), num_layers)
        cm_reps = load_layer_representations(os.path.join(model_path, "cm"), num_layers)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        continue
    
    print(f"Shape: {en_reps[0].shape}")
    
    # Process each language pair
    for lang1, lang2 in LANGUAGE_PAIRS:
        pair_name = f"{lang1}-{lang2}"
        print(f"\n  {pair_name}:")
        
        # Get representations
        reps1 = {"en": en_reps, "hi": hi_reps, "cm": cm_reps}[lang1]
        reps2 = {"en": en_reps, "hi": hi_reps, "cm": cm_reps}[lang2]
        
        # Compute CKA
        diagonal = compute_cka_diagonal(reps1, reps2, device=DEVICE)
        full_matrix = compute_cka_full_matrix(reps1, reps2, device=DEVICE)
        
        # Save results
        output_dir = os.path.join(OUTPUT_DIR, model_name, pair_name)
        save_results(diagonal, full_matrix, output_dir, pair_name)
        
        # Plot
        plot_diagonal(diagonal, f"{model_name} {pair_name} Diagonal CKA", 
                     os.path.join(output_dir, f"{pair_name}_diagonal"))
        plot_heatmap(full_matrix, f"{model_name} {pair_name} CKA Matrix",
                    os.path.join(output_dir, f"{pair_name}_heatmap"))
        
        print(f"    Diagonal mean: {np.mean(diagonal):.4f}")
        print(f"    Full mean: {np.mean(full_matrix):.4f}")
        print(f"    Saved to: {output_dir}")

print("\n" + "="*70)
print("✅ CKA COMPLETE!")
print(f"Results: {OUTPUT_DIR}")
print("="*70)
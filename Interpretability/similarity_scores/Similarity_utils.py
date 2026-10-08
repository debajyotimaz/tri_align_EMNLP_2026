#!/usr/bin/env python3
"""
Utilities for CKA and SVCCA computation on pre-extracted representations.
"""

import os
import json
import torch
import numpy as np
from typing import List, Tuple
from tqdm import tqdm


# ============================================================
# REPRESENTATION LOADING
# ============================================================

def load_layer_representations(rep_dir: str, num_layers: int) -> List[torch.Tensor]:
    """Load layer representations from .npy files."""
    layers = []
    for layer_idx in range(num_layers):
        layer_path = os.path.join(rep_dir, f"layer_{layer_idx:02d}.npy")
        if not os.path.exists(layer_path):
            raise FileNotFoundError(f"Missing: {layer_path}")
        
        layer_rep = np.load(layer_path)
        layers.append(torch.from_numpy(layer_rep))
    
    return layers


# ============================================================
# CKA COMPUTATION
# ============================================================

def linear_cka_torch(X, Y):
    """
    Compute linear CKA between two representation matrices.
    
    Args:
        X: torch.Tensor of shape (N, D1)
        Y: torch.Tensor of shape (N, D2)
    
    Returns:
        float: CKA similarity score between 0 and 1
    """
    X = X.float()
    Y = Y.float()

    # Center the representations
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)

    # Compute HSIC (Hilbert-Schmidt Independence Criterion)
    hs = torch.norm(Xc.T @ Yc, p="fro") ** 2
    denom = torch.norm(Xc.T @ Xc, p="fro") * torch.norm(Yc.T @ Yc, p="fro")

    if denom == 0:
        return 0.0

    return float((hs / denom).clamp(0, 1).item())


def compute_cka_diagonal(H1_list, H2_list, device="cuda"):
    """Compute diagonal CKA (layer i to layer i)."""
    num_layers = len(H1_list)
    diagonal_scores = []
    
    for i in tqdm(range(num_layers), desc="Diagonal CKA"):
        X = H1_list[i].to(device).float()
        Y = H2_list[i].to(device).float()
        
        diagonal_scores.append(linear_cka_torch(X, Y))
        
        del X, Y
        torch.cuda.empty_cache()
    
    return np.array(diagonal_scores)


def compute_cka_full_matrix(H1_list, H2_list, device="cuda"):
    """Compute full CKA matrix (all layer pairs)."""
    L1, L2 = len(H1_list), len(H2_list)
    M = np.zeros((L1, L2))

    for i in tqdm(range(L1), desc="Full CKA"):
        for j in range(L2):
            X = H1_list[i].to(device).float()
            Y = H2_list[j].to(device).float()
            M[i, j] = linear_cka_torch(X, Y)
            del X, Y
            torch.cuda.empty_cache()

    return M


# ============================================================
# SVCCA COMPUTATION
# ============================================================

def pca_topk(X, k=32):
    """PCA reduction to top k components."""
    X = X.float()
    X = X - X.mean(dim=0, keepdim=True)
    cov = X.t() @ X / (X.shape[0] - 1)
    _, eigvecs = torch.linalg.eigh(cov)
    return X @ eigvecs[:, -k:]


def torch_cca(X, Y, k=32, eps=1e-5):
    """Compute CCA between two representations."""
    X = X.float()
    Y = Y.float()
    X -= X.mean(dim=0, keepdim=True)
    Y -= Y.mean(dim=0, keepdim=True)

    Cxx = X.t() @ X / (X.shape[0]-1) + eps * torch.eye(X.shape[1], device=X.device)
    Cyy = Y.t() @ Y / (Y.shape[0]-1) + eps * torch.eye(Y.shape[1], device=Y.device)
    Cxy = X.t() @ Y / (Y.shape[0]-1)

    Ex, Ux = torch.linalg.eigh(Cxx)
    Ey, Uy = torch.linalg.eigh(Cyy)

    Wx = Ux @ torch.diag(1 / torch.sqrt(Ex + eps)) @ Ux.t()
    Wy = Uy @ torch.diag(1 / torch.sqrt(Ey + eps)) @ Uy.t()

    T = Wx @ Cxy @ Wy
    _, S, _ = torch.linalg.svd(T)

    return S[:k]


def compute_svcca_diagonal(H1_list, H2_list, k=32, device="cuda"):
    """Compute diagonal SVCCA (layer i to layer i)."""
    num_layers = len(H1_list)
    diagonal_scores = []
    
    for i in tqdm(range(num_layers), desc="Diagonal SVCCA"):
        Xp = pca_topk(H1_list[i].to(device), k)
        Yp = pca_topk(H2_list[i].to(device), k)
        cca_vals = torch_cca(Xp, Yp, k=k)
        diagonal_scores.append(cca_vals.mean().item())
        
        del Xp, Yp, cca_vals
        torch.cuda.empty_cache()
    
    return np.array(diagonal_scores)


def compute_svcca_full_matrix(H1_list, H2_list, k=32, device="cuda"):
    """Compute full SVCCA matrix (all layer pairs)."""
    L1, L2 = len(H1_list), len(H2_list)
    M = np.zeros((L1, L2))

    for i in tqdm(range(L1), desc="Full SVCCA"):
        Xp = pca_topk(H1_list[i].to(device), k)
        for j in range(L2):
            Yp = pca_topk(H2_list[j].to(device), k)
            cca_vals = torch_cca(Xp, Yp, k=k)
            M[i, j] = cca_vals.mean().item()
            del Yp, cca_vals
            torch.cuda.empty_cache()
        del Xp
        torch.cuda.empty_cache()

    return M


# ============================================================
# SIMPLE SAVE FUNCTIONS
# ============================================================

def save_results(diagonal, full_matrix, output_dir, pair_name):
    """Save diagonal and full matrix results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save diagonal
    diag_data = {
        "diagonal_scores": diagonal.tolist(),
        "mean": float(np.mean(diagonal)),
        "std": float(np.std(diagonal))
    }
    with open(os.path.join(output_dir, f"{pair_name}_diagonal.json"), 'w') as f:
        json.dump(diag_data, f, indent=2)
    
    # Save full matrix
    full_data = {
        "matrix": full_matrix.tolist(),
        "mean": float(np.mean(full_matrix)),
        "diagonal_mean": float(np.mean(np.diag(full_matrix)))
    }
    with open(os.path.join(output_dir, f"{pair_name}_full.json"), 'w') as f:
        json.dump(full_data, f, indent=2)
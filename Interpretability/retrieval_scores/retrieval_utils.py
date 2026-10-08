#!/usr/bin/env python3
"""
Utilities for retrieval computation with length-aware negative sampling.
Supports multiple similarity metrics: Linear CKA (original), Dot Product (cosine), SVCCA.
"""

import os
import json
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Optional


# ============================================================
# REPRESENTATION LOADING
# ============================================================

def load_layer_representations(rep_dir: str, num_layers: int) -> np.ndarray:
    """
    Load layer-wise representations from disk.
    
    Returns:
        numpy array of shape (num_layers, N, hidden_dim)
    """
    reps = []
    for l in range(num_layers):
        layer_file = os.path.join(rep_dir, f"layer_{l:02d}.npy")
        if not os.path.exists(layer_file):
            raise FileNotFoundError(f"Layer file not found: {layer_file}")
        reps.append(np.load(layer_file))
    
    return np.stack(reps)


# ============================================================
# LENGTH-AWARE NEGATIVE SAMPLING
# ============================================================

def get_token_lengths(sentences: List[str]) -> np.ndarray:
    """Get token length for each sentence (simple whitespace split)."""
    return np.array([len(s.strip().split()) for s in sentences])


def generate_length_aware_negatives(
    sentences: List[str],
    num_negatives: int = 10,
    percentile_window: float = 2.5,
    seed: int = 42
) -> List[List[int]]:
    """
    Generate negative samples from the same length percentile cluster as the query.
    """
    np.random.seed(seed)
    N = len(sentences)
    lengths = get_token_lengths(sentences)
    
    # Compute percentile for each sentence
    percentiles = np.zeros(N)
    for i in range(N):
        percentiles[i] = (lengths <= lengths[i]).sum() / N * 100
    
    negatives = []
    
    for i in range(N):
        query_percentile = percentiles[i]
        
        min_percentile = max(0, query_percentile - percentile_window)
        max_percentile = min(100, query_percentile + percentile_window)
        
        candidates = [
            j for j in range(N)
            if j != i and min_percentile <= percentiles[j] <= max_percentile
        ]
        
        # Expand window if needed
        expansion_steps = 0
        while len(candidates) < num_negatives and expansion_steps < 5:
            expansion_steps += 1
            expanded_window = percentile_window * (1 + 0.5 * expansion_steps)
            min_percentile = max(0, query_percentile - expanded_window)
            max_percentile = min(100, query_percentile + expanded_window)
            candidates = [
                j for j in range(N)
                if j != i and min_percentile <= percentiles[j] <= max_percentile
            ]
        
        if len(candidates) < num_negatives:
            candidates = [j for j in range(N) if j != i]
        
        if len(candidates) >= num_negatives:
            neg_samples = np.random.choice(
                candidates, size=num_negatives, replace=False
            ).tolist()
        else:
            neg_samples = np.random.choice(
                candidates, size=num_negatives, replace=True
            ).tolist()
        
        negatives.append(neg_samples)
    
    return negatives


def generate_negatives(N: int, num_negatives: int = 10, seed: int = 42) -> List[List[int]]:
    """
    Pre-generate negative samples for each query (original random sampling).
    """
    np.random.seed(seed)
    negatives = []
    
    for i in range(N):
        available = [j for j in range(N) if j != i]
        neg = np.random.choice(available, size=num_negatives, replace=False).tolist()
        negatives.append(neg)
    
    return negatives


# ============================================================
# SIMILARITY FUNCTIONS
# ============================================================

def linear_cka_batch(X, Y):
    """
    Computes squared cosine similarity (centered) between vector X and batch Y.
    EXACTLY AS IN ORIGINAL WORKING FILE.
    
    Args:
        X: Query tensor of shape (1, H)
        Y: Candidate tensors of shape (K, H)
        
    Returns:
        Tensor of shape (K,) containing CKA scores
    """
    # Center over features (dim 1)
    X_centered = X - X.mean(dim=1, keepdim=True)
    Y_centered = Y - Y.mean(dim=1, keepdim=True)

    # Numerator: Dot product squared
    numerator = torch.matmul(X_centered, Y_centered.T).pow(2).squeeze(0)  # (K,)

    # Denominator: Norm of X * Norm of each Y_k
    norm_x = torch.sum(X_centered**2)  # scalar
    norm_y = torch.sum(Y_centered**2, dim=1)  # (K,)

    # Avoid division by zero
    denominator = norm_x * norm_y
    denominator = torch.where(denominator < 1e-8, torch.ones_like(denominator), denominator)

    return numerator / denominator


def dot_product_cosine_batch(query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity via dot product after L2 normalization (batched version).
    """
    if query.dim() == 1:
        query = query.unsqueeze(0)
    if query.dim() == 2 and candidates.dim() == 3:
        if query.shape[0] == 1 and candidates.shape[0] > 1:
            query = query.expand(candidates.shape[0], -1)

    query_norm = query / (torch.norm(query, dim=-1, keepdim=True) + 1e-8)
    cand_norm = candidates / (torch.norm(candidates, dim=-1, keepdim=True) + 1e-8)

    if query_norm.dim() == 2 and cand_norm.dim() == 3:
        scores = torch.bmm(query_norm.unsqueeze(1), cand_norm.transpose(1, 2)).squeeze(1)
    else:
        scores = torch.matmul(query_norm, cand_norm.transpose(-2, -1))

    return scores


def svcca_batch(query: torch.Tensor, candidates: torch.Tensor, n_singular: int = 20) -> torch.Tensor:
    """
    SVCCA similarity (batched over queries) with robustness fixes.
    """
    if query.dtype == torch.float16:
        query = query.float()
    if candidates.dtype == torch.float16:
        candidates = candidates.float()
    
    if query.dim() == 2:
        query = query.unsqueeze(1)
    
    bs, _, dim = query.shape
    _, K, _ = candidates.shape
    
    scores = torch.zeros(bs, K, device=query.device)
    
    for b in range(bs):
        q = query[b]
        c = candidates[b]
        
        stacked = torch.cat([q, c], dim=0)
        valid_mask = torch.isfinite(stacked).all(dim=0) & (stacked.std(dim=0) > 1e-8)
        
        if valid_mask.sum() < 2:
            scores[b] = torch.zeros(K, device=query.device)
            continue
        
        stacked_clean = stacked[:, valid_mask]
        dim_clean = stacked_clean.shape[1]
        q_rank = min(n_singular, dim_clean // 2, 32)
        
        try:
            u, s, vh = torch.svd_lowrank(stacked_clean, q=q_rank)
        except RuntimeError as e:
            print(f"SVCCA SVD failed for batch {b}: {e}. Using zero similarity.")
            scores[b] = torch.zeros(K, device=query.device)
            continue
        
        u_q = u[:1, :q_rank]
        u_c = u[1:, :q_rank]
        corrs = torch.einsum("ik,jk->ij", u_q, u_c).abs()
        scores[b] = corrs.mean(dim=0)
    
    return scores


# ============================================================
# RETRIEVAL COMPUTATION - CKA (EXACTLY AS ORIGINAL - NO BATCHING)
# ============================================================

def compute_layerwise_retrieval(cm_reps, native_reps, negatives, device="cuda"):
    """
    Compute layer-wise retrieval accuracy using CKA similarity.
    
    EXACTLY AS IN ORIGINAL WORKING FILE - loop based, NO batching.
    This ensures correct accuracy calculation.
    
    Args:
        cm_reps: Code-mixed representations, shape (num_layers, N, hidden_dim)
        native_reps: Native representations, shape (num_layers, N, hidden_dim)
        negatives: List of negative sample indices for each query
        device: Device to run computations on
        
    Returns:
        Dictionary containing:
            - layerwise_accuracy: List of accuracy values per layer (as Python floats)
            - layer_correct: List of correct counts per layer
            - total_queries: Total number of queries
    """
    # Convert to torch tensors and move to device
    cm_reps = torch.from_numpy(cm_reps).to(device)
    native_reps = torch.from_numpy(native_reps).to(device)
    
    num_layers, N, hidden_dim = cm_reps.shape
    print(f"Computing retrieval for {num_layers} layers, {N} samples, hidden_dim={hidden_dim}")
    
    layer_correct = torch.zeros(num_layers, device=device)
    
    for i in tqdm(range(N), desc="Retrieval"):
        # Create candidate pool: [correct] + [negatives]
        candidate_ids = [i] + negatives[i]
        cand_idx = torch.tensor(candidate_ids, dtype=torch.long, device=device)
        
        for l in range(num_layers):
            query = cm_reps[l, i:i + 1]           # (1, H)
            pool = native_reps[l, cand_idx]       # (11, H) - 1 positive + 10 negatives
            
            scores = linear_cka_batch(query, pool)
            
            # Check if the highest score is for the correct sample (index 0)
            if torch.argmax(scores).item() == 0:
                layer_correct[l] += 1
    
    # Calculate accuracy - convert to Python floats immediately
    layerwise_accuracy = (layer_correct / N).cpu().numpy().tolist()
    layer_correct_list = layer_correct.cpu().numpy().tolist()
    
    return {
        "layerwise_accuracy": layerwise_accuracy,  # List of floats
        "layer_correct": layer_correct_list,       # List of ints
        "total_queries": N
    }


# ============================================================
# RETRIEVAL COMPUTATION - DOT PRODUCT (BATCHED)
# ============================================================

def compute_layerwise_retrieval_dot(
    query_reps: np.ndarray,
    target_reps: np.ndarray,
    negatives: List[List[int]],
    device: str = "cuda",
    batch_size: int = 256,
) -> Dict:
    """
    Compute layer-wise retrieval accuracy using Dot Product (cosine) similarity.
    Batched implementation for efficiency.
    """
    query_reps = torch.from_numpy(query_reps).to(device)
    target_reps = torch.from_numpy(target_reps).to(device)
    
    num_layers, N, hidden_dim = query_reps.shape
    print(f"Computing retrieval (DOT): {num_layers} layers, {N} samples, dim={hidden_dim}")
    
    pos_idx = torch.arange(N, device=device)[:, None]
    neg_idx = torch.tensor(negatives, dtype=torch.long, device=device)
    all_cand_idx = torch.cat([pos_idx, neg_idx], dim=1)
    
    layer_correct = torch.zeros(num_layers, dtype=torch.float64, device=device)
    
    for l in range(num_layers):
        print(f"  Layer {l:2d}/{num_layers-1} (DOT) ... ", end="", flush=True)
        
        correct_count = 0
        
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            bs = end - start
            
            queries = query_reps[l, start:end]
            cand_idx_batch = all_cand_idx[start:end]
            candidates = target_reps[l][cand_idx_batch]
            
            scores = dot_product_cosine_batch(queries, candidates)
            correct = (torch.argmax(scores, dim=1) == 0)
            correct_count += correct.sum().item()
        
        layer_correct[l] = correct_count
        acc = correct_count / N
        print(f"done  ({acc:.4f})")
    
    layerwise_accuracy = (layer_correct / N).cpu().numpy()
    
    return {
        "layerwise_accuracy": layerwise_accuracy.tolist(),
        "layer_correct": layer_correct.cpu().numpy().tolist(),
        "total_queries": N,
        "metric": "dot"
    }


# ============================================================
# RETRIEVAL COMPUTATION - SVCCA (BATCHED)
# ============================================================

def compute_layerwise_retrieval_svcca(
    query_reps: np.ndarray,
    target_reps: np.ndarray,
    negatives: List[List[int]],
    device: str = "cuda",
    batch_size: int = 256,
    svcca_n_singular: int = 20
) -> Dict:
    """
    Compute layer-wise retrieval accuracy using SVCCA similarity metric.
    """
    query_reps = torch.from_numpy(query_reps).to(device).float()
    target_reps = torch.from_numpy(target_reps).to(device).float()
    
    num_layers, N, hidden_dim = query_reps.shape
    print(f"Computing retrieval (SVCCA): {num_layers} layers, {N} samples, dim={hidden_dim}")
    
    pos_idx = torch.arange(N, device=device)[:, None]
    neg_idx = torch.tensor(negatives, dtype=torch.long, device=device)
    all_cand_idx = torch.cat([pos_idx, neg_idx], dim=1)
    
    layer_correct = torch.zeros(num_layers, dtype=torch.float64, device=device)
    
    for l in range(num_layers):
        print(f"  Layer {l:2d}/{num_layers-1} (SVCCA) ... ", end="", flush=True)
        
        correct_count = 0
        
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            
            queries = query_reps[l, start:end]
            cand_idx_batch = all_cand_idx[start:end]
            candidates = target_reps[l][cand_idx_batch]
            
            scores = svcca_batch(queries, candidates, n_singular=svcca_n_singular)
            correct = (torch.argmax(scores, dim=1) == 0)
            correct_count += correct.sum().item()
        
        layer_correct[l] = correct_count
        acc = correct_count / N
        print(f"done  ({acc:.4f})")
    
    layerwise_accuracy = (layer_correct / N).cpu().numpy()
    
    return {
        "layerwise_accuracy": layerwise_accuracy.tolist(),
        "layer_correct": layer_correct.cpu().numpy().tolist(),
        "total_queries": N,
        "metric": "svcca",
        "svcca_n_singular": svcca_n_singular
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_retrieval_results(results, model_name):
    """
    Print retrieval results in a formatted way.
    """
    metric = results.get('metric', 'cka').upper()
    
    print(f"\n{'='*60}")
    print(f"RETRIEVAL RESULTS: {model_name} ({metric})")
    print(f"{'='*60}")
    print(f"Total queries: {results['total_queries']}")
    print(f"\nLayer-wise Retrieval Accuracy ({metric}):")
    
    for l, acc in enumerate(results['layerwise_accuracy']):
        correct = results['layer_correct'][l]
        if isinstance(correct, (list, np.ndarray)):
            correct = int(correct[l]) if len(correct) > l else int(correct)
        else:
            correct = int(correct)
        print(f"Layer {l:02d}: {acc:.4f} ({correct}/{results['total_queries']})")
    
    accuracies = results['layerwise_accuracy']
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Mean Accuracy: {np.mean(accuracies):.4f}")
    print(f"  Best Layer: {np.argmax(accuracies):02d} ({max(accuracies):.4f})")
    print(f"  Worst Layer: {np.argmin(accuracies):02d} ({min(accuracies):.4f})")
    print(f"{'='*60}\n")


def print_retrieval_summary(results: Dict, pair_name: str):
    """Print retrieval results summary."""
    accuracies = np.array(results['layerwise_accuracy'])
    metric = results.get('metric', 'cka').upper()
    
    print(f"\n  {pair_name} ({metric}):")
    print(f"    Mean Accuracy (all layers): {accuracies.mean():.4f}")
    if len(accuracies) > 1:
        print(f"    Mean Accuracy (layers 1+):  {accuracies[1:].mean():.4f}")
    print(f"    Best Layer: {accuracies.argmax()} ({accuracies.max():.4f})")
    print(f"    Last Layer: {accuracies[-1]:.4f}")


def save_retrieval_results(results, output_path, model_name):
    """
    Save retrieval results to a file.
    """
    save_dict = {
        "model_name": model_name,
        "total_queries": results['total_queries'],
        "layerwise_accuracy": results['layerwise_accuracy'],
        "layer_correct": results['layer_correct'],
        "mean_accuracy": float(np.mean(results['layerwise_accuracy'])),
        "best_layer": int(np.argmax(results['layerwise_accuracy'])),
        "best_accuracy": float(max(results['layerwise_accuracy'])),
    }
    
    if 'metric' in results:
        save_dict['metric'] = results['metric']
    if 'svcca_n_singular' in results:
        save_dict['svcca_n_singular'] = results['svcca_n_singular']
    
    with open(output_path, 'w') as f:
        json.dump(save_dict, f, indent=2)
    
    print(f"Results saved to: {output_path}")
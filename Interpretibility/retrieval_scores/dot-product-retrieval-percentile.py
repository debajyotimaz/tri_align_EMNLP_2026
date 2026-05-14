#!/usr/bin/env python3
"""
Compute retrieval scores using DOT PRODUCT (Cosine Similarity).
Generates results for all 6 directional pairs.
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from retrieval_utils import (  # FIXED: retrieval (not retrival)
    load_layer_representations,
    compute_layerwise_retrieval_dot,  # FIXED: Use specific function name
    print_retrieval_summary
)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "/data/combined_dataset_cleaned.json"
REP_ROOT = "/rep-root/rep"
OUTPUT_DIR = "/output-dir/results_length_aware_dot-5"

# Cache directory for precomputed candidate pools
CANDIDATES_CACHE_DIR = os.path.join(
    "/root/retrival-scores",
    "negatives_cache-5"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CANDIDATES_CACHE_DIR, exist_ok=True)

# Models to evaluate
MODELS = {
    "mbert": {"path": "mbert", "num_layers": 13},
    # "hing_mbert": {"path": "hing_mbert", "num_layers": 13},
    # "hing_mbert_mixed": {"path": "hing_mbert_mixed", "num_layers": 13},
    # "hing_roberta": {"path": "hing_roberta", "num_layers": 13},
    # "hing_roberta_mixed": {"path": "hing_roberta_mixed", "num_layers": 13},
    # "xlm_roberta_base": {"path": "xlm_roberta_base", "num_layers": 13},
}

# Retrieval configuration
NUM_NEGATIVES = 10
PERCENTILE_WINDOW = 5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print(f"Percentile window: ±{PERCENTILE_WINDOW} percentiles")
print(f"Seed: {SEED}")
print("Similarity metric: DOT PRODUCT (Cosine Similarity)\n")

# ============================================================
# LOAD DATASET
# ============================================================

print("Loading dataset...")
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

english = [x["English"] for x in data]
hindi = [x["Hindi"] for x in data]
codemixed = [x["Roman_Hinglish"] for x in data]

N = len(data)
print(f"Loaded {N} samples\n")

# ============================================================
# LENGTH-AWARE CANDIDATES & NEGATIVE SAMPLING
# ============================================================

def get_token_lengths(sentences):
    return np.array([len(s.strip().split()) for s in sentences])


def precompute_length_aware_candidates(
    sentences,
    percentile_window=PERCENTILE_WINDOW,
    cache_key="lang"
):
    cache_file = os.path.join(CANDIDATES_CACHE_DIR, f"candidates_{cache_key}_{percentile_window}.json")
    
    if os.path.exists(cache_file):
        print(f"Loading precomputed candidates for {cache_key} from {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    print(f"Precomputing length-aware candidate pools for {cache_key}...")
    N = len(sentences)
    lengths = get_token_lengths(sentences)
    
    # Vectorized percentile ranks
    sorted_idx = np.argsort(lengths)
    ranks = np.empty(N, dtype=float)
    ranks[sorted_idx] = np.arange(N) / (N - 1) * 100  # 0 to 100
    
    candidates_per_query = []
    
    for i in tqdm(range(N), desc=f"Candidates {cache_key}"):
        query_perc = ranks[i]
        min_p = max(0, query_perc - percentile_window)
        max_p = min(100, query_perc + percentile_window)
        
        mask = (ranks >= min_p) & (ranks <= max_p) & (np.arange(N) != i)
        cands = np.where(mask)[0].tolist()
        
        candidates_per_query.append(cands)
    
    with open(cache_file, 'w') as f:
        json.dump(candidates_per_query, f)
    print(f"Saved candidates to {cache_file}")
    
    return candidates_per_query


def sample_negatives_from_candidates(candidates_list, num_negatives=NUM_NEGATIVES, seed=SEED):
    np.random.seed(seed)
    negatives = []
    
    for cands in candidates_list:
        if len(cands) >= num_negatives:
            chosen = np.random.choice(cands, num_negatives, replace=False).tolist()
        elif len(cands) > 0:
            chosen = np.random.choice(cands, num_negatives, replace=True).tolist()
        else:
            chosen = []  # fallback - very rare
        negatives.append(chosen)
    
    return negatives


# Precompute or load candidates
print("Preparing length-aware candidate pools...")
candidates_en = precompute_length_aware_candidates(english, cache_key="en")
candidates_hi = precompute_length_aware_candidates(hindi, cache_key="hi")
candidates_cm = precompute_length_aware_candidates(codemixed, cache_key="cm")

# Sample negatives
print("\nSampling negatives with seed =", SEED)
negatives_en = sample_negatives_from_candidates(candidates_en)
negatives_hi = sample_negatives_from_candidates(candidates_hi)
negatives_cm = sample_negatives_from_candidates(candidates_cm)
print("✓ Negatives ready for all languages\n")

# ============================================================
# MAIN RETRIEVAL LOOP
# ============================================================

all_results = {}

for model_name, model_config in MODELS.items():
    print(f"{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")

    model_output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    output_file = os.path.join(model_output_dir, "dot-retrival-triplets.json")

    if os.path.exists(output_file):
        print(f"Output already exists for {model_name}: {output_file}")
        print("→ Skipping computation\n")
        try:
            with open(output_file, 'r') as f:
                all_results[model_name] = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load existing result: {e}")
        continue

    model_path = os.path.join(REP_ROOT, model_config["path"])
    en_dir = os.path.join(model_path, "en")
    hi_dir = os.path.join(model_path, "hi")
    cm_dir = os.path.join(model_path, "cm")

    if not all(os.path.exists(d) for d in [en_dir, hi_dir, cm_dir]):
        print(f"⚠ Missing directories for {model_name}, skipping...\n")
        continue

    print("Loading representations...")
    try:
        en_reps = load_layer_representations(en_dir, model_config["num_layers"])
        hi_reps = load_layer_representations(hi_dir, model_config["num_layers"])
        cm_reps = load_layer_representations(cm_dir, model_config["num_layers"])
    except FileNotFoundError as e:
        print(f"⚠ Error loading representations: {e}\n")
        continue

    num_layers, n_samples, hidden_dim = en_reps.shape
    print(f"✓ Loaded: {num_layers} layers, {n_samples} samples, dim={hidden_dim}\n")

    assert en_reps.shape == hi_reps.shape == cm_reps.shape, "Shape mismatch!"

    results = {
        "model_name": model_name,
        "num_layers": num_layers,
        "num_samples": n_samples,
        "hidden_dim": hidden_dim,
        "num_negatives": NUM_NEGATIVES,
        "percentile_window": PERCENTILE_WINDOW,
        "seed": SEED,
        "metric": "dot",
        "retrieval_pairs": {}
    }

    pairs = [
        ("EN_to_CM", "EN → CM", "EN", "CM", en_reps, cm_reps, negatives_cm),
        ("CM_to_EN", "CM → EN", "CM", "EN", cm_reps, en_reps, negatives_en),
        ("EN_to_HI", "EN → HI", "EN", "HI", en_reps, hi_reps, negatives_hi),
        ("HI_to_EN", "HI → EN", "HI", "EN", hi_reps, en_reps, negatives_en),
        ("HI_to_CM", "HI → CM", "HI", "CM", hi_reps, cm_reps, negatives_cm),
        ("CM_to_HI", "CM → HI", "CM", "HI", cm_reps, hi_reps, negatives_hi),
    ]

    for pair_key, desc, query_lang, target_lang, q_reps, t_reps, negs in pairs:
        print(f"Computing {desc}...")
        # FIXED: Use compute_layerwise_retrieval_dot (no metric parameter)
        pair_result = compute_layerwise_retrieval_dot(
            q_reps, t_reps, negs, DEVICE
        )

        accuracies = pair_result["layerwise_accuracy"]
        mean_acc = sum(accuracies) / len(accuracies)
        best_idx = int(np.argmax(accuracies))
        best_acc = accuracies[best_idx]

        enriched = {
            "description": desc,
            "query": query_lang,
            "target": target_lang,
            "layerwise_accuracy": accuracies,
            "layer_correct": pair_result["layer_correct"],
            "mean_accuracy": mean_acc,
            "best_layer": best_idx,
            "best_accuracy": best_acc
        }

        results["retrieval_pairs"][pair_key] = enriched
        print_retrieval_summary(pair_result, desc)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved: {output_file}\n")
    all_results[model_name] = results

# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n" + "="*70)
print("SUMMARY TABLE - ALL MODELS (DOT PRODUCT / COSINE)")
print("="*70)

for model_name, results in all_results.items():
    print(f"\n{model_name}:")
    print(f"  {'Direction':<15} {'Mean (all)':<12} {'Mean (1+)':<12} {'Last Layer':<12}")
    print(f"  {'-'*55}")

    for pair_key in ["EN_to_CM", "CM_to_EN", "EN_to_HI", "HI_to_EN", "HI_to_CM", "CM_to_HI"]:
        if pair_key not in results["retrieval_pairs"]:
            continue
        acc = results["retrieval_pairs"][pair_key]["layerwise_accuracy"]
        mean_all = sum(acc) / len(acc)
        mean_1plus = sum(acc[1:]) / len(acc[1:]) if len(acc) > 1 else mean_all
        last = acc[-1]

        print(f"  {pair_key:<15} {mean_all:<12.4f} {mean_1plus:<12.4f} {last:<12.4f}")

print(f"\n✅ All DOT results saved under: {OUTPUT_DIR}")
# ============================================================
#   GLOBAL PSEUDO-PERPLEXITY (TEST SPLIT)
#   Roman_Hinglish Column
#   Grouped by Model Family
# ============================================================

import os
import math
import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import pandas as pd
from pathlib import Path

torch.set_grad_enabled(False)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------
DATA_FILE = "/data-root/cm_native_combined_dataset_cleaned.parquet"
TEST_IDX  = "/data-root/splits_random_123/test_indices.npy"

COLUMN_NAME = "Roman_Hinglish"
MAX_LEN = 512

# live results JSON — scores saved here after each model finishes
LIVE_RESULTS_JSON = "live_ppl_trans_results.json"

# ------------------------------------------------------------
# MODELS
# Format:
# (model_path_or_hf_name, display_name, family)
# ------------------------------------------------------------
MODELS = [
    # -------- XLMR FAMILY --------

    # ("xlm-roberta-base",
    #  "XLM-R-Base", "XLMR"),

    # ("l3cube-pune/hing-roberta",
    #  "HingRoBERTa", "XLMR"),

    # ("l3cube-pune/hing-roberta-mixed",
    #  "HingRoBERTa-Mixed", "XLMR"),



    # -------- MBERT FAMILY --------
    # ("bert-base-multilingual-cased",
    #  "mBERT-Base", "MBERT"),

    ("l3cube-pune/hing-mbert",
     "HingMBERT", "MBERT"),

    # ("l3cube-pune/hing-mbert-mixed",
    #  "HingMBERT-Mixed", "MBERT"),
]

MASK_BATCH = 64  # number of masked positions to process in one forward pass

# ------------------------------------------------------------
# LIVE JSON SAVE HELPER
# ------------------------------------------------------------
def save_live_results(results):
    with open(LIVE_RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[live] Saved {len(results)} result(s) to {LIVE_RESULTS_JSON}")

# ------------------------------------------------------------
# GLOBAL PPL FUNCTION
# ------------------------------------------------------------
def compute_ppl(model_path, display_name, family, df_test):  # <-- df_test passed in, not reloaded

    print(f"\n=== Loading {display_name} ({family}) ===")
    print(f"[debug] model_path: {model_path}")

    # use local_files_only=True for local paths to bypass HF hub validation entirely
    is_local = Path(model_path).exists()
    load_kwargs = {"local_files_only": True} if is_local else {}
    print(f"[debug] is_local: {is_local} | load_kwargs: {load_kwargs}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, **load_kwargs)

    model = AutoModelForMaskedLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="auto",
        **load_kwargs
    ).eval()

    texts = df_test[COLUMN_NAME].fillna("").astype(str).tolist()

    total_log_prob = 0.0
    total_tokens = 0
    special_ids = set(tokenizer.all_special_ids)

    print("Computing global pseudo-perplexity...")

    for text in tqdm(texts, desc=display_name):

        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN
        )

        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        seq_len = input_ids.size(1)

        # collect all non-special, non-padding positions
        non_special_positions = [
            t for t in range(seq_len)
            if attention_mask[0, t] == 1 and input_ids[0, t].item() not in special_ids
        ]

        # process in batches of MASK_BATCH instead of one forward pass per token
        for i in range(0, len(non_special_positions), MASK_BATCH):
            batch_positions = non_special_positions[i:i + MASK_BATCH]
            batch_size = len(batch_positions)

            tiled = input_ids.expand(batch_size, -1).clone()
            for j, t in enumerate(batch_positions):
                tiled[j, t] = tokenizer.mask_token_id

            outputs = model(tiled)

            for j, t in enumerate(batch_positions):
                token_id = input_ids[0, t].item()
                logits = outputs.logits[j, t]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                total_log_prob += log_probs[token_id].item()
                total_tokens += 1

    global_ppl = math.exp(-total_log_prob / total_tokens) if total_tokens > 0 else float("inf")

    print(f"{display_name} GLOBAL PPL: {global_ppl:.4f}")

    # free GPU memory before loading next model
    del model
    torch.cuda.empty_cache()

    return family, display_name, global_ppl


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":

    # load data once, outside the model loop
    df = pd.read_parquet(DATA_FILE)
    test_indices = np.load(TEST_IDX)
    df_test = df.iloc[test_indices].reset_index(drop=True)

    # load existing live results to resume from crash if available
    if Path(LIVE_RESULTS_JSON).exists():
        with open(LIVE_RESULTS_JSON) as f:
            live_results = json.load(f)
        print(f"[resume] Found existing results for: {[r['display_name'] for r in live_results]}")
    else:
        live_results = []

    already_done = {r["display_name"] for r in live_results}

    results = []

    for model_path, display_name, family in MODELS:

        # skip models already computed (resume after crash)
        if display_name in already_done:
            print(f"[skip] {display_name} already in live results, skipping.")
            continue

        try:
            family_out, name_out, ppl_out = compute_ppl(model_path, display_name, family, df_test)

            # append to live results and save immediately after each model
            live_results.append({
                "family": family_out,
                "display_name": name_out,
                "ppl": ppl_out
            })
            save_live_results(live_results)

            results.append((family_out, name_out, ppl_out))

        except Exception as e:
            print(f"\n[ERROR] Model '{display_name}' failed with: {e}")
            print(f"[ERROR] Skipping and continuing to next model...\n")
            # save error entry so we know which model crashed
            live_results.append({
                "family": family,
                "display_name": display_name,
                "ppl": None,
                "error": str(e)
            })
            save_live_results(live_results)
            continue

    # rebuild results from live_results for table (exclude errored ones)
    results = [
        (r["family"], r["display_name"], r["ppl"])
        for r in live_results if r["ppl"] is not None
    ]

    # --------------------------------------------------------
    # PRINT FORMATTED TABLE
    # --------------------------------------------------------
    print("\n\nPrediction of masked words\n")

    for fam in ["XLMR", "MBERT"]:

        print(f"\n{fam} Family")
        print("{:<30} {:>20}".format("Model", "Validation Perplexity"))
        print("-" * 55)

        for family, name, ppl_value in results:
            if family == fam:
                print("{:<30} {:>20.2f}".format(name, ppl_value))

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------
    df_results = pd.DataFrame(
        results,
        columns=["Family", "Model", "Validation Perplexity"]
    )

    df_results.to_csv("validation_perplexity_by_xlmr_family.csv", index=False)

    latex_table = df_results.to_latex(index=False, float_format="%.2f")
    with open("validation_perplexity_table.tex", "w") as f:
        f.write(latex_table)

    print("\nSaved:")
    print(" - validation_perplexity_by_family.csv")
    print(" - validation_perplexity_table.tex")
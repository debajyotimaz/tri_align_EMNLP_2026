# ============================================================
# COMPREHENSIVE REPRESENTATION EXTRACTION FOR HINGLISH
# CM, EN, HI only
# ============================================================

import os
import gc
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Import from utils
from utils import (
    extract_encoder_representations,
    save_layer_representations,
)

# ============================================================
# PATHS
# ============================================================
DATA_PATH = (
    "Data_path.json"
)

OUT_ROOT = (
    "/out_dir/"
    "representations/rep_cleaned"
)
os.makedirs(OUT_ROOT, exist_ok=True)

# ============================================================
# CONFIGURATION
# ============================================================
BATCH_SIZE = 32
MAX_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEPARATOR = " </s> "

# ============================================================
# ENCODER MODELS
# ============================================================
ENCODER_MODELS = {
    # "mbert": "bert-base-multilingual-cased",
    # "hing-mbert": "l3cube-pune/hing-mbert",
    # "hing-mbert-mixed": "l3cube-pune/hing-mbert-mixed",
    # "hing-roberta": "l3cube-pune/hing-roberta",
    # "hing-roberta-mixed": "l3cube-pune/hing-roberta-mixed",
    # "hindi_roberta": "l3cube-pune/hindi-roberta",
    # "xlm-roberta-base": "xlm-roberta-base",
}

# ============================================================
# LOAD DATA
# ============================================================
print("Loading dataset...")
import json

with open(DATA_PATH) as f:
    data = json.load(f)

# Extract sentences
codemixed = [x["Roman_Hinglish"] for x in data]
english = [x["English"] for x in data]
hindi = [x["Hindi"] for x in data]

N = len(data)
print(f"Loaded {N} parallel samples")

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def sanitize(name):
    """Sanitize model name for use in file paths"""
    return name.replace("/", "_").replace("-", "_").lower()

# ============================================================
# RUN EXTRACTION FOR ALL ENCODERS
# ============================================================
for tag, model_name in ENCODER_MODELS.items():
    print(f"\n{'='*60}")
    print(f"ENCODER: {tag}")
    print(f"{'='*60}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="balanced_low_0",
        low_cpu_mem_usage=True,
    )

    model_root = os.path.join(OUT_ROOT, sanitize(tag))
    os.makedirs(model_root, exist_ok=True)

    # ----------------------------
    # 1. Code-Mixed (CM)
    # ----------------------------
    print("→ CM (Code-Mixed)")
    out_dir = os.path.join(model_root, "cm")
    
    reps = extract_encoder_representations(
        model, tokenizer, codemixed, 
        batch_size=BATCH_SIZE, 
        max_len=MAX_LEN, 
        device=DEVICE
    )
    save_layer_representations(reps, out_dir)

    # ----------------------------
    # 2. English (EN)
    # ----------------------------
    print("→ EN (English)")
    out_dir = os.path.join(model_root, "en")
    
    reps = extract_encoder_representations(
        model, tokenizer, english, 
        batch_size=BATCH_SIZE, 
        max_len=MAX_LEN, 
        device=DEVICE
    )
    save_layer_representations(reps, out_dir)

    # ----------------------------
    # 3. Hindi (HI)
    # ----------------------------
    print("→ HI (Hindi)")
    out_dir = os.path.join(model_root, "hi")
    
    reps = extract_encoder_representations(
        model, tokenizer, hindi, 
        batch_size=BATCH_SIZE, 
        max_len=MAX_LEN, 
        device=DEVICE
    )
    save_layer_representations(reps, out_dir)

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

print("\n" + "="*60)
print("✅ All representations extracted successfully!")
print("="*60)
print(f"\nSaved to: {OUT_ROOT}")
print("\nDirectory structure for each model:")
print("  - cm/")
print("  - en/")
print("  - hi/")
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for an ACL paper on cross-lingual representation alignment in multilingual encoders, focusing on Hindi-English code-mixed (Hinglish) text. The project studies how models like mBERT and XLM-R represent code-mixed text and introduces Trilingual Post-Training Alignment (TPA) — a post-training objective combining MLM/NSP with cosine alignment loss.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Python 3.9+ required. GPU recommended for training; Translation pipeline requires a vLLM server (uncomment `vllm` in requirements.txt on that machine).

## Running Experiments

All scripts are standalone Python files — no build system, no test framework. Each script has hardcoded paths and configuration via a `TrainingConfig` dataclass or module-level constants that must be edited before running.

**Post-alignment training:**
```bash
cd Post_alignment_training
python mbert-trilingual.py    # BERT family (uses BertForPreTraining)
python xlmr-trilingual.py     # XLM-R family (uses AutoModelForMaskedLM)
```

**Interpretability analyses** (each in its own subdirectory under `Interpretibility/`):
```bash
python Interpretibility/similarity\ scores/cka.py       # CKA
python Interpretibility/similarity\ scores/svcca.py     # SVCCA
python Interpretibility/Token_level_saliency/Compute_RI.py
python Interpretibility/Uncertainity_scores/uncertainity_scores.py
python Interpretibility/TSNE/TSNE.py
python Interpretibility/perplexity/perplexity_encoder.py
python Interpretibility/retrieval_scores/dot-product-retrieval-percentile.py
python Interpretibility/retrieval_scores/dot-product-retrieval-faiss.py
```

**Downstream tasks:**
```bash
python Downstream_tasks/cross_lingual_hate.py
python Downstream_tasks/cross_lingual_sentiment.py
```

**Translation pipeline** (requires vLLM + GPU server):
```bash
cd Translation/trans_hate && bash trans.sh
cd Translation/trans_sentiment && bash trans.sh
```

## Architecture Notes

- **No shared library or config system.** Each script is self-contained with its own imports, config, data loading, and training loop. The two post-training scripts (`mbert-trilingual.py` and `xlmr-trilingual.py`) are structurally parallel but differ in model class: mBERT uses `BertForPreTraining` (MLM + NSP) while XLM-R uses `AutoModelForMaskedLM` (MLM only). Both access the inner encoder via `model.bert` or `model.roberta` respectively.
- **Interpretability scripts** depend on pre-extracted representations saved as `.npy` files. The `Similarity_utils.py` module in `Interpretibility/similarity scores/` provides shared utilities (CKA computation, loading layer representations). The retrieval scripts use `retrieval_utils.py`.
- **Downstream task scripts** have a `MODELS` dict mapping short names to HuggingFace IDs or local checkpoint paths. Entries are commented out by default — uncomment the models you want to evaluate. Results are saved to Excel via openpyxl.
- **Path conventions:** Most scripts use hardcoded absolute paths (e.g., `/data/...`, `/output-path/...`) that must be edited to match the local environment. The post-training scripts use relative paths for data (`../../data/...`). The `split_cache_dir` ensures mBERT and XLM-R use the same train/test split.
- **Translation pipeline** uses YAML configs specifying the vLLM model, data paths, and generation parameters. Two configs per task (English and Hindi target languages), driven by `run_inference.py` / `run_inference_hate.py`.

## Data

- Trilingual corpus: `Data/Cleaned-Data/cleaned-trilingual-corpus/combined_dataset_cleaned.json` — 21,139 triples with fields `English`, `Hindi`, `Roman_Hinglish`
- Parquet version: `cm_native_combined_dataset_cleaned.parquet` (used by post-training scripts)
- Downstream datasets in `Data/Cleaned-Data/cleaned-hate-translated/` and `cleaned-sentiment-translated/`

## Models Evaluated

Two families: BERT-based (`bert-base-multilingual-cased`, `l3cube-pune/hing-bert`, `l3cube-pune/hing-bert-mixed`) and RoBERTa-based (`xlm-roberta-base`, `l3cube-pune/hing-roberta`, `l3cube-pune/hing-roberta-mixed`). The `l3cube-pune` models are Hinglish-adapted variants.

## Key Caveat

The directory `Interpretibility` is intentionally spelled this way throughout the codebase (not "Interpretability"). Similarly `Uncertainity_scores` (not "Uncertainty"). Maintain these spellings when referencing existing paths.

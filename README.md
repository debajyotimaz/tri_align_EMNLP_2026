# Bridging the Code-Mix Gap: Representational Dynamics and Trilingual Alignment in Multilingual Encoders

> **ACL Submission** · Code-Mixed NLP · Cross-Lingual Representation Learning

---

## Overview

This repository accompanies our study on how multilingual encoder-based language models internally represent Hindi–English code-mixed (Hinglish) text, and whether those representations meaningfully connect to their constituent languages.

**Core contributions:**

1. **Trilingual Corpus Construction** — A unified 21,139-sentence corpus pairing English, Devanagari Hindi, and Romanized Hinglish code-mixed text, built from three public datasets.
2. **Probing Cross-Lingual Alignment** — A systematic interpretability suite using CKA, SVCCA, token-level saliency, entropy-based uncertainty, and length-controlled retrieval to characterize how well code-mixed representations align with their parent languages across model layers.
3. **Trilingual Post-Training Alignment (TPA)** — A novel post-training objective combining Masked Language Modeling (MLM) / Next Sentence Prediction (NSP) with a **cosine alignment loss** that jointly pulls code-mixed representations toward both English and Hindi.
4. **Cross-Lingual Downstream Evaluation** — Zero-shot and translate-train evaluation on hate speech detection and sentiment analysis across language directions.

---

## Repository Structure

```
CODE-ACL/
│
├── Data/
│   ├── Cleaned-Data/
│   │   ├── cleaned-hate-translated/
│   │   │   ├── splits/
│   │   │   │   ├── hate_train.csv
│   │   │   │   ├── hate_val.csv
│   │   │   │   └── hate_test.csv
│   │   │   └── hate_translated.csv              # Full translated hate speech dataset
│   │   ├── cleaned-sentiment-translated/
│   │   │   ├── sa_hineng_translated_train.csv
│   │   │   ├── sa_hineng_translated_val.csv
│   │   │   └── sa_hineng_translated_test_clean.csv
│   │   └── cleaned-trilingual-corpus/
│   │       ├── combined_dataset_cleaned.json    # Trilingual corpus (21,139 triples)
│   │       └── cm_native_combined_dataset_cleaned.parquet
│   └── Raw-Data/                                # Original source datasets
│
├── Downstream_tasks/
│   ├── cross_lingual_hate.py                    # Hate speech detection 
│   └── cross_lingual_sentiment.py               # Sentiment analysis 
│
├── Interpretibility/
│   ├── perplexity/
│   │   └── perplexity_encoder.py                # Code-mixed perplexity scores 
│   ├── retrieval_scores/
│   │   ├── dot-product-retrieval-faiss.py        # FAISS hard negative retrieval 
│   │   ├── dot-product-retrieval-percentile.py   # Percentile-based retrieval 
│   │   └── retrieval_utils.py
│   ├── similarity scores/
│   │   ├── cka.py                               # CKA layer-wise analysis 
│   │   ├── svcca.py                             # SVCCA analysis 
│   │   ├── representations.py                   # Representation extraction
│   │   ├── Similarity_utils.py
│   │   └── utils.py
│   ├── Token_level_saliency/
│   │   └── Compute_RI.py                        # Rank-Inverse saliency scores 
│   ├── TSNE/
│   │   └── TSNE.py                              # t-SNE geometry visualization 
│   └── Uncertainity_scores/
│       └── uncertainity_scores.py               # Entropy-based uncertainty reduction 
├── Post_alignment_training/
│   ├── mbert-trilingual.py                      # Trilingual post-training for mBERT family
│   └── xlmr-trilingual.py                       # Trilingual post-training for XLM-R family
│
└── Translation/
    ├── trans_hate/
    │   ├── english.yaml
    │   ├── hindi.yaml
    │   ├── run_inference_hate.py
    │   └── trans.sh
    └── trans_sentiment/
        ├── english.yaml
        ├── hindi.yaml
        ├── run_inference.py
        └── trans.sh
```

---

## Dataset

We build a unified trilingual corpus by merging three publicly available parallel datasets, each providing code-mixed–English sentence pairs, with Hindi added via human annotation or machine translation.

| Source Dataset | Samples | Languages |
|---|---|---|
| CM-En Parallel (Dhar et al., 2018) | 6,096 | CM ↔ EN |
| PHINC (Srivastava and Singh, 2020) | 13,738 | CM ↔ EN |
| LinCE 2021 (Aguilar et al., 2020) | 8,060 | CM ↔ EN ↔ HI |
| **Final Trilingual Corpus** | **21,139** | **CM · EN · HI** |

Missing Hindi translations were generated using the **Google Translate API** and verified using **IndicTrans2**. Cleaned corpus:

```
Data/Cleaned-Data/cleaned-trilingual-corpus/combined_dataset_cleaned.json
```

Each entry has three parallel fields: `English`, `Hindi`, `Roman_Hinglish`.

---

## Models

We evaluate two families of multilingual encoders — one BERT-based and one RoBERTa-based — along with their Hinglish-adapted variants:

| Model | HuggingFace ID | Family |
|---|---|---|
| mBERT | `bert-base-multilingual-cased` | BERT |
| Hing-mBERT | `l3cube-pune/hing-bert` | BERT |
| Hing-mBERT-Mixed | `l3cube-pune/hing-bert-mixed` | BERT |
| XLM-R Base | `xlm-roberta-base` | RoBERTa |
| Hing-RoBERTa | `l3cube-pune/hing-roberta` | RoBERTa |
| Hing-RoBERTa-Mixed | `l3cube-pune/hing-roberta-mixed` | RoBERTa |

Model names and local checkpoint paths are configured via the `TrainingConfig` dataclass inside each script.

---

## Experiments

### 1. Cross-Lingual Alignment Probing (Section 4.1)

We measure how well representations from one language retrieve parallel sentences in another language, using dot-product similarity over pre-extracted layer-wise representations.

**Percentile-based negative sampling** — controls for sentence length to avoid trivial retrieval:
```bash
cd Interpretibility/retrieval_scores
python dot-product-retrieval-percentile.py
```

**FAISS hard negative sampling** — mines semantically hard negatives using approximate nearest-neighbor search:
```bash
cd Interpretibility/retrieval_scores
python dot-product-retrieval-faiss.py
```

---

### 2. Interpretability Analysis (Section 4.2)

**Layer-wise CKA** — Centered Kernel Alignment between language representation matrices per layer:
```bash
cd Interpretibility/similarity_scores
python cka.py
```

**Layer-wise SVCCA** (Figure 9):
```bash
cd Interpretibility/similarity_scores
python svcca.py
```

**Token-level Saliency** (Figure 6):
```bash
cd Interpretibility/Token_level_saliency
python Compute_RI.py
```

**Entropy-based Uncertainty Reduction** (Figure 7):
```bash
cd Interpretibility/Uncertainity_scores
python uncertainity_scores.py
```

**t-SNE Visualization** (Figure 3):
```bash
cd Interpretibility/TSNE
python TSNE.py
```

**Perplexity** (Table 6):
```bash
cd Interpretibility/perplexity
python perplexity_encoder.py
```

### 3. Trilingual Post-Training Alignment (Section 4.3)

**mBERT family:**
```bash
cd Post_alignment_training
python mbert-trilingual.py
```

**XLM-R family:**
```bash
cd Post_alignment_training
python xlmr-trilingual.py
```

> **Ablation (without alignment loss):** Remove or zero out the `λ · L_align` term in the respective script.

### 4. Downstream Tasks (Appendix F)

**Sentiment Analysis:**
```bash
cd Downstream_tasks
python cross_lingual_sentiment.py
```

**Hate Speech Detection:**
```bash
cd Downstream_tasks
python cross_lingual_hate.py
```

### 5. Translation Pipeline (Appendix F.2)

Translations were generated using **QWEN2.5-72B-INSTRUCT** via vLLM. To reproduce:

```bash
cd Translation/trans_hate && bash trans.sh
cd Translation/trans_sentiment && bash trans.sh
```

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | {5e-6, 6e-6, 6.5e-6, 1e-5} |
| Batch Size | 16 |
| Epochs | 10 (mBERT best at epoch 8, XLM-R best at epoch 6) |
| Warmup | 10% of total steps |
| Token Masking | 15% (80% `[MASK]`, 10% random, 10% unchanged) |
| Alignment Weight λ | tuned over {0.05, 0.10, 0.15} |
| Random Seeds | 3 (all results averaged) |

---

## Setup

**Python 3.9+ is required.** We recommend using a virtual environment:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Install all dependencies
pip install -r requirements.txt
```

> **GPU note:** All scripts automatically fall back to CPU, but training and representation extraction are significantly faster on CUDA. For the Translation pipeline, a GPU server running **vLLM** is required — uncomment the `vllm` line in `requirements.txt` on that machine.

---

## Ethics Statement

This work studies social media text that may contain potentially offensive or harmful language present in the original datasets. These instances are included solely for research purposes to study linguistic phenomena in code-mixed settings. The authors do not endorse or promote any harmful or abusive content. AI-based writing assistants were used for grammar checking and language polishing during manuscript preparation.
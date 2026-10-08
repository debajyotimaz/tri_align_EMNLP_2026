# Neither Here Nor There: Cross-Lingual Representation Dynamics of Code-Mixed Text in Multilingual Encoders

[![arXiv](https://img.shields.io/badge/arXiv-2603.19771-b31b1b.svg)](https://arxiv.org/abs/2603.19771)
[![EMNLP Findings 2026](https://img.shields.io/badge/EMNLP-Findings%202026-1f6feb.svg)](https://arxiv.org/abs/2603.19771)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-yellow.svg)](https://huggingface.co/datasets/debajyotimaz/trilingual-hinglish-corpus)
[![Project Page](https://img.shields.io/badge/Project-Page-2ea44f.svg)](https://debajyotimaz.github.io/tri_align_EMNLP_2026/)

<p align="center"><img src="docs/assets/tri_align.gif" width="820" alt="Trilingual alignment animation"></p>

**Debajyoti Mazumder, Divyansh Pathak, Prashant Kodali, Jasabanta Patro**

**Preprint:** <https://arxiv.org/abs/2603.19771>. Accepted at **EMNLP Findings 2026**.

---

## TL;DR

- **Code-mixed text is "neither here nor there."** Off-the-shelf multilingual encoders (mBERT, XLM-R) align
  English and Hindi well, but code-mixed (CM) sentences sit on the periphery, weakly anchored to both:
  EN↔HI > EN↔CM > HI↔CM.
- **Code-mixed adaptation trades one gap for another.** Continued pretraining on Hinglish improves EN↔CM
  alignment but degrades EN↔HI alignment. Interpretability analyses show models read code-mixed text through an
  English-dominant subspace, while native-script Hindi adds complementary signal that reduces uncertainty.
- **Trilingual alignment restores balance.** A post-training cosine alignment loss over EN–HI, EN–CM and HI–CM
  pairs raises the Cross-Lingual Alignment Score (CLAS = MeanAcc − DirBias − SetupStd) for every model, e.g.
  mBERT 14.20 → 50.97 and Hing-RoBERTa 3.74 → 58.98, and improves cross-lingual consistency on sentiment
  analysis and hate speech detection.

---

## Overview

This repository accompanies our study of how multilingual encoder language models represent Hindi–English
code-mixed (Hinglish) text internally, and whether those representations connect meaningfully to the two
languages being mixed.

**Contributions:**

1. **Trilingual corpus construction.** A unified corpus of 21,139 sentences, each given in English, Devanagari
   Hindi, and Romanized Hinglish code-mixed text, built from three public datasets.
2. **Probing cross-lingual alignment.** A systematic interpretability suite: CKA, SVCCA, token-level saliency,
   entropy-based uncertainty, and length-controlled retrieval. Together these show how well code-mixed
   representations align with their parent languages at each model layer.
3. **Trilingual Post-Training Alignment (TPA).** A post-training objective that combines Masked Language
   Modeling (MLM) / Next Sentence Prediction (NSP) with a **cosine alignment loss**, which pulls code-mixed
   representations toward English and Hindi at the same time.
4. **Cross-lingual downstream evaluation.** Zero-shot and translate-train evaluation on hate speech detection
   and sentiment analysis in several language directions.

---

## Repository Structure

```
tri_align_EMNLP_2026/
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
│   │       └── cm_native_combined_dataset_cleaned.parquet   # Same rows, Parquet (read by post-training scripts)
│   └── Raw-Data/                                # Original source datasets
│
├── Downstream_tasks/
│   ├── cross_lingual_hate.py                    # Hate speech detection
│   └── cross_lingual_sentiment.py               # Sentiment analysis
│
├── Interpretability/
│   ├── perplexity/
│   │   └── perplexity_encoder.py                # Code-mixed perplexity scores
│   ├── retrieval_scores/
│   │   ├── dot-product-retrieval-faiss.py       # FAISS hard-negative retrieval
│   │   ├── dot-product-retrieval-percentile.py  # Percentile-based retrieval
│   │   └── retrieval_utils.py
│   ├── similarity_scores/
│   │   ├── cka.py                               # CKA layer-wise analysis
│   │   ├── svcca.py                             # SVCCA analysis
│   │   ├── representations.py                   # Representation extraction
│   │   ├── Similarity_utils.py
│   │   └── utils.py
│   ├── Token_level_saliency/
│   │   └── Compute_RI.py                        # Rank-Inverse saliency scores
│   ├── TSNE/
│   │   └── TSNE.py                              # t-SNE geometry visualization
│   └── Uncertainty_scores/
│       └── uncertainty_scores.py                # Entropy-based uncertainty reduction
│
├── Post_alignment_training/
│   ├── mbert-trilingual.py                      # Trilingual post-training for the mBERT family
│   └── xlmr-trilingual.py                       # Trilingual post-training for the XLM-R family
│
├── Translation/                                 # Downstream data translation (Qwen2.5-72B-Instruct via vLLM)
│   ├── trans_hate/
│   │   ├── english.yaml
│   │   ├── hindi.yaml
│   │   ├── run_inference_hate.py
│   │   └── trans.sh
│   └── trans_sentiment/
│       ├── english.yaml
│       ├── hindi.yaml
│       ├── run_inference.py
│       └── trans.sh
│
├── requirements.txt
└── README.md
```

---

## Installation

**Python 3.9+ is required.** We recommend a virtual environment:

```bash
git clone https://github.com/debajyotimaz/tri_align_EMNLP_2026.git
cd tri_align_EMNLP_2026

python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

> **GPU note:** Every script falls back to CPU automatically. Training and representation extraction are much
> faster on CUDA. The translation pipeline needs a GPU server running **vLLM**; uncomment the `vllm` line in
> `requirements.txt` on that machine.

---

## Data

### Dataset on Hugging Face

The trilingual corpus is available on the Hugging Face Hub:

```python
from datasets import load_dataset

ds = load_dataset("debajyotimaz/trilingual-hinglish-corpus")
print(ds)          # train: 16,911 rows, test: 4,228 rows
print(ds["train"][0])   # {'id': ..., 'english': ..., 'hindi': ..., 'hinglish': ...}
```

The columns are `english`, `hindi` (Devanagari), `hinglish` (Romanized code-mixed), and `id`, which is the row
index in the original JSON file. The `train`/`test` split is the 80/20 split used for trilingual post-training
(`numpy.random.default_rng(123)`, `test_ratio=0.2`).

### Local copy

We built a unified trilingual corpus by merging three public parallel datasets. Each provides code-mixed–English
sentence pairs. Hindi comes from human annotation or machine translation.

| Source Dataset | Samples | Languages |
|---|---|---|
| CM-En Parallel (Dhar et al., 2018) | 6,096 | CM ↔ EN |
| PHINC (Srivastava and Singh, 2020) | 13,738 | CM ↔ EN |
| LinCE 2021 (Aguilar et al., 2020) | 8,060 | CM ↔ EN ↔ HI |
| **Final Trilingual Corpus** | **21,139** | **CM · EN · HI** |

Missing Hindi translations were generated with the **Google Translate API** and checked with **IndicTrans2**.
Duplicates and sentences with fewer than 5 tokens were removed. The cleaned corpus is at:

```
Data/Cleaned-Data/cleaned-trilingual-corpus/combined_dataset_cleaned.json      # fields: English, Hindi, Roman_Hinglish
Data/Cleaned-Data/cleaned-trilingual-corpus/cm_native_combined_dataset_cleaned.parquet   # same rows, same order
```

The downstream task data (hate speech and sentiment) is in `Data/Cleaned-Data/cleaned-hate-translated/` and
`Data/Cleaned-Data/cleaned-sentiment-translated/`. Each row holds the code-mixed `sentence`, a label
(`label` or `sentiment`), and its translations `translation_en` and `translation_hi`.

---

## Models

We evaluate two families of multilingual encoders, one BERT-based and one RoBERTa-based, together with their
Hinglish-adapted variants:

| Model | HuggingFace ID | Family |
|---|---|---|
| mBERT | `bert-base-multilingual-cased` | BERT |
| Hing-mBERT | `l3cube-pune/hing-bert` | BERT |
| Hing-mBERT-Mixed | `l3cube-pune/hing-bert-mixed` | BERT |
| XLM-R Base | `xlm-roberta-base` | RoBERTa |
| Hing-RoBERTa | `l3cube-pune/hing-roberta` | RoBERTa |
| Hing-RoBERTa-Mixed | `l3cube-pune/hing-roberta-mixed` | RoBERTa |

Model names and local checkpoint paths are set in the `TrainingConfig` dataclass or in the module-level
constants of each script. Most scripts hard-code data and output paths, so edit those before running.

---

## Reproducing Experiments

### 1. Cross-Lingual Alignment Probing (Section 4.1)

We measure how well the representations of one language retrieve the parallel sentences in another language.
Retrieval uses dot-product similarity over pre-extracted layer-wise representations.

**Percentile-based negative sampling** controls for sentence length so that retrieval is not trivial:
```bash
cd Interpretability/retrieval_scores
python dot-product-retrieval-percentile.py
```

**FAISS hard-negative sampling** mines semantically hard negatives with approximate nearest-neighbor search:
```bash
cd Interpretability/retrieval_scores
python dot-product-retrieval-faiss.py
```

### 2. Interpretability Analysis (Section 4.2)

**Layer-wise CKA:** Centered Kernel Alignment between the representation matrices of each language, per layer.
```bash
cd Interpretability/similarity_scores
python cka.py
```

**Layer-wise SVCCA** (Figure 9):
```bash
cd Interpretability/similarity_scores
python svcca.py
```

**Token-level saliency** (Figure 6):
```bash
cd Interpretability/Token_level_saliency
python Compute_RI.py
```

**Entropy-based uncertainty reduction** (Figure 7):
```bash
cd Interpretability/Uncertainty_scores
python uncertainty_scores.py
```

**t-SNE visualization** (Figure 3):
```bash
cd Interpretability/TSNE
python TSNE.py
```

**Perplexity** (Table 6):
```bash
cd Interpretability/perplexity
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

Both scripts use the same cached 80/20 train/test split (`split_seed=123`, `split_cache_dir`), so the two
model families are evaluated on identical held-out data.

> **Ablation (without alignment loss):** remove or zero out the `λ · L_align` term in the corresponding script.

**Training details**

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

### 4. Downstream Tasks (Appendix F)

**Sentiment analysis:**
```bash
cd Downstream_tasks
python cross_lingual_sentiment.py
```

**Hate speech detection:**
```bash
cd Downstream_tasks
python cross_lingual_hate.py
```

### 5. Translation Pipeline (Appendix F.2)

The English and Hindi versions of the downstream sentiment and hate speech data were generated with
**Qwen2.5-72B-Instruct** via vLLM. (The Hindi side of the trilingual corpus was produced separately with the
Google Translate API.) To reproduce:

```bash
cd Translation/trans_hate && bash trans.sh
cd Translation/trans_sentiment && bash trans.sh
```

---

## Results Summary

CLAS before and after Trilingual Post-Training Alignment. Scores are last-layer, with 10 negatives, averaged
over 3 seeds.

| Model | Before TPA | After TPA | Δ |
|---|---:|---:|---:|
| mBERT | 14.20 | **50.97** | +36.77 |
| Hing-mBERT | 17.01 | **51.07** | +34.06 |
| Hing-mBERT-Mixed | 34.95 | **42.51** | +7.56 |
| XLM-R Base | 39.53 | **47.50** | +7.97 |
| Hing-RoBERTa | 3.74 | **58.98** | +55.24 |
| Hing-RoBERTa-Mixed | 73.09 | **73.50** | +0.41 |

See the paper for the full layer-wise analyses and the downstream results.

---

## Citation

If you use this code or the trilingual corpus, please cite:

```bibtex
@article{mazumder2026neither,
  title   = {Neither Here Nor There: Cross-Lingual Representation Dynamics of Code-Mixed Text in Multilingual Encoders},
  author  = {Mazumder, Debajyoti and Pathak, Divyansh and Kodali, Prashant and Patro, Jasabanta},
  journal = {arXiv preprint arXiv:2603.19771},
  year    = {2026}
}
```

Please also cite the source datasets: CM-En Parallel (Dhar et al., 2018), PHINC (Srivastava and Singh, 2020),
and LinCE (Aguilar et al., 2020).

---

## Ethics Statement

This work studies social media text that may contain offensive or harmful language from the original datasets.
These instances are included only for research, to study linguistic phenomena in code-mixed settings. The
authors do not endorse or promote any harmful or abusive content. AI-based writing assistants were used for
grammar checking and language polishing while preparing the manuscript.

---

## Contact & License

For questions, please open an issue on [GitHub](https://github.com/debajyotimaz/tri_align_EMNLP_2026/issues)
or contact the authors listed in the [paper](https://arxiv.org/abs/2603.19771).

**License:** The trilingual corpus (curation and Hindi translations) is released under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) for research use. The original sentences
in `Data/` come from third-party datasets (CM-En Parallel, PHINC, LinCE / CALCS 2021) and remain subject to
their original terms.

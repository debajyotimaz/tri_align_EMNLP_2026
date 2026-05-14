# ============================================================
# Normal End-to-End Fine-tuning for Multiclass Sentiment Analysis
# ============================================================

import torch
import pandas as pd
import numpy as np
import random
import os
import json

from pathlib import Path
from collections import Counter
from tqdm import tqdm
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.cuda.amp import GradScaler, autocast
# ADDED: DataLoader for faster batching with pin_memory + num_workers
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# CONFIG
# ============================================================

TRAIN_CSV = Path("/data/sa_hineng_translated_train.csv")
VAL_CSV   = Path("/data/sa_hineng_translated_val.csv")
TEST_CSV  = Path("/data/sa_hineng_translated_test_clean.csv")

# CHANGED: all outputs now go to one absolute folder
# OUT_ROOT   = Path("./normal_sentiment_results_21k_new_LR")
# EXCEL_PATH = Path("./normal_sentiment_results_21k/results_mbert.xlsx")
# CKPT_DIR   = Path("./checkpoints")
OUT_ROOT   = Path("/output/Finetuning/normal_sentiment_results_21k_new_LR")
EXCEL_PATH = OUT_ROOT / "results_mbert.xlsx"
# Checkpoint directory — stores progress so a crash doesn't lose everything
CKPT_DIR   = OUT_ROOT / "checkpoints_fin"
# ADDED: predictions root directory — saves per-epoch predictions for val and test
PREDS_ROOT = OUT_ROOT / "predictions"

MODELS = {
    # "mbert": "bert-base-multilingual-cased",
    # "mbert-tri-aligned": "local-adress",
    # "mbert-ablation":"local-adress",
    # "xlmr-tri-aligned": "local-adress",
    # "xlmr-ablation":"local-adress",
    # "xlm-roberta-base": "xlm-roberta-base",

}

EPOCHS       = 10
LR           = 1e-5
# CHANGED: batch size 16 -> 32 for speed (safe with mixed precision)
# BATCH_SIZE   = 16
BATCH_SIZE   = 32
RANDOM_STATE = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = device == "cuda"   # mixed precision only on GPU

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

OUT_ROOT.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_PATH.parent.mkdir(parents=True, exist_ok=True)
# ADDED: create predictions root directory
PREDS_ROOT.mkdir(parents=True, exist_ok=True)

# ============================================================
# ADDED: helper to detect local paths (fixes HFValidationError
# for absolute paths like /data4/... passed to from_pretrained)
# ============================================================

def is_local_path(p):
    return os.path.isabs(p) or os.path.exists(p)

# ============================================================
# ADDED: DataLoader builder — pin_memory + num_workers for speed
# ============================================================

def make_dataloader(enc, labels, batch_size, shuffle=False):
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], labels)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        pin_memory=True,         # faster CPU->GPU transfer
        num_workers=2,           # parallel data loading
        persistent_workers=True
    )

# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def ckpt_key(train_col, model_tag):
    """Unique string key for a (train_col, model_tag) experiment."""
    return f"{train_col}__{model_tag}"

def ckpt_path(train_col, model_tag):
    key = ckpt_key(train_col, model_tag)
    return CKPT_DIR / f"{key}.json"

def load_checkpoint(train_col, model_tag):
    """
    Returns a dict with keys:
      - completed_epochs : list of epoch numbers already done
      - results          : list of result dicts already saved
    Returns None if no checkpoint exists.
    """
    p = ckpt_path(train_col, model_tag)
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return None

def save_checkpoint(train_col, model_tag, completed_epochs, results):
    p = ckpt_path(train_col, model_tag)
    with open(p, "w") as f:
        json.dump({
            "completed_epochs": completed_epochs,
            "results": results,
        }, f, indent=2)

def load_all_checkpointed_results():
    """Gather all previously saved results from every checkpoint file."""
    all_results = []
    for p in CKPT_DIR.glob("*.json"):
        with open(p, "r") as f:
            data = json.load(f)
            all_results.extend(data.get("results", []))
    return all_results

# ============================================================
# EVALUATION HELPER  (unchanged logic, + autocast + non_blocking)
# CHANGED: now accepts a DataLoader instead of raw enc+labels+batch_size
# Old signature: evaluate(model, enc, labels, batch_size, device)
# def evaluate(model, enc, labels, batch_size, device):
#     model.eval()
#     all_preds = []
#
#     with torch.no_grad():
#         for i in range(0, len(labels), batch_size):
#             input_ids      = enc["input_ids"][i:i+batch_size].to(device, non_blocking=True)
#             attention_mask = enc["attention_mask"][i:i+batch_size].to(device, non_blocking=True)
#
#             with autocast(enabled=USE_AMP):
#                 outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#
#             preds = outputs.logits.argmax(dim=1).cpu()
#             all_preds.extend(preds.tolist())
#
#     acc = accuracy_score(labels, all_preds)
#     _, _, f1, _ = precision_recall_fscore_support(
#         labels, all_preds, average="macro", zero_division=0
#     )
#     return acc, f1
# ============================================================

def evaluate(model, loader, device):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, y in loader:
            input_ids      = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            with autocast(enabled=USE_AMP):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            preds = outputs.logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    acc = accuracy_score(all_labels, all_preds)
    _, _, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return acc, f1

# ============================================================
# ADDED: evaluate_and_save_preds — same as evaluate() but also
# writes a CSV of per-sample predictions to save_path.
# Folder structure: predictions/<model_tag>/train_<train_col>/epoch_<N>/
#                   val_<test_col>.csv  /  test_<test_col>.csv
# Each CSV columns: true_label, pred_label, true_idx, pred_idx, correct
# save_path=None means skip saving (acts like plain evaluate).
# ============================================================

def evaluate_and_save_preds(model, loader, device, save_path=None):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, y in loader:
            input_ids      = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            with autocast(enabled=USE_AMP):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            preds = outputs.logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    acc = accuracy_score(all_labels, all_preds)
    _, _, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    # ADDED: save predictions CSV if a path is provided
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df = pd.DataFrame({
            "true_label": [ID2LABEL[i] for i in all_labels],
            "pred_label": [ID2LABEL[i] for i in all_preds],
            "true_idx":   all_labels,
            "pred_idx":   all_preds,
            "correct":    [int(p == l) for p, l in zip(all_preds, all_labels)]
        })
        pred_df.to_csv(save_path, index=False)

    return acc, f1

# ============================================================
# LOAD DATA ONCE
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)[["sentence", "translation_en", "translation_hi", "sentiment"]].copy()
val_df   = pd.read_csv(VAL_CSV)[["sentence", "translation_en", "translation_hi", "sentiment"]].copy()
test_df  = pd.read_csv(TEST_CSV)[["sentence", "translation_en", "translation_hi", "sentiment"]].copy()

train_df = train_df.rename(columns={"sentiment": "answer"})
val_df   = val_df.rename(columns={"sentiment": "answer"})
test_df  = test_df.rename(columns={"sentiment": "answer"})

train_df = train_df.fillna("")
val_df   = val_df.fillna("")
test_df  = test_df.fillna("")

label_counter = Counter(train_df["answer"])
LABELS        = sorted(label_counter.keys())
NUM_CLASSES   = len(LABELS)
LABEL_MAP     = {l: i for i, l in enumerate(LABELS)}
ID2LABEL      = {i: l for l, i in LABEL_MAP.items()}

print(f"Train size : {len(train_df)}")
print(f"Val size   : {len(val_df)}")
print(f"Test size  : {len(test_df)}")
print(f"Labels     : {LABELS}")

train_labels = torch.tensor([LABEL_MAP[x] for x in train_df["answer"]])
val_labels   = torch.tensor([LABEL_MAP[x] for x in val_df["answer"]])
test_labels  = torch.tensor([LABEL_MAP[x] for x in test_df["answer"]])

TEXT_COLS = ["sentence", "translation_en", "translation_hi"]

# ============================================================
# PRE-TOKENIZE: once per model, outside the train_col loop
# ============================================================
# Structure: all_train_encs[model_tag][col], same for val/test.
# This avoids re-tokenizing val/test 3× (once per train_col).

print("\nPre-tokenizing all splits for all models...")
all_train_encs = {}
all_val_encs   = {}
all_test_encs  = {}

for model_tag, hf_name in MODELS.items():
    print(f"  Tokenizing for {model_tag} ...")

    # ADDED: local_files_only=True for local paths (fixes HFValidationError)
    local = is_local_path(hf_name)
    load_kwargs = {"local_files_only": True} if local else {}

    # ADDED: try/except so one bad model skips instead of crashing the run
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_name, use_fast=True, **load_kwargs)
    except Exception as e:
        print(f"  [SKIP] Could not load tokenizer for {model_tag}: {e}")
        continue

    all_train_encs[model_tag] = {}
    all_val_encs[model_tag]   = {}
    all_test_encs[model_tag]  = {}

    for col in TEXT_COLS:
        all_train_encs[model_tag][col] = tokenizer(
            train_df[col].tolist(), padding=True, truncation=True, return_tensors="pt"
        )
        all_val_encs[model_tag][col] = tokenizer(
            val_df[col].tolist(), padding=True, truncation=True, return_tensors="pt"
        )
        all_test_encs[model_tag][col] = tokenizer(
            test_df[col].tolist(), padding=True, truncation=True, return_tensors="pt"
        )

print("Pre-tokenization done.\n")

# ADDED: build all DataLoaders once after pre-tokenization
# Structure: all_train_loaders[model_tag], all_val_loaders[model_tag][col], etc.
print("Building DataLoaders...")
all_train_loaders = {}
all_val_loaders   = {}
all_test_loaders  = {}

for model_tag in all_train_encs:  # only models that tokenized successfully
    all_train_loaders[model_tag] = {}
    all_val_loaders[model_tag]   = {}
    all_test_loaders[model_tag]  = {}

    for col in TEXT_COLS:
        all_train_loaders[model_tag][col] = make_dataloader(
            all_train_encs[model_tag][col], train_labels, BATCH_SIZE, shuffle=True
        )
        all_val_loaders[model_tag][col] = make_dataloader(
            all_val_encs[model_tag][col], val_labels, BATCH_SIZE, shuffle=False
        )
        all_test_loaders[model_tag][col] = make_dataloader(
            all_test_encs[model_tag][col], test_labels, BATCH_SIZE, shuffle=False
        )

print("DataLoaders ready.\n")

# ============================================================
# EXCEL SETUP  (resume-aware: reload if file exists)
# ============================================================

def build_excel_headers(ws):
    header_fill    = PatternFill("solid", start_color="1F4E79", end_color="1F4E79")
    subheader_fill = PatternFill("solid", start_color="2E75B6", end_color="2E75B6")
    header_font    = Font(name="Arial", bold=True, color="FFFFFF", size=11)
    subheader_font = Font(name="Arial", bold=True, color="FFFFFF", size=10)
    center         = Alignment(horizontal="center", vertical="center")

    ws.column_dimensions["A"].width = 25
    ws.column_dimensions["B"].width = 18
    ws.column_dimensions["C"].width = 8

    ws.merge_cells("A1:C1")
    ws["A1"] = "Experiment"
    ws["A1"].font = header_font
    ws["A1"].fill = header_fill
    ws["A1"].alignment = center

    col = 4
    for tc in TEXT_COLS:
        ws.merge_cells(start_row=1, start_column=col, end_row=1, end_column=col+1)
        cell = ws.cell(row=1, column=col, value=f"Test: {tc}")
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center
        ws.cell(row=1, column=col+1).fill = header_fill
        ws.column_dimensions[cell.column_letter].width = 14
        col += 2

    for c, label in zip(["A", "B", "C"], ["Model", "Train On", "Epoch"]):
        cell = ws[f"{c}2"]
        cell.value = label
        cell.font = subheader_font
        cell.fill = subheader_fill
        cell.alignment = center

    col = 4
    for tc in TEXT_COLS:
        for lbl in ["Accuracy", "Macro-F1"]:
            cell = ws.cell(row=2, column=col, value=lbl)
            cell.font = subheader_font
            cell.fill = subheader_fill
            cell.alignment = center
            col += 1

if EXCEL_PATH.exists():
    wb = load_workbook(EXCEL_PATH)
    ws = wb.active
    # find next empty row
    excel_row = ws.max_row + 1
    print(f"Resuming Excel file at row {excel_row}")
else:
    wb = Workbook()
    ws = wb.active
    ws.title = "Results"
    build_excel_headers(ws)
    excel_row = 3

alt_fill   = PatternFill("solid", start_color="D6E4F0", end_color="D6E4F0")
white_fill = PatternFill("solid", start_color="FFFFFF", end_color="FFFFFF")
cell_font  = Font(name="Arial", size=10)
center     = Alignment(horizontal="center", vertical="center")

# Load any results already saved to checkpoints
all_results = load_all_checkpointed_results()

# ============================================================
# LOOP: train on each text col, test on all three
# ============================================================

for train_col in TEXT_COLS:

    print(f"\n{'='*60}")
    print(f"TRAIN ON: {train_col}")
    print(f"{'='*60}")

    for model_tag, hf_name in MODELS.items():

        print(f"\n---- MODEL: {model_tag} ----")

        # ADDED: skip models that failed tokenization
        if model_tag not in all_train_loaders:
            print(f"  [SKIP] No DataLoader found for {model_tag} (tokenization failed earlier).")
            continue

        # ---- Check checkpoint ----
        ckpt = load_checkpoint(train_col, model_tag)
        if ckpt is not None:
            completed_epochs   = ckpt["completed_epochs"]   # e.g. [1,2,3]
            experiment_results = ckpt["results"]
        else:
            completed_epochs   = []
            experiment_results = []

        if len(completed_epochs) == EPOCHS:
            print(f"  Already fully completed ({EPOCHS} epochs). Skipping.")
            continue

        if completed_epochs:
            print(f"  Resuming from epoch {max(completed_epochs)+1} (epochs {completed_epochs} done).")

        # ---- DataLoaders (already built) ----
        train_loader = all_train_loaders[model_tag][train_col]
        val_loaders  = all_val_loaders[model_tag]
        test_loaders = all_test_loaders[model_tag]

        # ---- Model + optimizer ----
        # ADDED: local_files_only=True for local paths
        local = is_local_path(hf_name)
        load_kwargs = {"local_files_only": True} if local else {}

        # ADDED: try/except so one bad model skips instead of crashing the run
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                hf_name,
                num_labels=NUM_CLASSES,
                id2label=ID2LABEL,
                label2id=LABEL_MAP,
                **load_kwargs  # ADDED: local_files_only propagated here too
            ).to(device)
        except Exception as e:
            print(f"  [SKIP] Could not load model for {model_tag}: {e}")
            continue

        # ADDED: torch.compile for extra speed on PyTorch 2.0+
        if hasattr(torch, "compile"):
            model = torch.compile(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        scaler    = GradScaler(enabled=USE_AMP)

        # ================= TRAIN + EVAL PER EPOCH =================

        for epoch in range(EPOCHS):
            epoch_num = epoch + 1

            # Skip already-done epochs
            if epoch_num in completed_epochs:
                print(f"  Epoch {epoch_num}/{EPOCHS} already done, skipping.")
                continue

            # ---- TRAIN ----
            model.train()
            # REMOVED: manual randperm — DataLoader shuffle=True handles this
            # indices = torch.randperm(len(train_labels))

            total_loss  = 0
            num_batches = 0

            # CHANGED: iterate DataLoader instead of manual index slicing
            # Old loop: for i in tqdm(range(0, len(indices), BATCH_SIZE), ...):
            #               batch_idx      = indices[i:i+BATCH_SIZE]
            #               input_ids      = train_enc["input_ids"][batch_idx].to(device, non_blocking=True)
            #               attention_mask = train_enc["attention_mask"][batch_idx].to(device, non_blocking=True)
            #               y              = train_labels[batch_idx].to(device, non_blocking=True)
            for input_ids, attention_mask, y in tqdm(
                train_loader,
                desc=f"Epoch {epoch_num}/{EPOCHS} [{model_tag} | train={train_col}]",
                leave=False
            ):
                input_ids      = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                y              = y.to(device, non_blocking=True)

                optimizer.zero_grad()

                with autocast(enabled=USE_AMP):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=y
                    )
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss  += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches

            # ---- EVAL ----
            val_scores   = {}
            epoch_scores = {}

            # CHANGED: use evaluate_and_save_preds instead of evaluate
            # saves val predictions to predictions/<model_tag>/train_<train_col>/epoch_<N>/val_<test_col>.csv
            for test_col in TEXT_COLS:
                # CHANGED: pass loader instead of enc+labels+batch_size
                # acc, f1 = evaluate(model, val_encs[test_col], val_labels, BATCH_SIZE, device)
                val_pred_path = (
                    PREDS_ROOT / model_tag / f"train_{train_col}" /
                    f"epoch_{epoch_num}" / f"val_{test_col}.csv"
                )
                acc, f1 = evaluate_and_save_preds(model, val_loaders[test_col], device, save_path=val_pred_path)
                val_scores[test_col] = {"acc": round(acc, 4), "f1": round(f1, 4)}

            # CHANGED: use evaluate_and_save_preds instead of evaluate
            # saves test predictions to predictions/<model_tag>/train_<train_col>/epoch_<N>/test_<test_col>.csv
            for test_col in TEXT_COLS:
                # CHANGED: pass loader instead of enc+labels+batch_size
                # acc, f1 = evaluate(model, test_encs[test_col], test_labels, BATCH_SIZE, device)
                test_pred_path = (
                    PREDS_ROOT / model_tag / f"train_{train_col}" /
                    f"epoch_{epoch_num}" / f"test_{test_col}.csv"
                )
                acc, f1 = evaluate_and_save_preds(model, test_loaders[test_col], device, save_path=test_pred_path)
                epoch_scores[test_col] = {"acc": round(acc, 4), "f1": round(f1, 4)}

            val_str = "  |  ".join([
                f"{tc}: Acc={val_scores[tc]['acc']:.4f} F1={val_scores[tc]['f1']:.4f}"
                for tc in TEXT_COLS
            ])
            scores_str = "  |  ".join([
                f"{tc}: Acc={epoch_scores[tc]['acc']:.4f} F1={epoch_scores[tc]['f1']:.4f}"
                for tc in TEXT_COLS
            ])
            print(f"Epoch {epoch_num}/{EPOCHS} | Loss: {avg_loss:.4f}")
            print(f"  VAL  | {val_str}")
            print(f"  TEST | {scores_str}")

            # ---- Collect result ----
            result = {
                "model":    model_tag,
                "train_on": train_col,
                "epoch":    epoch_num,
                **{f"val_{tc}_acc": val_scores[tc]["acc"]  for tc in TEXT_COLS},
                **{f"val_{tc}_f1":  val_scores[tc]["f1"]   for tc in TEXT_COLS},
                **{f"{tc}_acc":     epoch_scores[tc]["acc"] for tc in TEXT_COLS},
                **{f"{tc}_f1":      epoch_scores[tc]["f1"]  for tc in TEXT_COLS},
            }
            all_results.append(result)
            experiment_results.append(result)

            # ---- Write to Excel ----
            fill = alt_fill if excel_row % 2 == 0 else white_fill
            for c, val in zip([1, 2, 3], [model_tag, train_col, epoch_num]):
                cell = ws.cell(row=excel_row, column=c, value=val)
                cell.font = cell_font
                cell.fill = fill
                cell.alignment = center

            col = 4
            for tc in TEXT_COLS:
                for val in [epoch_scores[tc]["acc"], epoch_scores[tc]["f1"]]:
                    c = ws.cell(row=excel_row, column=col, value=val)
                    c.font = cell_font
                    c.fill = fill
                    c.alignment = center
                    col += 1

            excel_row += 1

            # ---- Save checkpoint after every epoch ----
            # If the script crashes mid-run, we resume from here next time.
            completed_epochs.append(epoch_num)
            save_checkpoint(train_col, model_tag, completed_epochs, experiment_results)

            # ---- Save Excel after every epoch (safe incremental save) ----
            # CHANGED: atomic .tmp -> os.replace() so a crash never corrupts the file
            # wb.save(EXCEL_PATH)
            tmp = str(EXCEL_PATH) + ".tmp"
            wb.save(tmp)
            os.replace(tmp, EXCEL_PATH)

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

# ============================================================
# TERMINAL SUMMARY TABLE
# ============================================================

print(f"\n{'='*80}")
print("FINAL SUMMARY TABLE")
print(f"{'='*80}")
hdr = f"{'Model':<22} {'TrainOn':<20} {'Ep':<5}"
for tc in TEXT_COLS:
    hdr += f"  {tc[:6]+'-Acc':<12}{tc[:6]+'-F1':<12}"
print(hdr)
print("-" * len(hdr))
for r in sorted(all_results, key=lambda x: (x["train_on"], x["model"], x["epoch"])):
    row_str = f"{r['model']:<22} {r['train_on']:<20} {r['epoch']:<5}"
    for tc in TEXT_COLS:
        row_str += f"  {r[f'{tc}_acc']:<12.4f}{r[f'{tc}_f1']:<12.4f}"
    print(row_str)

# ============================================================
# FINAL EXCEL SAVE
# ============================================================

# CHANGED: atomic save for final write too
# wb.save(EXCEL_PATH)
tmp = str(EXCEL_PATH) + ".tmp"
wb.save(tmp)
os.replace(tmp, EXCEL_PATH)
print(f"\nExcel results saved to: {EXCEL_PATH}")
print(f"Predictions saved to  : {PREDS_ROOT}")
print("\nALL DONE.")
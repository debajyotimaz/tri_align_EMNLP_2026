import json
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from captum.attr import IntegratedGradients

# ============================================================
# PATHS
# ============================================================
DATA_PATH = "/path/combined-data-with-lid.jsonl"
OUT_DIR = "/output/combined"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================
# ENCODER MODELS
# ============================================================
MODELS = {
    # "mbert": "bert-base-multilingual-cased",
    # "hing-mbert": "l3cube-pune/hing-mbert",
    # "hing-mbert-mixed": "l3cube-pune/hing-mbert-mixed",
    "hing-roberta": "l3cube-pune/hing-roberta",
    "hing-roberta-mixed": "l3cube-pune/hing-roberta-mixed",
    "xlmr-base": "xlm-roberta-base",
}

# ============================================================
# CONFIG (ENCODER-SAFE)
# ============================================================
BATCH_SIZE = 4
MAX_LEN = 256
IG_STEPS = 8   # enough for ranking-based RI

# ============================================================
# LOAD DATA ONCE
# ============================================================
samples = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line)
            if "sentence" in obj and "label" in obj:
                samples.append(obj)
        except json.JSONDecodeError:
            continue

print(f"Loaded total samples: {len(samples)}")

# ============================================================
# MAIN LOOP OVER MODELS
# ============================================================
for TAG, MODEL_NAME in MODELS.items():

    print(f"\n==============================")
    print(f"Running ENCODER RI for: {TAG}")
    print(f"Model: {MODEL_NAME}")
    print(f"==============================")

    OUT_PATH = f"{OUT_DIR}/lid_RI_{TAG}.jsonl"

    # ========================================================
    # RESUME SUPPORT
    # ========================================================
    processed_sentences = set()
    if Path(OUT_PATH).exists():
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    processed_sentences.add(json.loads(line)["sentence"])
                except Exception:
                    continue

    print(f"Resuming — already processed: {len(processed_sentences)}")

    # ========================================================
    # TOKENIZER
    # ========================================================
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ========================================================
    # MODEL (SAFE FOR TORCH < 2.6)
    # ========================================================
    try:
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,
            device_map={"": 0},
            use_safetensors=True   # <<< critical fix
        )
    except Exception as e:
        print(f"❌ Skipping {TAG} (no safetensors available)")
        print(e)
        continue

    model.eval()
    device = model.device

    # ========================================================
    # FORWARD FUNCTION (ENCODER)
    # ========================================================
    def forward_func(inputs_embeds, attention_mask):
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        cls_repr = outputs.last_hidden_state[:, 0, :]
        return cls_repr.norm(dim=-1)

    ig = IntegratedGradients(forward_func)

    # ========================================================
    # RI COMPUTATION
    # ========================================================
    def compute_RI_batch(texts):
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device).detach()

        # IMPORTANT: embeddings must match model dtype (FP16)
        inputs_embeds = model.get_input_embeddings()(input_ids)
        baseline_embeds = torch.zeros_like(inputs_embeds)

        attributions = ig.attribute(
            inputs=inputs_embeds,
            baselines=baseline_embeds,
            additional_forward_args=(attention_mask,),
            n_steps=IG_STEPS
        )

        token_attr = attributions.abs().sum(dim=-1)

        batch_results = []
        for b in range(token_attr.size(0)):
            valid_len = attention_mask[b].sum().item()

            attrs = token_attr[b][:valid_len].detach().cpu().numpy()
            tokens = tokenizer.convert_ids_to_tokens(
                input_ids[b][:valid_len].detach().cpu()
            )

            ranks = (-attrs).argsort().argsort() + 1
            RI = 1.0 / ranks

            batch_results.append([
                {
                    "token": tok,
                    "attribution": float(attr),
                    "rank": int(rank),
                    "RI": float(ri)
                }
                for tok, attr, rank, ri in zip(tokens, attrs, ranks, RI)
            ])

        # cleanup
        del inputs_embeds, baseline_embeds, attributions, token_attr
        torch.cuda.empty_cache()

        return batch_results

    # ========================================================
    # STREAMING WRITE (RESUMABLE)
    # ========================================================
    written = len(processed_sentences)
    skipped = 0
    buffer = []

    with open(OUT_PATH, "a", encoding="utf-8") as fout:
        for sample in tqdm(samples, desc=f"Processing {TAG}"):
            sentence = sample["sentence"].strip()
            label = sample["label"]

            if not sentence:
                skipped += 1
                continue

            if sentence in processed_sentences:
                continue

            buffer.append((sentence, label))

            if len(buffer) == BATCH_SIZE:
                texts = [x[0] for x in buffer]
                ri_outputs = compute_RI_batch(texts)

                for (sent, lab), ri in zip(buffer, ri_outputs):
                    fout.write(json.dumps({
                        "sentence": sent,
                        "label": lab,
                        "RI_tokens": ri
                    }, ensure_ascii=False) + "\n")
                    processed_sentences.add(sent)
                    written += 1

                buffer = []

        # leftovers
        if buffer:
            texts = [x[0] for x in buffer]
            ri_outputs = compute_RI_batch(texts)

            for (sent, lab), ri in zip(buffer, ri_outputs):
                fout.write(json.dumps({
                    "sentence": sent,
                    "label": lab,
                    "RI_tokens": ri
                }, ensure_ascii=False) + "\n")
                written += 1

    print(f"✅ Total written    : {written}")
    print(f"⚠️ Samples skipped : {skipped}")
    print(f"📁 Saved to        : {OUT_PATH}")

    # ========================================================
    # CLEANUP
    # ========================================================
    del model
    torch.cuda.empty_cache()

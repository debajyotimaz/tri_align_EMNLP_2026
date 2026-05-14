import yaml
import json
import os
import time
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# ========= LOAD CONFIG =========

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ========= LOAD DATA =========

def load_data(cfg):
    data_cfg = cfg["data"]
    source   = data_cfg["source"]

    if source == "file":
        path = data_cfg["input_path"]
        ext  = Path(path).suffix.lower()
        if ext == ".parquet":
            df = pd.read_parquet(path)
        elif ext == ".csv":
            df = pd.read_csv(path)
        elif ext in [".jsonl", ".json"]:
            df = pd.read_json(path, lines=(ext == ".jsonl"))
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    else:
        raise ValueError(f"Unsupported data source: {source}")

    id_col   = data_cfg["fields"].get("id", None)
    text_col = data_cfg["fields"]["text"]

    print(f"Loaded {len(df)} rows | columns: {df.columns.tolist()}")
    return df, id_col, text_col


# ========= BUILD PROMPTS =========

def build_prompts(df, id_col, text_col, cfg, tokenizer):
    data_cfg = cfg["data"]
    mode     = data_cfg["mode"]
    chat_cfg = data_cfg.get("chat", {})
    system   = chat_cfg.get("system", "You are a helpful assistant.")
    max_seq  = data_cfg.get("max_seq_length", None)

    prompts = []
    ids     = []

    for _, row in df.iterrows():
        rid  = str(row[id_col]) if id_col and id_col in row else str(_)
        text = str(row[text_col])

        if mode == "chat":
            messages = [
                {"role": "system",    "content": system},
                {"role": "user",      "content": text},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            template = data_cfg.get("prompt_template", None)
            prompt   = template.format(prompt=text) if template else text

        if max_seq:
            tokens = tokenizer.encode(prompt)
            if len(tokens) > max_seq:
                tokens = tokens[:max_seq]
                prompt = tokenizer.decode(tokens, skip_special_tokens=False)

        prompts.append(prompt)
        ids.append(rid)

    return prompts, ids


# ========= PARSE MODEL OUTPUT =========

def parse_output(text):
    """Try to extract JSON from model output."""
    text = text.strip()
    # Find first { ... } block
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    return {"raw_output": text}


# ========= MAIN =========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---- Output dir ----
    out_cfg  = cfg["output"]
    out_dir  = os.path.join(out_cfg["base_dir"], out_cfg["run_name"])
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "results.jsonl")

    # Save config copy
    if out_cfg.get("save_config_copy", False):
        with open(os.path.join(out_dir, "config.yaml"), "w") as f:
            yaml.dump(cfg, f)

    print(f"\n{'='*60}")
    print(f"Output dir : {out_dir}")
    print(f"{'='*60}\n")

    # ---- Load tokenizer ----
    model_cfg = cfg["model"]
    print(f"Loading tokenizer: {model_cfg['name_or_path']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        revision=model_cfg.get("revision", None),
    )

    # ---- Load data ----
    df, id_col, text_col = load_data(cfg)

    # ---- Build prompts ----
    prompts, ids = build_prompts(df, id_col, text_col, cfg, tokenizer)

    # ---- Load vLLM model ----
    print(f"\nLoading vLLM model: {model_cfg['name_or_path']}")
    llm = LLM(
        model                  = model_cfg["name_or_path"],
        dtype                  = model_cfg.get("dtype", "auto"),
        revision               = model_cfg.get("revision", None),
        trust_remote_code      = model_cfg.get("trust_remote_code", True),
        gpu_memory_utilization = model_cfg.get("gpu_memory_utilization", 0.9),
        max_model_len          = model_cfg.get("max_model_len", 8192),
        quantization           = "gptq_marlin",
    )

    # ---- Sampling params ----
    gen_cfg = cfg["generation"]
    stop    = gen_cfg.get("stop", [])
    if tokenizer.eos_token and tokenizer.eos_token not in stop:
        stop.append(tokenizer.eos_token)

    sampling_params = SamplingParams(
        max_tokens         = gen_cfg.get("max_new_tokens", 512),
        temperature        = gen_cfg.get("temperature", 0.7),
        top_p              = gen_cfg.get("top_p", 0.9),
        repetition_penalty = gen_cfg.get("repetition_penalty", 1.1),
        stop               = stop,
    )

    # ---- Inference in batches ----
    batch_size    = cfg["runtime"].get("batch_size", 8)
    total         = len(prompts)
    results       = []
    checkpoint_every = 100

    print(f"\nRunning inference on {total} samples (batch_size={batch_size})\n")

    with open(out_file, "w", encoding="utf-8") as fout:
        for start in tqdm(range(0, total, batch_size), desc="Batches"):
            batch_prompts = prompts[start : start + batch_size]
            batch_ids     = ids[start : start + batch_size]
            batch_rows    = df.iloc[start : start + batch_size]

            outputs = llm.generate(batch_prompts, sampling_params)

            for i, output in enumerate(outputs):
                raw_text   = output.outputs[0].text
                parsed     = parse_output(raw_text)
                row        = batch_rows.iloc[i]

                record = {
                    "id":                  batch_ids[i],
                    "original_index":      int(row.get("original_index", start + i)),
                    "sentence":            str(row.get("sentence", "")),
                    "label":               str(row.get("label", "")),
                    "target_language":     str(row.get("target_language", "")),
                    "translated_sentence": parsed.get("translated_sentence", ""),
                    "output_label":        parsed.get("label", ""),
                    "language":            parsed.get("language", ""),
                    "raw_output":          raw_text,
                }

                results.append(record)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            # ---- Live checkpoint print ----
            done = min(start + batch_size, total)
            if done % checkpoint_every == 0 or done == total:
                print(f"\n--- Checkpoint [{done}/{total}] ---")
                last = results[-1]
                print(f"  id             : {last['id']}")
                print(f"  sentence       : {last['sentence']}")
                print(f"  label          : {last['label']}")
                print(f"  target_lang    : {last['target_language']}")
                print(f"  translation    : {last['translated_sentence']}")
                print(f"  output saved to: {out_file}")
                print()

    # ---- Save final combined JSON ----
    final_json = os.path.join(out_dir, "results.json")
    with open(final_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! {len(results)} results saved.")
    print(f"  JSONL : {out_file}")
    print(f"  JSON  : {final_json}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
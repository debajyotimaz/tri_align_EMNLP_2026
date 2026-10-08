# utils.py
import torch
import numpy as np
from tqdm import tqdm


def extract_encoder_representations(model, tokenizer, texts, batch_size=32, max_len=256, device="cuda"):
    """
    Extract CLS token representations from all hidden layers of an encoder model.
    """
    buffers = None
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting", leave=False):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True)

            hs = out.hidden_states
            if buffers is None:
                buffers = [[] for _ in range(len(hs))]

            for l in range(len(hs)):
                buffers[l].append(hs[l][:, 0].cpu())

            del enc, out, hs
            torch.cuda.empty_cache()

    return [torch.cat(x).numpy() for x in buffers]


def extract_decoder_representations(model, tokenizer, texts, batch_size=8, max_len=256):
    """
    Extract last token representations from all hidden layers of a decoder model.
    """
    buffers = None
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting", leave=False):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}

            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states

            if buffers is None:
                buffers = [[] for _ in range(len(hs))]

            idx = enc["attention_mask"].sum(1) - 1
            b = torch.arange(idx.size(0), device=idx.device)

            for l in range(len(hs)):
                buffers[l].append(hs[l][b, idx].cpu())

            del enc, out, hs
            torch.cuda.empty_cache()

    return [torch.cat(x).numpy() for x in buffers]


def save_layer_representations(representations, output_dir):
    """
    Save layer-wise representations to disk.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for layer_idx, layer_reps in enumerate(representations):
        filepath = os.path.join(output_dir, f"layer_{layer_idx:02d}.npy")
        np.save(filepath, layer_reps)


def load_hinglish_dataset(data_path):
    """
    Load Hinglish dataset (English, Hindi, Roman_Hinglish)
    Returns:
        data, sentences_dict, english, hindi, roman_hinglish
    """
    import json

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = {
        "CodeMixed": [x["Roman_Hinglish"] for x in data],
        "English": [x["English"] for x in data],
        "Hindi": [x["Hindi"] for x in data],
    }

    english = [x["English"] for x in data]
    hindi = [x["Hindi"] for x in data]
    roman_hinglish = [x["Roman_Hinglish"] for x in data]

    return data, sentences, english, hindi, roman_hinglish
import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForPreTraining, get_linear_schedule_with_warmup
import pandas as pd
from tqdm import tqdm
import logging
from dataclasses import dataclass
from typing import Optional

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Seed
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainingConfig:
    model_name: str = "l3cube-pune/hing-mbert-mixed"
    data_path: str = "../../data/cm_native_combined_dataset_cleaned.parquet"
    output_dir: str = "mbert_trilingual_aligned_seed_123"

    batch_size: int = 16
    learning_rate: float = 1e-5
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    max_length: int = 512
    mlm_probability: float = 0.15

    align_weight: float = 0.3
    max_grad_norm: float = 1.0
    seed: int = 123


    # --- NEW ---
    split_seed: int = 123                          # fixed so mBERT and XLM-R share same split
    test_ratio: float = 0.2                       # 80-20 split
    split_cache_dir: str = "../data/splits_random_123"       # where train/test indices are saved as .npy
    best_model_dir: str = "mbert_trilingual_aligned_best_seed_123"  # saves best epoch checkpoint here
# =========================
# Cross Lingual Consistency
# =========================
@torch.no_grad()
def get_sentence_embeddings(model, tokenizer, sentences, device, max_length=512, batch_size=32):
    model.eval()
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        enc = tokenizer(sentences[i:i+batch_size], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        outputs = model.bert(**enc, return_dict=True)
        cls = F.normalize(outputs.last_hidden_state[:,0], dim=1)
        embeddings.append(cls.cpu())
    return torch.cat(embeddings)

def retrieval_accuracy(src_emb, tgt_emb):
    sim = torch.matmul(src_emb, tgt_emb.T)
    preds = sim.argmax(dim=1)
    labels = torch.arange(len(src_emb))
    return (preds == labels).float().mean().item()

def cross_lingual_consistency(model, tokenizer, df, device, sample_size=1000):
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    eng = df["English"].astype(str).tolist()
    hin = df["Hindi"].astype(str).tolist()
    hing = df["Roman_Hinglish"].astype(str).tolist()

    e = get_sentence_embeddings(model, tokenizer, eng, device)
    h = get_sentence_embeddings(model, tokenizer, hin, device)
    c = get_sentence_embeddings(model, tokenizer, hing, device)

    return {
        "EN-HI": retrieval_accuracy(e,h),
        "EN-HING": retrieval_accuracy(e,c),
        "HI-HING": retrieval_accuracy(h,c),
        "AVG": (retrieval_accuracy(e,h)+retrieval_accuracy(e,c)+retrieval_accuracy(h,c))/3
    }

# -----------------------------
# Train/Test Split
# -----------------------------
def get_or_create_splits(data_path, test_ratio, split_seed, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    train_path = os.path.join(cache_dir, "train_indices.npy")
    test_path  = os.path.join(cache_dir, "test_indices.npy")
    if os.path.exists(train_path) and os.path.exists(test_path):
        logger.info("Loading cached splits...")
        return np.load(train_path), np.load(test_path)
    df = pd.read_parquet(data_path)
    indices = np.arange(len(df))
    rng = np.random.default_rng(split_seed)
    rng.shuffle(indices)
    split = int(len(indices) * (1 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]
    np.save(train_path, train_idx)
    np.save(test_path, test_idx)
    logger.info(f"Splits created — train: {len(train_idx)}, test: {len(test_idx)}")
    return train_idx, test_idx

# -----------------------------
# Dataset
# -----------------------------
class TrilingualDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, mlm_probability=0.15, indices=None):
        # self.df = pd.read_parquet(data_path)
        df = pd.read_parquet(data_path)
        self.df = df.iloc[indices].reset_index(drop=True) if indices is not None else df       #change for split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        logger.info(f"Loaded {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def mask_tokens(self, inputs, special_tokens_mask):
        labels = inputs.clone()
        prob = torch.full(labels.shape, self.mlm_probability)
        prob.masked_fill_(special_tokens_mask.bool(), 0)
        masked = torch.bernoulli(prob).bool()
        labels[~masked] = -100

        replace = torch.bernoulli(torch.full(labels.shape,0.8)).bool() & masked
        inputs[replace] = self.tokenizer.mask_token_id

        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        rand = torch.bernoulli(torch.full(labels.shape,0.5)).bool() & masked & ~replace
        inputs[rand] = random_words[rand]

        return inputs, labels

    def encode_single(self, text):
        enc = self.tokenizer(text, padding="max_length", truncation=True,
                             max_length=self.max_length, return_tensors="pt")
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def create_pair(self, sent_a, sent_b, positive=True):
        # RANDOMIZE DIRECTION
        if random.random() < 0.5:
            sent_a, sent_b = sent_b, sent_a

        enc = self.tokenizer(sent_a, sent_b, padding="max_length", truncation=True,
                             max_length=self.max_length, return_tensors="pt",
                             return_special_tokens_mask=True)

        ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        type_ids = enc["token_type_ids"].squeeze(0)
        sp_mask = enc["special_tokens_mask"].squeeze(0)

        ids, labels = self.mask_tokens(ids, sp_mask)
        nsp = torch.tensor(1 if positive else 0)

        return ids, mask, type_ids, labels, nsp

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        e, h, c = str(row["English"]), str(row["Hindi"]), str(row["Roman_Hinglish"])

        positives = [(e,h),(e,c),(h,c)]
        rand_idx = random.randint(0,len(self.df)-1)
        while rand_idx==idx: rand_idx=random.randint(0,len(self.df)-1)
        neg = (e,str(self.df.iloc[rand_idx]["Hindi"]))

        pair = random.choice(positives+[neg])
        positive = pair in positives

        ids,mask,type_ids,labels,nsp = self.create_pair(pair[0],pair[1],positive)

        # also return triplet encodings for alignment loss
        e_ids,e_mask = self.encode_single(e)
        h_ids,h_mask = self.encode_single(h)
        c_ids,c_mask = self.encode_single(c)

        return {
            "input_ids":ids,"attention_mask":mask,"token_type_ids":type_ids,
            "labels":labels,"next_sentence_label":nsp,
            "eng_ids":e_ids,"eng_mask":e_mask,
            "hin_ids":h_ids,"hin_mask":h_mask,
            "hing_ids":c_ids,"hing_mask":c_mask
        }

# -----------------------------
# Alignment Loss
# -----------------------------
def alignment_loss(model,batch,device):
    def embed(ids,mask):
        out=model.bert(input_ids=ids,attention_mask=mask,return_dict=True)
        return F.normalize(out.last_hidden_state[:,0],dim=1)

    e=embed(batch["eng_ids"],batch["eng_mask"])
    h=embed(batch["hin_ids"],batch["hin_mask"])
    c=embed(batch["hing_ids"],batch["hing_mask"])

    return ((1-(e*h).sum(1)).mean()+(1-(e*c).sum(1)).mean()+(1-(h*c).sum(1)).mean())/3

# -----------------------------
# Training
# -----------------------------
def train(config:TrainingConfig):
    set_seed(config.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer=BertTokenizer.from_pretrained(config.model_name)
    model=BertForPreTraining.from_pretrained(config.model_name).to(device)

    # dataset=TrilingualDataset(config.data_path,tokenizer,config.max_length)
    # loader=DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=4)
    train_idx, test_idx = get_or_create_splits(
    config.data_path, config.test_ratio, config.split_seed, config.split_cache_dir)
    full_df = pd.read_parquet(config.data_path)
    test_df  = full_df.iloc[test_idx].reset_index(drop=True)
    dataset = TrilingualDataset(config.data_path, tokenizer, config.max_length, indices=train_idx)
    loader  = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    # --- results dict initialized before baseline ---
    results = {"baseline": None, "epochs": []}
    results_path = f"{config.output_dir}_results_seed_123.json"

    # --- FIX: ensure output and best_model directories exist before saving ---
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.best_model_dir, exist_ok=True)

    logger.info("Baseline evaluation...")
    # print(cross_lingual_consistency(model, tokenizer, test_df, device))
    baseline = cross_lingual_consistency(model, tokenizer, test_df, device)
    results["baseline"] = baseline
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(baseline)
    # --- FIX: flush logs so baseline is visible immediately ---
    for handler in logging.getLogger().handlers:
        handler.flush()

    optimizer=AdamW(model.parameters(),lr=config.learning_rate)
    # scheduler=get_linear_schedule_with_warmup(optimizer,0,len(loader)*config.num_epochs)   # old: warmup=0
    # --- FIX: use warmup_ratio like RoBERTa script ---
    warmup = int(len(loader) * config.num_epochs * config.warmup_ratio)
    scheduler=get_linear_schedule_with_warmup(optimizer, warmup, len(loader)*config.num_epochs)

    best_avg   = 0.0
    best_epoch = -1

    for epoch in range(config.num_epochs):
        model.train()
        for batch in tqdm(loader,desc=f"Epoch {epoch+1}"):
            batch={k:v.to(device) for k,v in batch.items()}

            out=model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],
                      token_type_ids=batch["token_type_ids"],labels=batch["labels"],
                      next_sentence_label=batch["next_sentence_label"])

            align=alignment_loss(model,batch,device)
            loss=out.loss+config.align_weight*align

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),config.max_grad_norm)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        logger.info(f"Epoch {epoch+1} evaluation:")
        scores = cross_lingual_consistency(model, tokenizer, test_df, device)
        results["epochs"].append({"epoch": epoch+1, **scores})
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(scores)
        # --- FIX: flush after every epoch so scores appear live in logs ---
        for handler in logging.getLogger().handlers:
            handler.flush()

        if scores["AVG"] > best_avg:
            best_avg   = scores["AVG"]
            best_epoch = epoch + 1
            model.save_pretrained(config.best_model_dir)
            tokenizer.save_pretrained(config.best_model_dir)
            logger.info(f"  → Best model saved at epoch {best_epoch} (AVG={best_avg:.4f})")
            # --- FIX: flush after best model save so log line appears immediately ---
            for handler in logging.getLogger().handlers:
                handler.flush()

    # model.save_pretrained(config.output_dir)
    # tokenizer.save_pretrained(config.output_dir)
    results["best_epoch"] = best_epoch
    results["best_avg"] = best_avg
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Training complete. Best epoch: {best_epoch} | Best AVG: {best_avg:.4f}")
    logger.info(f"Best model saved at: {config.best_model_dir}")


if __name__=="__main__":
    train(TrainingConfig())
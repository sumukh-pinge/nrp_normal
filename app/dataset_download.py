#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def make_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    f = open(log_path, "a", buffering=1)

    def log(msg):
        line = str(msg)
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()

    return log


def build_passage_embeddings_mpnet(data_dir, batch_size, log):
    corpus_path = os.path.join(data_dir, "corpus.jsonl")
    out_path = os.path.join(data_dir, "sample_passage_embeddings_hotpotqa.npy")

    # Sanity checks
    if not os.path.exists(corpus_path):
        log(f"❌ corpus.jsonl not found at: {corpus_path}")
        sys.exit(1)

    if os.path.exists(out_path):
        log(f"🔄 Found existing embeddings at: {out_path}")
        emb = np.load(out_path)
        log(f"   Loaded existing embeddings with shape {emb.shape}")
        return

    # Device + model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}")
    log("Loading encoder: all-mpnet-base-v2")
    encoder = SentenceTransformer("all-mpnet-base-v2", device=device)

    # Load passages
    log(f"Reading corpus from: {corpus_path}")
    passage_texts = []
    passage_ids = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading corpus"):
            obj = json.loads(line)
            passage_ids.append(obj["_id"])
            text = (obj.get("title", "") + " " + obj.get("text", "")).strip()
            passage_texts.append(text)

    n = len(passage_texts)
    log(f"Total passages: {n}")

    # Encode in batches
    all_chunks = []
    log(f"Encoding passages with batch_size={batch_size} ...")
    for i in tqdm(range(0, n, batch_size), desc="Encoding batches"):
        batch = passage_texts[i:i + batch_size]
        emb = encoder.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        all_chunks.append(emb)

    embeddings = np.vstack(all_chunks).astype("float32")
    log(f"Embeddings shape: {embeddings.shape}")

    # Save
    np.save(out_path, embeddings)
    log(f"✅ Saved embeddings to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate MPNet passage embeddings for hotpotqa_mpnet on PVC."
    )
    parser.add_argument("--data_root", default="/mnt/work/datasets")
    parser.add_argument("--dataset", default="hotpotqa_mpnet")
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    data_dir = os.path.join(args.data_root, args.dataset)
    os.makedirs(data_dir, exist_ok=True)

    log_path = os.path.join(data_dir, f"gen_{args.dataset}_mpnet.log")
    log = make_logger(log_path)

    log(f"=== MPNet embedding generation for {args.dataset} ===")
    log(f"Data dir : {data_dir}")
    log(f"Log file : {log_path}")

    build_passage_embeddings_mpnet(data_dir, args.batch_size, log)

    log("🎉 Done.")
    log("=========================================")


if __name__ == "__main__":
    main()

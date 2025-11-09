#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def make_logger(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    f = open(log_path, "a", buffering=1)

    def log(msg: str):
        line = str(msg)
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()

    return log


def resolve_out_path(data_dir: str, dataset: str, log):
    """
    Pick a filename consistent with your NRP code expectations.
    """
    # Explicit mappings for your known setups
    if dataset in ("hotpotqa", "hotpotqa_mpnet"):
        name = "sample_passage_embeddings_hotpotqa.npy"
    elif dataset in ("beir_nq", "beir_nq_mpnet"):
        name = "sample_passage_embeddings_nq.npy"
    else:
        # Fallback: dataset-specific
        name = f"sample_passage_embeddings_{dataset}.npy"
    out_path = os.path.join(data_dir, name)
    log(f"[config] Using output file: {out_path}")
    return out_path


def build_passage_embeddings_mpnet(data_dir: str, dataset: str, batch_size: int, log):
    corpus_path = os.path.join(data_dir, "corpus.jsonl")
    out_path = resolve_out_path(data_dir, dataset, log)

    # Sanity checks
    if not os.path.exists(corpus_path):
        log(f"❌ corpus.jsonl not found at: {corpus_path}")
        sys.exit(1)

    if os.path.exists(out_path):
        try:
            emb = np.load(out_path)
            log(f"🔄 Found existing embeddings at: {out_path}")
            log(f"   Existing embeddings shape: {emb.shape}, dtype: {emb.dtype}")
            return
        except Exception as e:
            log(f"⚠️ Failed to load existing embeddings at {out_path}, will recompute. Error: {e}")

    # Device + model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[env] Using device: {device}")
    if device == "cuda":
        try:
            log(f"[env] CUDA device: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    model_name = "all-mpnet-base-v2"
    log(f"[model] Loading encoder: {model_name}")
    encoder = SentenceTransformer(model_name, device=device)

    # Load passages
    log(f"[data] Reading corpus from: {corpus_path}")
    passage_texts = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading corpus", mininterval=5.0):
            obj = json.loads(line)
            text = (obj.get("title", "") + " " + obj.get("text", "")).strip()
            passage_texts.append(text)

    n = len(passage_texts)
    log(f"[data] Total passages: {n}")

    # Encode in batches
    all_chunks = []
    log(f"[encode] Encoding passages with batch_size={batch_size} ...")
    for i in range(0, n, batch_size):
        batch = passage_texts[i:i + batch_size]

        emb = encoder.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        all_chunks.append(emb)

        # Verbose progress every 100 batches (or at end)
        batch_idx = i // batch_size
        if (batch_idx % 100) == 0 or (i + batch_size) >= n:
            done = min(i + batch_size, n)
            pct = (done / n) * 100.0
            log(f"[encode] {done}/{n} ({pct:.2f}%) passages encoded")

    embeddings = np.vstack(all_chunks).astype("float32")
    log(f"[encode] Final embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    # Save
    np.save(out_path, embeddings)
    log(f"✅ Saved embeddings to: {out_path}")

    # Quick verify
    try:
        test = np.load(out_path, mmap_mode="r")
        log(f"🔍 Reload check OK: {test.shape}, dtype: {test.dtype}")
    except Exception as e:
        log(f"❌ Reload check FAILED for {out_path}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate MPNet passage embeddings on PVC."
    )
    parser.add_argument("--data_root", default="/mnt/work/datasets")
    parser.add_argument("--dataset", required=True,
                        help="E.g. hotpotqa_mpnet, beir_nq_mpnet")
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    data_dir = os.path.join(args.data_root, args.dataset)
    os.makedirs(data_dir, exist_ok=True)

    log_path = os.path.join(data_dir, f"gen_{args.dataset}_mpnet.log")
    log = make_logger(log_path)

    log(f"=== MPNet embedding generation for {args.dataset} ===")
    log(f"[paths] Data dir : {data_dir}")
    log(f"[paths] Log file : {log_path}")

    build_passage_embeddings_mpnet(data_dir, args.dataset, args.batch_size, log)

    log("🎉 Done.")
    log("=========================================")


if __name__ == "__main__":
    main()

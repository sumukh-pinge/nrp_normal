#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def make_logger(log_path: str):
    """
    Simple tee logger: prints to stdout and appends to a log file.
    """
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
    Choose output filename consistent with your NRP code expectations.
    """
    if dataset in ("hotpotqa", "hotpotqa_mpnet"):
        name = "sample_passage_embeddings_hotpotqa.npy"
    elif dataset in ("beir_nq", "beir_nq_mpnet"):
        name = "sample_passage_embeddings_nq.npy"
    else:
        # Generic fallback for any future dataset
        name = f"sample_passage_embeddings_{dataset}.npy"

    out_path = os.path.join(data_dir, name)
    log(f"[config] Using output file: {out_path}")
    return out_path


def count_lines(path: str) -> int:
    """
    Count number of lines in a file. Used to size memmap.
    """
    c = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            c += 1
    return c


def build_passage_embeddings_mpnet(data_dir: str, dataset: str, batch_size: int, log):
    corpus_path = os.path.join(data_dir, "corpus.jsonl")
    out_path = resolve_out_path(data_dir, dataset, log)

    # Sanity: corpus exists
    if not os.path.exists(corpus_path):
        log(f"❌ corpus.jsonl not found at: {corpus_path}")
        sys.exit(1)

    # If embeddings already exist and are loadable, skip
    if os.path.exists(out_path):
        try:
            emb = np.load(out_path, mmap_mode="r")
            log(f"🔄 Found existing embeddings at: {out_path}")
            log(f"   Existing embeddings shape: {emb.shape}, dtype: {emb.dtype}")
            return
        except Exception as e:
            log(f"⚠️ Existing embeddings at {out_path} are unreadable, will recompute. Error: {e}")

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

    # Count total passages first (cheap; needed for memmap shape)
    log(f"[data] Counting lines in corpus: {corpus_path}")
    total_passages = count_lines(corpus_path)
    if total_passages == 0:
        log("❌ No passages found in corpus.jsonl")
        sys.exit(1)
    log(f"[data] Total passages: {total_passages}")

    # Now encode in a streaming fashion and write directly into memmap
    log(f"[encode] Starting encoding with batch_size={batch_size}")

    # Reopen corpus for actual reading
    f = open(corpus_path, "r", encoding="utf-8")

    # ---- First batch: determine embedding dim + init memmap ----
    first_lines = []
    for _ in range(batch_size):
        line = f.readline()
        if not line:
            break
        first_lines.append(line)

    if not first_lines:
        log("❌ Failed to read even a single passage from corpus.")
        f.close()
        sys.exit(1)

    first_texts = []
    for line in first_lines:
        obj = json.loads(line)
        text = (obj.get("title", "") + " " + obj.get("text", "")).strip()
        first_texts.append(text)

    first_emb = encoder.encode(
        first_texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    if first_emb.ndim != 2:
        log(f"❌ Unexpected embedding shape for first batch: {first_emb.shape}")
        f.close()
        sys.exit(1)

    d = first_emb.shape[1]
    log(f"[encode] Embedding dimension: {d}")

    # Create memmap backing file: (total_passages, d)
    mm = np.memmap(out_path, dtype="float32", mode="w+", shape=(total_passages, d))

    # Write first batch
    n_first = first_emb.shape[0]
    mm[0:n_first, :] = first_emb.astype("float32")
    mm.flush()
    offset = n_first
    log(f"[encode] batch=0 done={offset}/{total_passages} ({offset/total_passages*100:.4f}%)")

    # ---- Remaining batches ----
    batch_idx = 1
    while True:
        lines = []
        for _ in range(batch_size):
            line = f.readline()
            if not line:
                break
            lines.append(line)

        if not lines:
            break  # EOF

        texts = []
        for line in lines:
            obj = json.loads(line)
            text = (obj.get("title", "") + " " + obj.get("text", "")).strip()
            texts.append(text)

        emb = encoder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        if emb.ndim != 2:
            log(f"❌ Unexpected embedding shape at batch {batch_idx}: {emb.shape}")
            f.close()
            del mm
            sys.exit(1)

        bsz = emb.shape[0]
        end = offset + bsz
        if end > total_passages:
            log(f"⚠️ Computed end index {end} > total_passages {total_passages}, clipping.")
            end = total_passages
            bsz = end - offset
            emb = emb[:bsz]

        mm[offset:end, :] = emb.astype("float32")
        mm.flush()
        offset = end

        pct = (offset / total_passages) * 100.0
        log(f"[encode] batch={batch_idx} done={offset}/{total_passages} ({pct:.4f}%)")

        batch_idx += 1

        if offset >= total_passages:
            break

    f.close()
    del mm  # ensure data is flushed

    if offset != total_passages:
        log(f"⚠️ Finished with offset={offset}, expected={total_passages}. Some passages may be missing.")
    else:
        log("[encode] All passages encoded successfully.")

    # Reload check
    try:
        test = np.load(out_path, mmap_mode="r")
        log(f"🔍 Reload check OK: {test.shape}, dtype: {test.dtype}")
    except Exception as e:
        log(f"❌ Reload check FAILED for {out_path}: {e}")
        sys.exit(1)

    log(f"✅ Saved embeddings to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate MPNet passage embeddings on PVC."
    )
    parser.add_argument(
        "--data_root",
        default="/mnt/work/datasets",
        help="Root directory where datasets live (inside PVC).",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, e.g. hotpotqa_mpnet, beir_nq_mpnet.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for encoding.",
    )
    args = parser.parse_args()

    data_dir = os.path.join(args.data_root, args.dataset)
    os.makedirs(data_dir, exist_ok=True)

    log_path = os.path.join(
        data_dir,
        f"gen_{args.dataset}_mpnet.log",
    )
    log = make_logger(log_path)

    log(f"=== MPNet embedding generation for {args.dataset} ===")
    log(f"[paths] Data dir : {data_dir}")
    log(f"[paths] Log file : {log_path}")

    build_passage_embeddings_mpnet(data_dir, args.dataset, args.batch_size, log)

    log("🎉 Done.")
    log("=========================================")


if __name__ == "__main__":
    main()

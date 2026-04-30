#!/usr/bin/env python3
"""
PaliBench Reference Embedding Script

Embeds all human reference translations (Sujato, Thanissaro, Bodhi) using
an embedding model via OpenRouter. Results are cached for reuse across
multiple AI evaluations.

Usage:
    python benchmark_embed_references.py
    python benchmark_embed_references.py --model "openai/text-embedding-3-small"

Output:
    benchmark/reference_embeddings.npz - Numpy archive with embeddings
    benchmark/reference_embeddings_meta.json - Metadata (model, dimensions, etc.)
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

try:
    import numpy as np
except ImportError:
    print("Error: numpy not installed. Run: pip install numpy")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Error: requests not installed. Run: pip install requests")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/embeddings"
DEFAULT_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
BATCH_SIZE = 50  # Texts per API call
DEFAULT_DELAY = 0.3  # Seconds between batches
MAX_RETRIES = 3


# =============================================================================
# API Interaction
# =============================================================================

def get_embeddings(
    api_key: str,
    texts: list[str],
    model: str
) -> list[list[float]]:
    """
    Get embeddings for a batch of texts.

    Returns list of embedding vectors.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/palibench",
        "X-Title": "PaliBench Reference Embeddings"
    }

    payload = {
        "model": model,
        "input": texts
    }

    response = requests.post(
        OPENROUTER_API_URL,
        headers=headers,
        json=payload,
        timeout=120
    )

    response.raise_for_status()
    result = response.json()

    # Extract embeddings in order
    embeddings = [None] * len(texts)
    for item in result.get("data", []):
        idx = item["index"]
        embeddings[idx] = item["embedding"]

    return embeddings


def embed_with_retry(
    api_key: str,
    texts: list[str],
    model: str,
    max_retries: int = MAX_RETRIES
) -> list[list[float]]:
    """Get embeddings with retry logic."""
    last_error = None

    for attempt in range(max_retries):
        try:
            return get_embeddings(api_key, texts, model)
        except requests.exceptions.HTTPError as e:
            last_error = e
            if e.response.status_code == 429:
                wait_time = 2 ** (attempt + 2)
                print(f"    Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            elif e.response.status_code >= 500:
                wait_time = 2 ** attempt
                print(f"    Server error. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
        except requests.exceptions.Timeout:
            last_error = "Timeout"
            wait_time = 2 ** attempt
            print(f"    Timeout. Waiting {wait_time}s...")
            time.sleep(wait_time)

    raise Exception(f"Failed after {max_retries} retries: {last_error}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Embed human reference translations for PaliBench'
    )
    parser.add_argument('--model', type=str, default=DEFAULT_EMBEDDING_MODEL,
                        help=f'Embedding model (default: {DEFAULT_EMBEDDING_MODEL})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Texts per API call (default: {BATCH_SIZE})')
    parser.add_argument('--delay', type=float, default=DEFAULT_DELAY,
                        help=f'Delay between batches (default: {DEFAULT_DELAY})')
    args = parser.parse_args()

    # Load environment
    load_dotenv()
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in environment or .env file")
        sys.exit(1)

    # Paths
    repo_root = Path(__file__).resolve().parent.parent
    test_set_path = repo_root / "data" / "test_set.json"
    output_path = repo_root / "data" / "reference_embeddings.npz"
    meta_path = repo_root / "data" / "reference_embeddings_meta.json"

    # Load test set
    print(f"Loading test set from {test_set_path}...")
    with open(test_set_path, 'r', encoding='utf-8') as f:
        test_set = json.load(f)
    print(f"  Loaded {len(test_set)} passages")

    # Prepare texts for embedding
    print("\nPreparing texts...")
    passage_ids = sorted(test_set.keys())
    translators = ['sujato', 'thanissaro', 'bodhi']

    # We'll embed in order: all sujato, then all thanissaro, then all bodhi
    all_texts = []
    text_map = []  # (passage_id, translator) for each text

    for translator in translators:
        for pid in passage_ids:
            text = test_set[pid].get(translator, "")
            all_texts.append(text)
            text_map.append((pid, translator))

    total_texts = len(all_texts)
    print(f"  Total texts to embed: {total_texts} ({len(passage_ids)} passages × {len(translators)} translators)")

    # Embed in batches
    print(f"\nEmbedding with {args.model}...")
    print(f"  Batch size: {args.batch_size}")

    all_embeddings = []
    num_batches = (total_texts + args.batch_size - 1) // args.batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, total_texts)
        batch_texts = all_texts[start:end]

        print(f"  Batch {batch_idx + 1}/{num_batches} ({end}/{total_texts} texts)...")

        embeddings = embed_with_retry(api_key, batch_texts, args.model)
        all_embeddings.extend(embeddings)

        if batch_idx < num_batches - 1 and args.delay > 0:
            time.sleep(args.delay)

    # L2 normalize all embeddings
    print("\nNormalizing embeddings...")
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    embeddings_array = embeddings_array / (norms + 1e-10)

    embedding_dim = embeddings_array.shape[1]
    print(f"  Embedding dimension: {embedding_dim}")

    # Reshape into (num_passages, num_translators, embedding_dim)
    num_passages = len(passage_ids)
    num_translators = len(translators)
    embeddings_reshaped = embeddings_array.reshape(num_translators, num_passages, embedding_dim)
    # Transpose to (num_passages, num_translators, embedding_dim)
    embeddings_reshaped = embeddings_reshaped.transpose(1, 0, 2)

    # Save embeddings
    print(f"\nSaving embeddings to {output_path}...")
    np.savez_compressed(
        output_path,
        embeddings=embeddings_reshaped,
        passage_ids=np.array(passage_ids, dtype=object),
        translators=np.array(translators, dtype=object)
    )

    # Save metadata
    metadata = {
        "model": args.model,
        "embedding_dim": embedding_dim,
        "num_passages": num_passages,
        "translators": translators,
        "normalized": True,
        "created_at": datetime.now().isoformat(),
        "passage_ids": passage_ids
    }

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saving metadata to {meta_path}...")

    print(f"\n{'='*60}")
    print("EMBEDDING COMPLETE")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Passages: {num_passages}")
    print(f"Translators: {', '.join(translators)}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

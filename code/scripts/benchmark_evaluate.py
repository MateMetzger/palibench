#!/usr/bin/env python3
"""
PaliBench Evaluation Script

Evaluates AI translations against human references using semantic similarity.

Usage:
    python benchmark_evaluate.py --input results/translations/model_reasoning-none.json
    python benchmark_evaluate.py --input results/translations/*.json  # Evaluate all

Requires:
    - Reference embeddings (run benchmark_embed_references.py first)
    - AI translation file(s) from benchmark_translate.py

Output:
    results/evaluations/{model_name}.json - Detailed evaluation results
"""

import json
import os
import sys
import time
import glob
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

try:
    import sacrebleu
    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False
    print("Warning: sacrebleu not installed. chrF++ and BLEU metrics will be skipped.")
    print("  Install with: pip install sacrebleu")

try:
    from comet import download_model, load_from_checkpoint
    HAS_COMET = True
except ImportError:
    HAS_COMET = False


# =============================================================================
# Configuration
# =============================================================================

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/embeddings"
BATCH_SIZE = 50
DEFAULT_DELAY = 0.3
MAX_RETRIES = 3
DRIFT_EPSILON = 0.01  # Floor for normalized drift calculation


# =============================================================================
# API Interaction
# =============================================================================

def get_embeddings(api_key: str, texts: list[str], model: str) -> list[list[float]]:
    """Get embeddings for a batch of texts."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/palibench",
        "X-Title": "PaliBench Evaluation"
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

    embeddings = [None] * len(texts)
    for item in result.get("data", []):
        idx = item["index"]
        embeddings[idx] = item["embedding"]

    return embeddings


def embed_with_retry(api_key: str, texts: list[str], model: str, max_retries: int = MAX_RETRIES) -> list[list[float]]:
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
# Evaluation Logic
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors (assumes L2 normalized)."""
    return float(np.dot(a, b))


def compute_passage_metrics(
    ai_embedding: np.ndarray,
    ref_embeddings: np.ndarray,  # Shape: (3, dim) for sujato, thanissaro, bodhi
    translator_names: list[str]
) -> dict:
    """
    Compute all metrics for a single passage.

    Args:
        ai_embedding: L2-normalized AI translation embedding
        ref_embeddings: L2-normalized reference embeddings (3, dim)
        translator_names: ['sujato', 'thanissaro', 'bodhi']

    Returns:
        Dict with all metrics
    """
    # Per-translator similarities
    similarities = {}
    for i, name in enumerate(translator_names):
        similarities[f"sim_{name}"] = cosine_similarity(ai_embedding, ref_embeddings[i])

    # Best match
    sim_values = list(similarities.values())
    sim_best = max(sim_values)
    closest_idx = sim_values.index(sim_best)
    closest_translator = translator_names[closest_idx]

    # Centroid (average of reference embeddings, then normalize)
    centroid = ref_embeddings.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-10)

    # Similarity to centroid
    sim_centroid = cosine_similarity(ai_embedding, centroid)

    # Human variance: how much do humans differ from centroid?
    human_dists = []
    for i in range(len(translator_names)):
        dist = 1.0 - cosine_similarity(ref_embeddings[i], centroid)
        human_dists.append(dist)
    human_variance = np.mean(human_dists)

    # AI drift from centroid
    ai_drift = 1.0 - sim_centroid

    # Normalized drift
    normalized_drift = ai_drift / (human_variance + DRIFT_EPSILON)

    return {
        **similarities,
        "sim_best": sim_best,
        "sim_centroid": sim_centroid,
        "closest_translator": closest_translator,
        "human_variance": human_variance,
        "ai_drift": ai_drift,
        "normalized_drift": normalized_drift
    }


def compute_text_metrics(
    ai_text: str,
    ref_texts: list[str],
) -> dict:
    """
    Compute text-based metrics for a single passage.

    Returns dict with chrf, bleu, and length_ratio.
    """
    metrics = {}

    if HAS_SACREBLEU:
        chrf_score = sacrebleu.sentence_chrf(ai_text, ref_texts, word_order=2)
        metrics["chrf"] = chrf_score.score

        bleu_score = sacrebleu.sentence_bleu(ai_text, ref_texts)
        metrics["bleu"] = bleu_score.score

    # Length ratio: AI length / mean of reference lengths
    ref_lengths = [len(r) for r in ref_texts if r]
    mean_ref_len = sum(ref_lengths) / len(ref_lengths) if ref_lengths else 0
    if mean_ref_len > 0:
        metrics["length_ratio"] = len(ai_text) / mean_ref_len
    else:
        metrics["length_ratio"] = 0.0

    return metrics


def compute_comet_scores(
    ai_translations: dict,
    test_set: dict,
    common_passages: list[str],
    ref_translators: list[str],
) -> dict:
    """
    Compute COMET scores for all passages in batch.

    For each of the 3 reference translators, runs COMET with
    source=Pali, hypothesis=AI, reference=human. Returns per-passage
    comet_avg (mean across refs) and comet_best (max across refs).
    """
    import torch

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

    print("    Loading COMET model (Unbabel/wmt22-comet-da)...")
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)

    gpus = 1 if torch.cuda.is_available() else 0
    if gpus:
        print("    Using GPU for COMET inference")
    else:
        print("    Using CPU for COMET inference (this may take a while)")

    # Collect per-reference scores
    per_ref_scores = {pid: [] for pid in common_passages}

    for translator in ref_translators:
        print(f"    Computing COMET vs {translator}...")
        data = []
        for pid in common_passages:
            data.append({
                "src": test_set[pid]["pali"],
                "mt": ai_translations[pid],
                "ref": test_set[pid][translator],
            })

        output = model.predict(data, batch_size=64, gpus=gpus)

        for i, pid in enumerate(common_passages):
            per_ref_scores[pid].append(output.scores[i])

    # Aggregate per-passage
    results = {}
    for pid in common_passages:
        scores = per_ref_scores[pid]
        results[pid] = {
            "comet_avg": sum(scores) / len(scores),
            "comet_best": max(scores),
        }

    return results


def aggregate_metrics(per_passage: dict) -> dict:
    """Compute aggregate statistics from per-passage metrics."""
    if not per_passage:
        return {}

    # Collect all metric values
    metric_names = [
        "sim_sujato", "sim_thanissaro", "sim_bodhi",
        "sim_best", "sim_centroid", "human_variance", "normalized_drift",
        "chrf", "bleu", "length_ratio", "comet_avg", "comet_best",
    ]

    aggregates = {}
    for metric in metric_names:
        values = [p[metric] for p in per_passage.values() if metric in p]
        if values:
            aggregates[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values))
            }

    # Closest translator distribution
    closest_counts = {}
    for p in per_passage.values():
        t = p.get("closest_translator")
        if t:
            closest_counts[t] = closest_counts.get(t, 0) + 1
    aggregates["closest_translator_distribution"] = closest_counts

    # Outliers (normalized_drift > 2.0)
    outliers = [
        pid for pid, metrics in per_passage.items()
        if metrics.get("normalized_drift", 0) > 2.0
    ]
    aggregates["outlier_count"] = len(outliers)
    aggregates["outlier_percentage"] = 100.0 * len(outliers) / len(per_passage) if per_passage else 0

    return aggregates


# =============================================================================
# Main
# =============================================================================

def evaluate_single_file(
    input_path: Path,
    ref_embeddings: np.ndarray,
    ref_passage_ids: list[str],
    ref_translators: list[str],
    embedding_model: str,
    api_key: str,
    output_dir: Path,
    delay: float,
    test_set: dict = None,
    skip_text_metrics: bool = False,
    skip_comet: bool = False,
) -> dict:
    """Evaluate a single AI translation file."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {input_path.name}")
    print(f"{'='*60}")

    # Load AI translations
    with open(input_path, 'r', encoding='utf-8') as f:
        ai_data = json.load(f)

    ai_translations = ai_data.get("translations", {})
    model_name = ai_data.get("metadata", {}).get("model", input_path.stem)

    print(f"  Model: {model_name}")
    print(f"  Translations: {len(ai_translations)}")

    # Find common passages (in case AI is missing some)
    ref_passage_set = set(ref_passage_ids)
    common_passages = [pid for pid in ref_passage_ids if pid in ai_translations]
    print(f"  Common passages: {len(common_passages)}")

    if not common_passages:
        print("  ERROR: No common passages found!")
        return None

    # Create passage_id to index mapping for references
    pid_to_idx = {pid: i for i, pid in enumerate(ref_passage_ids)}

    # Prepare AI texts for embedding
    ai_texts = [ai_translations[pid] for pid in common_passages]

    # Embed AI translations
    print(f"\n  Embedding AI translations...")
    ai_embeddings = []
    num_batches = (len(ai_texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(ai_texts))
        batch_texts = ai_texts[start:end]

        if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
            print(f"    Batch {batch_idx + 1}/{num_batches}...")

        embeddings = embed_with_retry(api_key, batch_texts, embedding_model)
        ai_embeddings.extend(embeddings)

        if batch_idx < num_batches - 1 and delay > 0:
            time.sleep(delay)

    # Normalize AI embeddings
    ai_embeddings = np.array(ai_embeddings, dtype=np.float32)
    norms = np.linalg.norm(ai_embeddings, axis=1, keepdims=True)
    ai_embeddings = ai_embeddings / (norms + 1e-10)

    # Compute metrics for each passage
    print(f"\n  Computing metrics...")
    per_passage_metrics = {}

    for i, pid in enumerate(common_passages):
        ref_idx = pid_to_idx[pid]
        passage_refs = ref_embeddings[ref_idx]  # Shape: (3, dim)

        metrics = compute_passage_metrics(
            ai_embeddings[i],
            passage_refs,
            ref_translators
        )
        per_passage_metrics[pid] = metrics

    # Text-based metrics (chrF++, BLEU, length ratio)
    if test_set and not skip_text_metrics:
        print(f"\n  Computing text-based metrics (chrF++, BLEU, length ratio)...")
        for pid in common_passages:
            ref_texts = [test_set[pid].get(t, "") for t in ref_translators]
            text_metrics = compute_text_metrics(
                ai_text=ai_translations[pid],
                ref_texts=ref_texts,
            )
            per_passage_metrics[pid].update(text_metrics)

    # COMET
    if test_set and not skip_comet and HAS_COMET:
        print(f"\n  Computing COMET scores (this may take a few minutes)...")
        comet_scores = compute_comet_scores(
            ai_translations=ai_translations,
            test_set=test_set,
            common_passages=common_passages,
            ref_translators=ref_translators,
        )
        for pid in common_passages:
            per_passage_metrics[pid].update(comet_scores[pid])

    # Aggregate metrics
    aggregate = aggregate_metrics(per_passage_metrics)

    # Corpus-level BLEU and chrF++
    if test_set and not skip_text_metrics and HAS_SACREBLEU:
        hypotheses = [ai_translations[pid] for pid in common_passages]
        refs_by_translator = [
            [test_set[pid].get(t, "") for pid in common_passages]
            for t in ref_translators
        ]
        corpus_bleu = sacrebleu.corpus_bleu(hypotheses, refs_by_translator)
        corpus_chrf = sacrebleu.corpus_chrf(hypotheses, refs_by_translator, word_order=2)
        aggregate["corpus_bleu"] = corpus_bleu.score
        aggregate["corpus_chrf"] = corpus_chrf.score

    # Build output
    output = {
        "model": model_name,
        "source_file": str(input_path),
        "evaluated_at": datetime.now().isoformat(),
        "embedding_model": embedding_model,
        "num_passages_evaluated": len(common_passages),
        "num_passages_in_ai": len(ai_translations),
        "num_passages_in_reference": len(ref_passage_ids),
        "aggregate_scores": aggregate,
        "per_passage_scores": per_passage_metrics
    }

    # Extract outlier passage IDs
    outliers = [
        pid for pid, metrics in per_passage_metrics.items()
        if metrics.get("normalized_drift", 0) > 2.0
    ]
    output["outlier_passages"] = sorted(outliers)[:50]  # Limit to 50

    # Save evaluation
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = input_path.stem  # Already includes model name and reasoning
    output_path = output_dir / f"{safe_name}_eval.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {output_path}")

    # Print summary
    print(f"\n  {'-'*40}")
    print(f"  SUMMARY")
    print(f"  {'-'*40}")
    agg = aggregate
    print(f"  sim_best:        {agg.get('sim_best', {}).get('mean', 0):.3f} (+-{agg.get('sim_best', {}).get('std', 0):.3f})")
    print(f"  sim_centroid:    {agg.get('sim_centroid', {}).get('mean', 0):.3f} (+-{agg.get('sim_centroid', {}).get('std', 0):.3f})")
    print(f"  sim_sujato:      {agg.get('sim_sujato', {}).get('mean', 0):.3f}")
    print(f"  sim_thanissaro:  {agg.get('sim_thanissaro', {}).get('mean', 0):.3f}")
    print(f"  sim_bodhi:       {agg.get('sim_bodhi', {}).get('mean', 0):.3f}")
    print(f"  normalized_drift:{agg.get('normalized_drift', {}).get('mean', 0):.2f} (+-{agg.get('normalized_drift', {}).get('std', 0):.2f})")
    print(f"  outliers (>2x):  {agg.get('outlier_count', 0)} ({agg.get('outlier_percentage', 0):.1f}%)")

    if "chrf" in agg:
        print(f"  chrF++:          {agg['chrf']['mean']:.1f} (+-{agg['chrf']['std']:.1f})")
    if "bleu" in agg:
        print(f"  BLEU:            {agg['bleu']['mean']:.1f} (+-{agg['bleu']['std']:.1f})")
    if "length_ratio" in agg:
        print(f"  length_ratio:    {agg['length_ratio']['mean']:.3f} (+-{agg['length_ratio']['std']:.3f})")
    if "comet_avg" in agg:
        print(f"  COMET (avg):     {agg['comet_avg']['mean']:.4f} (+-{agg['comet_avg']['std']:.4f})")
    if "corpus_bleu" in agg:
        print(f"  corpus BLEU:     {agg['corpus_bleu']:.1f}")
    if "corpus_chrf" in agg:
        print(f"  corpus chrF++:   {agg['corpus_chrf']:.1f}")

    dist = agg.get('closest_translator_distribution', {})
    if dist:
        print(f"\n  Closest translator distribution:")
        for t, count in sorted(dist.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(common_passages)
            print(f"    {t}: {count} ({pct:.1f}%)")

    return output


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate AI translations against human references'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='AI translation file(s) - supports glob patterns')
    parser.add_argument('--embedding-model', type=str, default=None,
                        help='Embedding model (default: same as references)')
    parser.add_argument('--delay', type=float, default=DEFAULT_DELAY,
                        help=f'Delay between batches (default: {DEFAULT_DELAY})')
    parser.add_argument('--skip-text-metrics', action='store_true',
                        help='Skip chrF++, BLEU, and length ratio metrics')
    parser.add_argument('--skip-comet', action='store_true',
                        help='Skip COMET metric (requires large model download)')
    args = parser.parse_args()

    # Load environment
    load_dotenv()
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in environment or .env file")
        sys.exit(1)

    # Paths
    repo_root = Path(__file__).resolve().parent.parent
    ref_embeddings_path = repo_root / "data" / "reference_embeddings.npz"
    ref_meta_path = repo_root / "data" / "reference_embeddings_meta.json"
    output_dir = repo_root / "results" / "evaluations"

    # Check for reference embeddings
    if not ref_embeddings_path.exists():
        print(f"Error: Reference embeddings not found at {ref_embeddings_path}")
        print("Run benchmark_embed_references.py first.")
        sys.exit(1)

    # Load reference embeddings
    print("Loading reference embeddings...")
    ref_data = np.load(ref_embeddings_path, allow_pickle=True)
    ref_embeddings = ref_data['embeddings']  # Shape: (num_passages, 3, dim)
    ref_passage_ids = list(ref_data['passage_ids'])
    ref_translators = list(ref_data['translators'])

    with open(ref_meta_path, 'r', encoding='utf-8') as f:
        ref_meta = json.load(f)

    embedding_model = args.embedding_model or ref_meta['model']
    print(f"  Passages: {len(ref_passage_ids)}")
    print(f"  Embedding model: {embedding_model}")
    print(f"  Embedding dim: {ref_embeddings.shape[2]}")

    # Load test set for text-based metrics
    test_set = None
    if not args.skip_text_metrics or not args.skip_comet:
        test_set_path = repo_root / "data" / "test_set.json"
        if test_set_path.exists():
            print("Loading test set for text-based metrics...")
            with open(test_set_path, 'r', encoding='utf-8') as f:
                test_set = json.load(f)
            print(f"  Test set passages: {len(test_set)}")
        else:
            print(f"Warning: Test set not found at {test_set_path}")
            print("  Text-based metrics (chrF++, BLEU, COMET) will be skipped.")

    # Find input files
    input_pattern = args.input
    if '*' in input_pattern:
        input_files = sorted(glob.glob(input_pattern))
    else:
        input_files = [input_pattern]

    if not input_files:
        print(f"Error: No files found matching '{input_pattern}'")
        sys.exit(1)

    print(f"\nFiles to evaluate: {len(input_files)}")
    for f in input_files:
        print(f"  - {f}")

    # Evaluate each file
    results = []
    for input_file in input_files:
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"\nWarning: File not found: {input_path}")
            continue

        result = evaluate_single_file(
            input_path=input_path,
            ref_embeddings=ref_embeddings,
            ref_passage_ids=ref_passage_ids,
            ref_translators=ref_translators,
            embedding_model=embedding_model,
            api_key=api_key,
            output_dir=output_dir,
            delay=args.delay,
            test_set=test_set,
            skip_text_metrics=args.skip_text_metrics,
            skip_comet=args.skip_comet,
        )

        if result:
            results.append(result)

    # Final summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        has_chrf = any('chrf' in r['aggregate_scores'] for r in results)
        has_comet = any('comet_avg' in r['aggregate_scores'] for r in results)

        header = f"{'Model':<40} {'sim_best':>10} {'chrF++':>8}" if has_chrf else f"{'Model':<40} {'sim_best':>10}"
        if has_comet:
            header += f" {'COMET':>8}"
        print(header)
        print("-" * len(header))
        for r in sorted(results, key=lambda x: -x['aggregate_scores'].get('sim_best', {}).get('mean', 0)):
            name = r['model'][:38]
            agg = r['aggregate_scores']
            sim_best = agg.get('sim_best', {}).get('mean', 0)
            line = f"{name:<40} {sim_best:>10.3f}"
            if has_chrf:
                chrf_val = agg.get('chrf', {}).get('mean', 0)
                line += f" {chrf_val:>8.1f}"
            if has_comet:
                comet_val = agg.get('comet_avg', {}).get('mean', 0)
                line += f" {comet_val:>8.4f}"
            print(line)


if __name__ == "__main__":
    main()

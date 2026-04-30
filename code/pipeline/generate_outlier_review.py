#!/usr/bin/env python3
"""
Generate Outlier Review Files

Creates a review file for each model containing all outlier passages with:
- The Pali source
- Human reference translations (Sujato, Thanissaro, Bodhi)
- The LLM's translation
- Similarity scores

Usage:
    python generate_outlier_review.py
    python generate_outlier_review.py --output-dir results/outlier_reviews
    python generate_outlier_review.py --format json
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description='Generate outlier review files for each model'
    )
    parser.add_argument('--output-dir', type=str, default='results/outlier_reviews',
                        help='Output directory for review files')
    parser.add_argument('--format', type=str, choices=['txt', 'json'], default='txt',
                        help='Output format (txt or json)')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    test_set_path = repo_root / "data" / "test_set.json"
    results_dir = repo_root / "results" / "translations"
    evals_dir = repo_root / "results" / "evaluations"
    output_dir = repo_root / args.output_dir

    # Load test set
    print(f"Loading test set...")
    with open(test_set_path, 'r', encoding='utf-8') as f:
        test_set = json.load(f)

    # Find all evaluation files
    eval_files = list(evals_dir.glob('*_eval.json'))
    print(f"Found {len(eval_files)} evaluation files")

    output_dir.mkdir(parents=True, exist_ok=True)

    for eval_path in sorted(eval_files):
        print(f"\nProcessing: {eval_path.name}")

        # Load evaluation
        with open(eval_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)

        model_name = eval_data.get('model', eval_path.stem)
        outlier_passages = eval_data.get('outlier_passages', [])
        per_passage_scores = eval_data.get('per_passage_scores', {})

        # Find corresponding results file
        results_path = Path(eval_data.get('source_file', ''))
        if not results_path.is_absolute():
            results_path = repo_root / results_path

        if not results_path.exists():
            # Try to find it by pattern
            pattern = eval_path.stem.replace('_eval', '')
            matches = list(results_dir.glob(f'{pattern}.json'))
            if matches:
                results_path = matches[0]
            else:
                print(f"  Warning: Could not find results file for {model_name}")
                continue

        with open(results_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)

        translations = results_data.get('translations', {})

        # Get ALL outliers from per_passage_scores (not just the limited list)
        all_outliers = [
            pid for pid, scores in per_passage_scores.items()
            if scores.get('normalized_drift', 0) > 2.0
        ]
        all_outliers.sort()

        print(f"  Model: {model_name}")
        print(f"  Outliers: {len(all_outliers)}")

        if args.format == 'json':
            output_data = generate_json_output(
                model_name, all_outliers, test_set, translations, per_passage_scores
            )
            output_path = output_dir / f"{eval_path.stem.replace('_eval', '')}_outliers.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        else:
            output_text = generate_text_output(
                model_name, all_outliers, test_set, translations, per_passage_scores
            )
            output_path = output_dir / f"{eval_path.stem.replace('_eval', '')}_outliers.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_text)

        print(f"  Saved: {output_path}")

    print(f"\nDone! Review files saved to: {output_dir}")


def generate_text_output(model_name, outliers, test_set, translations, scores):
    """Generate human-readable text output."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"OUTLIER REVIEW: {model_name}")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Total outliers: {len(outliers)}")
    lines.append("=" * 80)
    lines.append("")

    for i, pid in enumerate(outliers, 1):
        ref = test_set.get(pid, {})
        llm_trans = translations.get(pid, "[MISSING]")
        passage_scores = scores.get(pid, {})

        lines.append("-" * 80)
        lines.append(f"[{i}/{len(outliers)}] {pid}")
        lines.append("-" * 80)
        lines.append("")

        # Scores
        lines.append("SCORES:")
        lines.append(f"  sim_centroid:     {passage_scores.get('sim_centroid', 0):.3f}")
        lines.append(f"  sim_best:         {passage_scores.get('sim_best', 0):.3f}")
        lines.append(f"  normalized_drift: {passage_scores.get('normalized_drift', 0):.2f}")
        lines.append(f"  closest_to:       {passage_scores.get('closest_translator', 'N/A')}")
        lines.append(f"  sim_sujato:       {passage_scores.get('sim_sujato', 0):.3f}")
        lines.append(f"  sim_thanissaro:   {passage_scores.get('sim_thanissaro', 0):.3f}")
        lines.append(f"  sim_bodhi:        {passage_scores.get('sim_bodhi', 0):.3f}")
        lines.append("")

        # Pali source
        lines.append("PALI:")
        lines.append(ref.get('pali', '[N/A]'))
        lines.append("")

        # Human translations
        lines.append("SUJATO:")
        lines.append(ref.get('sujato', '[N/A]'))
        lines.append("")

        lines.append("THANISSARO:")
        lines.append(ref.get('thanissaro', '[N/A]'))
        lines.append("")

        lines.append("BODHI:")
        lines.append(ref.get('bodhi', '[N/A]'))
        lines.append("")

        # LLM translation
        lines.append("LLM TRANSLATION:")
        lines.append(llm_trans)
        lines.append("")
        lines.append("")

    return "\n".join(lines)


def generate_json_output(model_name, outliers, test_set, translations, scores):
    """Generate JSON output."""
    output = {
        "model": model_name,
        "generated_at": datetime.now().isoformat(),
        "total_outliers": len(outliers),
        "outliers": []
    }

    for pid in outliers:
        ref = test_set.get(pid, {})
        passage_scores = scores.get(pid, {})

        outlier_data = {
            "passage_id": pid,
            "scores": {
                "sim_centroid": passage_scores.get('sim_centroid', 0),
                "sim_best": passage_scores.get('sim_best', 0),
                "normalized_drift": passage_scores.get('normalized_drift', 0),
                "closest_translator": passage_scores.get('closest_translator', ''),
                "sim_sujato": passage_scores.get('sim_sujato', 0),
                "sim_thanissaro": passage_scores.get('sim_thanissaro', 0),
                "sim_bodhi": passage_scores.get('sim_bodhi', 0),
            },
            "pali": ref.get('pali', ''),
            "references": {
                "sujato": ref.get('sujato', ''),
                "thanissaro": ref.get('thanissaro', ''),
                "bodhi": ref.get('bodhi', ''),
            },
            "llm_translation": translations.get(pid, '')
        }
        output["outliers"].append(outlier_data)

    return output


if __name__ == "__main__":
    main()

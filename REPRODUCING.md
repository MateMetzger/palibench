# Reproducing The Reported Results

This package supports two levels of reproducibility.

## Level 1: Inspect Reported Results Without Restricted Text

Use the included derived files:

- `results/aggregate_scores.csv`
- `results/aggregate_scores.json`
- `results/per_passage_scores/*.json`
- `results/translations/*.json`
- `metadata/passage_manifest.csv`

These files allow readers to verify aggregate rankings, per-passage metric distributions, closest-translator patterns, outlier rates, and truncation patterns without access to full human reference translations.

## Level 2: Recompute Metrics From The Full Benchmark

To recompute all metrics, the full local/private benchmark is required, including:

- `data/test_set.json`
- `data/reference_embeddings.npz`, or enough access to regenerate it
- the machine translation files in `results/translations/`

From the full repository:

```bash
pip install -r requirements.txt
python3 scripts/benchmark_embed_references.py
python3 scripts/benchmark_evaluate.py --input "results/translations/*.json"
```

To skip COMET:

```bash
python3 scripts/benchmark_evaluate.py --input "results/translations/*.json" --skip-comet
```

## Rebuild This Public Package

From the full repository:

```bash
python3 reproducibility_package/build_package.py
```

This regenerates the shareable package while excluding full source/reference texts and reference embeddings.


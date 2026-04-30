# PaliBench Reproducibility Package

This folder contains shareable artifacts for the PaliBench paper:

**PaliBench: A Multi-Reference Blueprint for Classical Language Translation Benchmarks**

Planned public repository: <https://github.com/MateMetzger/palibench>

The package is designed for public release alongside the article while avoiding redistribution of copyright-sensitive human reference translations. It supports inspection of the reported results, reuse of the benchmark-construction code, and reconstruction of the full benchmark by researchers who have lawful access to the underlying source translations.

## What Is Included

- `code/`: scripts and pipeline code used for translation, embedding, evaluation, alignment, verification, filtering, and outlier review generation.
- `prompts/`: translation and alignment prompts used in the study.
- `metadata/metadata.json`: filtering configuration and corpus-construction statistics.
- `metadata/reference_embeddings_meta.json`: metadata for the reference embedding run, excluding the embeddings themselves.
- `metadata/passage_manifest.csv`: passage identifiers, collection identifiers, segment IDs, segment counts, and text-length statistics. It does not include full Pali source text or human translations.
- `results/aggregate_scores.csv` and `results/aggregate_scores.json`: aggregate metrics for the ten evaluated models.
- `results/per_passage_scores/`: per-passage metric outputs without source text or human reference text.
- `results/translations/`: model-generated translation outputs used for evaluation.
- `package_manifest.json`: machine-readable list of included files and intentionally excluded materials.
- `build_package.py`: script for rebuilding this folder from the full local research repository.

## What Is Excluded

The following files are intentionally not included in this public package:

- `data/test_set.json`, because it contains full Pali passages and aligned human reference translations.
- `data/reference_embeddings.npz`, because it contains embeddings derived from the full human reference translations.
- `results/outlier_reviews/*.json`, because these files include full Pali passages, human references, and model output snippets for manual inspection.
- `source/*`, because it contains raw and aligned source materials, including copyright-sensitive translations.

See `DATA_AVAILABILITY.md` for the rationale and reconstruction policy.

## Rebuilding The Package

From the full private/local repository:

```bash
python3 palibench_reproducibility_package/build_package.py
```

The script regenerates the sanitized metadata, aggregate results, per-passage scores, copied prompts, copied code, and model outputs.

## Recommended Public Hosting

Use a citable research repository for the archival package and a Git host for active code development.

Recommended setup:

- Archive release: Zenodo, OSF, or institutional repository with a DOI.
- Code mirror: GitLab or GitHub.

GitHub is suitable for the public working repository. For journal submission, the most important requirement is not the Git host itself but a stable, citable URL or DOI. A good workflow is to host this package on GitHub and archive a tagged release on Zenodo or OSF.

## License

The package metadata, code, prompts, and derived metrics may be released under the project license, subject to the rights of third-party source texts and model providers. The excluded source/reference texts remain under their respective rights holders and licenses.

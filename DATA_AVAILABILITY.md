# Data Availability

This reproducibility package is designed to make the PaliBench study inspectable and reusable without redistributing copyright-sensitive reference translations.

## Publicly Shareable Materials

The following materials can be made public:

- Pipeline and evaluation code.
- Translation and alignment prompts.
- Corpus construction metadata and filtering statistics.
- Passage identifiers, segment identifiers, and text-length metadata.
- Aggregate model evaluation results.
- Per-passage metric scores with no source or reference text.
- Model-generated translation outputs.
- Instructions for reconstructing the benchmark from authorized source materials.

## Restricted Or Withheld Materials

The full benchmark dataset used internally contains Pali source text and aligned English translations by Bhikkhu Sujato, Bhikkhu Thanissaro, and Bhikkhu Bodhi. These materials are freely accessible online from the cited providers, but they are not redistributed in this package because their licenses differ and because aligned full-text extraction may raise derivative-work questions. Availability from a cited website or publisher should not be read as permission to redistribute aligned full-text copies.

In particular:

- Sujato translations on SuttaCentral are released under Creative Commons Zero (CC0), subject to SuttaCentral's stated terms and metadata practices.
- Thanissaro translations used here are available through dhammatalks.org / Access to Insight under Creative Commons Attribution-NonCommercial 4.0.
- Bodhi translations used here are published by Wisdom Publications under Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported. The NoDerivs condition is the main reason this public package does not redistribute aligned full-text reference data.

For that reason, `data/test_set.json`, raw source files, aligned human-reference files, reference embeddings, and outlier-review files containing full text are excluded from the public package. The package applies this exclusion uniformly to all human translations rather than treating the three translators differently.

## Reconstruction

Researchers with lawful access to the source translations can reconstruct the full benchmark locally by:

1. Obtaining the relevant Pali source texts and English translations from the cited sources.
2. Running the alignment pipeline in `code/pipeline/align.py` with the prompt in `prompts/alignment_prompt.txt`.
3. Verifying alignments using `code/pipeline/verify_alignment.py`.
4. Applying the filtering and deduplication pipeline in `code/pipeline/filter_corpus_v2.py`.
5. Regenerating reference embeddings with `code/scripts/benchmark_embed_references.py`.
6. Evaluating model outputs with `code/scripts/benchmark_evaluate.py`.

The public `metadata/passage_manifest.csv` provides passage and segment identifiers needed to compare reconstructed data against the reported benchmark composition.

## Citation

If using this package, cite the associated PaliBench article once published. Until publication, cite the repository release and include the release date and DOI if available.

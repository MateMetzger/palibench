#!/usr/bin/env python3
"""Build a copyright-conscious reproducibility package for PaliBench.

The package intentionally excludes full human reference translations and
reference embeddings. It exports code, prompts, metadata, model outputs, and
derived metrics sufficient to inspect the reported results and rerun the
pipeline locally when the underlying source texts are available.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG = Path(__file__).resolve().parent

EXCLUDE_FROM_CLEAN = {
    "build_package.py",
    "README.md",
    "DATA_AVAILABILITY.md",
    "REPRODUCING.md",
    "LICENSE",
    "CITATION.cff",
    ".gitignore",
}


def clean_package() -> None:
    for child in PKG.iterdir():
        if child.name in EXCLUDE_FROM_CLEAN:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def make_passage_manifest() -> None:
    test_set = load_json(ROOT / "data" / "test_set.json")
    rows = []
    for passage_id, item in sorted(test_set.items()):
        collection = passage_id[:2].upper()
        sutta_id = passage_id.split(":", 1)[0]
        rows.append(
            {
                "passage_id": passage_id,
                "collection": collection,
                "sutta_id": sutta_id,
                "segment_count": item.get("segment_count", len(item.get("segments", []))),
                "segments": " ".join(item.get("segments", [])),
                "pali_chars": len(item.get("pali", "")),
                "sujato_chars": len(item.get("sujato", "")),
                "thanissaro_chars": len(item.get("thanissaro", "")),
                "bodhi_chars": len(item.get("bodhi", "")),
            }
        )

    out = PKG / "metadata" / "passage_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_aggregate_results() -> None:
    eval_dir = ROOT / "results" / "evaluations"
    rows = []
    for path in sorted(eval_dir.glob("*_eval.json")):
        data = load_json(path)
        row = {
            "model": data.get("model"),
            "source_file": data.get("source_file"),
            "num_passages_evaluated": data.get("num_passages_evaluated"),
        }
        for key, value in data.get("aggregate_scores", {}).items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    row[f"{key}_{subkey}"] = subvalue
            else:
                row[key] = value
        rows.append(row)

    rows.sort(key=lambda r: r.get("sim_best_mean", 0), reverse=True)
    out_json = PKG / "results" / "aggregate_scores.json"
    write_json(out_json, rows)

    out_csv = PKG / "results" / "aggregate_scores.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    priority = ["model", "source_file", "num_passages_evaluated"]
    fieldnames = priority + [f for f in fieldnames if f not in priority]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_per_passage_scores() -> None:
    out_dir = PKG / "results" / "per_passage_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / "results" / "evaluations").glob("*_eval.json")):
        data = load_json(path)
        rows = []
        for passage_id, scores in sorted(data.get("per_passage_scores", {}).items()):
            rows.append({"passage_id": passage_id, **scores})
        write_json(out_dir / path.name.replace("_eval.json", "_per_passage_scores.json"), rows)


def copy_machine_translations() -> None:
    out_dir = PKG / "results" / "translations"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / "results" / "translations").glob("*.json")):
        copy_file(path, out_dir / path.name)


def copy_code_and_prompts() -> None:
    for rel in [
        "requirements.txt",
        "scripts/benchmark_translate.py",
        "scripts/benchmark_embed_references.py",
        "scripts/benchmark_evaluate.py",
        "pipeline/align.py",
        "pipeline/verify_alignment.py",
        "pipeline/filter_corpus_v2.py",
        "pipeline/generate_outlier_review.py",
    ]:
        copy_file(ROOT / rel, PKG / "code" / rel)

    copy_file(ROOT / "pipeline" / "sysprompt_v2.txt", PKG / "prompts" / "alignment_prompt.txt")

    translate_script = (ROOT / "scripts" / "benchmark_translate.py").read_text(encoding="utf-8")
    marker = 'SYSTEM_PROMPT = """'
    start = translate_script.find(marker)
    if start != -1:
        start += len(marker)
        end = translate_script.find('"""', start)
        if end != -1:
            (PKG / "prompts").mkdir(parents=True, exist_ok=True)
            (PKG / "prompts" / "translation_prompt.txt").write_text(
                translate_script[start:end].strip() + "\n",
                encoding="utf-8",
            )


def copy_metadata() -> None:
    copy_file(ROOT / "data" / "metadata.json", PKG / "metadata" / "metadata.json")
    copy_file(
        ROOT / "data" / "reference_embeddings_meta.json",
        PKG / "metadata" / "reference_embeddings_meta.json",
    )


def make_manifest() -> None:
    files = []
    for path in sorted(PKG.rglob("*")):
        if path.is_file() and path.name != "build_package.py":
            files.append(
                {
                    "path": str(path.relative_to(PKG)),
                    "bytes": path.stat().st_size,
                }
            )
    write_json(
        PKG / "package_manifest.json",
        {
            "name": "PaliBench reproducibility package",
            "description": "Copyright-conscious reproducibility artifacts for the PaliBench paper.",
            "excluded": [
                "data/test_set.json: contains full Pali passages and human reference translations",
                "data/reference_embeddings.npz: embeddings derived from full human reference translations",
                "results/outlier_reviews/*.json: contains full Pali, human references, and model output snippets",
                "source/*: raw source and aligned translation files, including copyright-sensitive materials",
            ],
            "files": files,
        },
    )


def main() -> None:
    clean_package()
    copy_code_and_prompts()
    copy_metadata()
    make_passage_manifest()
    make_aggregate_results()
    make_per_passage_scores()
    copy_machine_translations()
    make_manifest()


if __name__ == "__main__":
    main()

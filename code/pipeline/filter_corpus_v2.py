#!/usr/bin/env python3
"""
Corpus Filtering Script v2 (Passage-Level)

Applies filtering criteria at the PASSAGE level to determine which passages
would graduate into the final benchmark test set.

Criteria for DROPPING a passage:
1. Any segment has null value (incomplete data)
2. Any segment failed verification (hallucination/error)
3. Passage total length < 100 chars (any translator)
4. Any two passage-level translations are >=90% similar
5. Passage length ratio > 2.0 (max/min total length among English translations)
6. Text contains null characters (alignment corruption)
7. Consecutive segments have duplicate content (alignment error)

Usage:
  python filter_corpus_v2.py                    # Use defaults
  python filter_corpus_v2.py --min-length 150   # Adjust min passage length
  python filter_corpus_v2.py --max-ratio 2.5    # Adjust max length ratio
  python filter_corpus_v2.py --similarity 0.85  # Adjust similarity threshold
"""

import json
import re
import unicodedata
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field


# =============================================================================
# Configuration - Defaults (can be overridden via command line)
# =============================================================================

DEFAULT_MIN_PASSAGE_LENGTH = 100    # Minimum total chars for passage (any translator)
DEFAULT_MAX_LENGTH_RATIO = 2.0      # Max ratio between longest and shortest passage translation
DEFAULT_SIMILARITY_THRESHOLD = 0.90 # Threshold for "too similar" (90%)
DEFAULT_DEDUP_THRESHOLD = 0.85      # Threshold for Pali deduplication (85%)


# =============================================================================
# Helper Functions
# =============================================================================

def normalize_for_comparison(text: str) -> str:
    """Normalize text for similarity comparison."""
    if not text:
        return ""
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    # Lowercase
    text = text.lower()
    # Remove all punctuation and whitespace
    text = re.sub(r'[^\w]', '', text)
    return text


def char_similarity(s1: str, s2: str) -> float:
    """Calculate character-level similarity using longest common subsequence ratio."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    # Normalize both
    n1 = normalize_for_comparison(s1)
    n2 = normalize_for_comparison(s2)

    if not n1 and not n2:
        return 1.0
    if not n1 or not n2:
        return 0.0

    if n1 == n2:
        return 1.0

    # Use a more robust similarity: ratio of matching characters
    # considering both strings
    len1, len2 = len(n1), len(n2)

    # For longer texts, use a windowed approach for efficiency
    if len1 > 1000 or len2 > 1000:
        # Sample-based similarity for very long texts
        shorter, longer = (n1, n2) if len1 <= len2 else (n2, n1)
        chunk_size = 100
        matches = 0
        checks = 0
        for i in range(0, len(shorter), chunk_size):
            chunk = shorter[i:i+chunk_size]
            if chunk in longer:
                matches += 1
            checks += 1
        return matches / checks if checks > 0 else 0.0

    # For moderate length, count common character sequences
    # Use 3-gram overlap as similarity measure
    def get_ngrams(s, n=3):
        return set(s[i:i+n] for i in range(len(s) - n + 1))

    if len(n1) < 3 or len(n2) < 3:
        # Too short for ngrams, use direct comparison
        return 1.0 if n1 == n2 else 0.0

    ngrams1 = get_ngrams(n1)
    ngrams2 = get_ngrams(n2)

    if not ngrams1 or not ngrams2:
        return 0.0

    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)

    return intersection / union if union > 0 else 0.0


def get_passage_id(segment_id: str) -> str:
    """Extract passage ID from segment ID.

    Segment: mn2:1.1, mn2:1.2, mn2:1.3
    Passage: mn2:1
    """
    last_dot = segment_id.rfind('.')
    if last_dot > 0:
        return segment_id[:last_dot]
    return segment_id


def has_null_char(text: str) -> bool:
    """Check if text contains null characters (alignment corruption)."""
    return '\x00' in text if text else False


def sort_segment_ids(segment_ids: list) -> list:
    """Sort segment IDs numerically (e.g., an3.35:2.1, an3.35:2.2, ... an3.35:2.10)."""
    def segment_key(sid):
        # Extract the numeric suffix after the last dot
        last_dot = sid.rfind('.')
        if last_dot > 0:
            prefix = sid[:last_dot]
            try:
                suffix = int(sid[last_dot+1:])
            except ValueError:
                suffix = 0
            return (prefix, suffix)
        return (sid, 0)
    return sorted(segment_ids, key=segment_key)


def check_segment_duplicates(segments: dict, segment_ids: list, translator: str) -> list:
    """
    Check if segments have duplicate/overlapping content.

    Returns list of segment IDs where duplicates were found.

    Patterns detected:
    1. Overlap: segment N contains text from a later segment M (any pair, not just consecutive)
       Example: an3.35:2.1 contains text that is segment an3.35:2.4
    2. Verbatim: any two segments are identical
       Example: sn35.19:1.2 == sn35.19:1.6 == sn35.19:1.9
    """
    duplicates = set()
    sorted_ids = sort_segment_ids(segment_ids)
    texts = [segments.get(sid, {}).get(translator, '') or '' for sid in sorted_ids]
    normalized = [normalize_for_comparison(t) for t in texts]

    # Check for verbatim duplicates (any pair)
    seen_texts = {}  # normalized_text -> first segment_id
    for i, norm_text in enumerate(normalized):
        if norm_text and len(norm_text) > 15:  # Skip very short
            if norm_text in seen_texts:
                duplicates.add(sorted_ids[i])
            else:
                seen_texts[norm_text] = sorted_ids[i]

    # Check for containment: any segment j contained in earlier segment i
    # This catches cases like mn1:17 where segment 4 is contained in segment 2
    for i in range(len(normalized)):
        t1 = normalized[i]
        if len(t1) < 50:
            continue

        for j in range(i + 1, len(normalized)):
            t2 = normalized[j]
            if len(t2) < 15:  # Skip very short
                continue

            if t2 in t1 and sorted_ids[j] not in duplicates:
                duplicates.add(sorted_ids[j])

    return list(duplicates)


@dataclass
class PassageData:
    passage_id: str
    segment_ids: list = field(default_factory=list)
    segments: dict = field(default_factory=dict)  # segment_id -> {pali, sujato, thanissaro, bodhi}

    # Concatenated passage texts
    pali_text: str = ""
    sujato_text: str = ""
    thanissaro_text: str = ""
    bodhi_text: str = ""

    # Flags
    has_null: bool = False
    has_verification_failure: bool = False
    has_null_char: bool = False  # Text contains \x00 (alignment corruption)
    has_segment_duplicate: bool = False  # Consecutive segments have duplicate content
    null_segments: list = field(default_factory=list)
    verification_failed_segments: list = field(default_factory=list)
    null_char_locations: list = field(default_factory=list)  # (translator, segment_id)
    duplicate_segments: list = field(default_factory=list)  # (translator, segment_id)


@dataclass
class PassageFilterResult:
    passage_id: str
    passed: bool
    drop_reasons: list = field(default_factory=list)
    segment_count: int = 0


@dataclass
class FilterStats:
    total_passages: int = 0
    total_segments: int = 0

    passed_passages: int = 0
    passed_segments: int = 0  # segments within passed passages

    # Drop reasons
    dropped_has_null: int = 0
    dropped_verification_failed: int = 0
    dropped_too_short: int = 0
    dropped_too_similar: int = 0
    dropped_length_ratio: int = 0
    dropped_null_char: int = 0  # Null character corruption
    dropped_segment_duplicate: int = 0  # Duplicate content in segments

    # Track which pairs are too similar
    similar_pairs: dict = field(default_factory=lambda: {
        'bodhi_thanissaro': 0,
        'bodhi_sujato': 0,
        'thanissaro_sujato': 0
    })

    # Deduplication stats
    pre_dedup_passages: int = 0
    pre_dedup_segments: int = 0
    duplicates_removed: int = 0
    duplicate_segments_removed: int = 0


def load_suspicious_segments(verification_file: Path) -> set:
    """Load the set of suspicious segment IDs from verification results."""
    suspicious = set()

    if not verification_file.exists():
        print(f"Warning: Verification file not found: {verification_file}")
        return suspicious

    with open(verification_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for translator in ['bodhi', 'thanissaro']:
        if translator in data and 'suspicious_segments' in data[translator]:
            for seg in data[translator]['suspicious_segments']:
                suspicious.add(seg['segment_id'])

    return suspicious


def build_passages(aligned_dir: Path, suspicious_segments: set) -> dict[str, PassageData]:
    """Build passage data structures from aligned files."""
    passages = {}

    for collection in ['an', 'mn', 'sn']:
        collection_dir = aligned_dir / collection
        if not collection_dir.exists():
            continue

        for json_file in sorted(collection_dir.glob('*.json')):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for segment_id, segment_data in data.items():
                passage_id = get_passage_id(segment_id)

                if passage_id not in passages:
                    passages[passage_id] = PassageData(passage_id=passage_id)

                passage = passages[passage_id]
                passage.segment_ids.append(segment_id)
                passage.segments[segment_id] = segment_data

                # Check for nulls
                pali = segment_data.get('pali')
                sujato = segment_data.get('sujato')
                thanissaro = segment_data.get('thanissaro')
                bodhi = segment_data.get('bodhi')

                if any(v is None for v in [pali, sujato, thanissaro, bodhi]):
                    passage.has_null = True
                    passage.null_segments.append(segment_id)

                # Check for verification failures
                if segment_id in suspicious_segments:
                    passage.has_verification_failure = True
                    passage.verification_failed_segments.append(segment_id)

    # Now concatenate texts for each passage and check for issues
    for passage_id, passage in passages.items():
        pali_parts = []
        sujato_parts = []
        thanissaro_parts = []
        bodhi_parts = []

        for segment_id in passage.segment_ids:
            seg = passage.segments[segment_id]
            if seg.get('pali'):
                pali_parts.append(seg['pali'])
            if seg.get('sujato'):
                sujato_parts.append(seg['sujato'])
            if seg.get('thanissaro'):
                thanissaro_parts.append(seg['thanissaro'])
            if seg.get('bodhi'):
                bodhi_parts.append(seg['bodhi'])

            # Check for null characters in any field
            for translator in ['pali', 'sujato', 'thanissaro', 'bodhi']:
                text = seg.get(translator, '')
                if text and has_null_char(text):
                    passage.has_null_char = True
                    passage.null_char_locations.append((translator, segment_id))

        passage.pali_text = ' '.join(pali_parts)
        passage.sujato_text = ' '.join(sujato_parts)
        passage.thanissaro_text = ' '.join(thanissaro_parts)
        passage.bodhi_text = ' '.join(bodhi_parts)

        # Check for segment duplicates in Thanissaro and Bodhi (alignment-generated text)
        for translator in ['thanissaro', 'bodhi']:
            dup_segments = check_segment_duplicates(
                passage.segments, passage.segment_ids, translator
            )
            if dup_segments:
                passage.has_segment_duplicate = True
                for seg_id in dup_segments:
                    passage.duplicate_segments.append((translator, seg_id))

    return passages


def filter_passage(passage: PassageData, config: dict) -> PassageFilterResult:
    """Apply all filtering criteria to a single passage."""
    result = PassageFilterResult(
        passage_id=passage.passage_id,
        passed=True,
        segment_count=len(passage.segment_ids)
    )

    min_length = config['min_length']
    max_ratio = config['max_ratio']
    similarity_threshold = config['similarity']

    # Criterion 1: Any segment has null value
    if passage.has_null:
        result.drop_reasons.append(('has_null', passage.null_segments))

    # Criterion 2: Any segment failed verification
    if passage.has_verification_failure:
        result.drop_reasons.append(('verification_failed', passage.verification_failed_segments))

    # Criterion 6: Text contains null characters (alignment corruption)
    if passage.has_null_char:
        result.drop_reasons.append(('null_char', passage.null_char_locations))

    # Criterion 7: Consecutive segments have duplicate content
    if passage.has_segment_duplicate:
        result.drop_reasons.append(('segment_duplicate', passage.duplicate_segments))

    # For remaining criteria, we need non-empty texts
    lengths = [
        len(passage.sujato_text),
        len(passage.thanissaro_text),
        len(passage.bodhi_text)
    ]

    # Criterion 3: Passage total length < minimum (any translator)
    if any(l < min_length for l in lengths):
        result.drop_reasons.append('too_short')

    # Only check similarity and ratio if we have text
    if all(l > 0 for l in lengths):
        # Criterion 4: Any two passage-level translations are >=90% similar
        sim_bt = char_similarity(passage.bodhi_text, passage.thanissaro_text)
        sim_bs = char_similarity(passage.bodhi_text, passage.sujato_text)
        sim_ts = char_similarity(passage.thanissaro_text, passage.sujato_text)

        similar_pairs = []
        if sim_bt >= similarity_threshold:
            similar_pairs.append(('bodhi_thanissaro', sim_bt))
        if sim_bs >= similarity_threshold:
            similar_pairs.append(('bodhi_sujato', sim_bs))
        if sim_ts >= similarity_threshold:
            similar_pairs.append(('thanissaro_sujato', sim_ts))

        if similar_pairs:
            result.drop_reasons.append(('too_similar', similar_pairs))

        # Criterion 5: Length ratio > threshold
        ratio = max(lengths) / min(lengths)
        if ratio > max_ratio:
            result.drop_reasons.append(('length_ratio', ratio))

    result.passed = len(result.drop_reasons) == 0
    return result


def analyze_corpus(passages: dict[str, PassageData], config: dict) -> tuple[FilterStats, dict]:
    """Analyze the entire corpus and return statistics."""
    stats = FilterStats()
    results = {}

    for passage_id, passage in passages.items():
        stats.total_passages += 1
        stats.total_segments += len(passage.segment_ids)

        result = filter_passage(passage, config)
        results[passage_id] = result

        if result.passed:
            stats.passed_passages += 1
            stats.passed_segments += result.segment_count
        else:
            for reason in result.drop_reasons:
                if isinstance(reason, tuple):
                    reason_type = reason[0]
                    if reason_type == 'has_null':
                        stats.dropped_has_null += 1
                    elif reason_type == 'verification_failed':
                        stats.dropped_verification_failed += 1
                    elif reason_type == 'too_similar':
                        stats.dropped_too_similar += 1
                        for pair_info in reason[1]:
                            pair_name = pair_info[0]
                            stats.similar_pairs[pair_name] += 1
                    elif reason_type == 'length_ratio':
                        stats.dropped_length_ratio += 1
                    elif reason_type == 'null_char':
                        stats.dropped_null_char += 1
                    elif reason_type == 'segment_duplicate':
                        stats.dropped_segment_duplicate += 1
                else:
                    if reason == 'too_short':
                        stats.dropped_too_short += 1

    return stats, results


def get_ngrams_set(text: str, n: int = 3) -> set:
    """Get set of n-grams from normalized text."""
    normalized = normalize_for_comparison(text)
    if len(normalized) < n:
        return {normalized} if normalized else set()
    return set(normalized[i:i+n] for i in range(len(normalized) - n + 1))


def ngram_similarity(ngrams1: set, ngrams2: set) -> float:
    """Calculate Jaccard similarity between two n-gram sets."""
    if not ngrams1 or not ngrams2:
        return 0.0
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    return intersection / union if union > 0 else 0.0


def deduplicate_passages(passages: dict[str, PassageData],
                         results: dict[str, PassageFilterResult],
                         dedup_threshold: float) -> tuple[list[str], list[tuple[str, str, float]]]:
    """
    Deduplicate passages based on Pali text similarity.

    Returns:
        - List of passage IDs to keep (unique passages)
        - List of (duplicate_id, kept_id, similarity) tuples for reporting
    """
    # Get list of passed passages sorted by ID for deterministic ordering
    passed_ids = sorted([pid for pid, result in results.items() if result.passed])

    if not passed_ids:
        return [], []

    print(f"  Deduplicating {len(passed_ids)} passages based on Pali similarity...")
    print(f"  Pre-computing n-grams...")

    # Pre-compute n-grams for all passages (much faster than repeated computation)
    passage_ngrams = {}
    for pid in passed_ids:
        passage_ngrams[pid] = get_ngrams_set(passages[pid].pali_text)

    print(f"  Comparing passages...")

    kept_passages = []  # List of passage IDs we're keeping
    kept_ngrams = []    # Corresponding n-gram sets for fast comparison
    duplicates = []     # List of (duplicate_id, similar_to_id, similarity)

    for i, pid in enumerate(passed_ids):
        if i % 500 == 0 and i > 0:
            print(f"    Processed {i}/{len(passed_ids)} passages...")

        ngrams = passage_ngrams[pid]
        is_duplicate = False

        # Compare against all kept passages using pre-computed n-grams
        for j, kept_id in enumerate(kept_passages):
            similarity = ngram_similarity(ngrams, kept_ngrams[j])

            if similarity >= dedup_threshold:
                # This is a duplicate
                duplicates.append((pid, kept_id, similarity))
                is_duplicate = True
                break

        if not is_duplicate:
            kept_passages.append(pid)
            kept_ngrams.append(ngrams)

    print(f"  Deduplication complete: {len(kept_passages)} unique, {len(duplicates)} duplicates")

    return kept_passages, duplicates


def print_report(stats: FilterStats, config: dict):
    """Print a detailed filtering report."""
    print("\n" + "=" * 80)
    print("CORPUS FILTERING REPORT v2 (PASSAGE-LEVEL)")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  MIN_PASSAGE_LENGTH = {config['min_length']} chars")
    print(f"  MAX_LENGTH_RATIO = {config['max_ratio']}")
    print(f"  SIMILARITY_THRESHOLD = {config['similarity']} ({int(config['similarity']*100)}%)")
    print(f"  DEDUP_THRESHOLD = {config['dedup']} ({int(config['dedup']*100)}%)")

    print("\n" + "-" * 80)
    print("PASSAGE-LEVEL FILTERING")
    print("-" * 80)

    print(f"\nTotal passages: {stats.total_passages:,}")
    print(f"Total segments: {stats.total_segments:,}")

    print(f"\nPassed passages: {stats.passed_passages:,} ({100*stats.passed_passages/stats.total_passages:.1f}%)")
    print(f"Dropped passages: {stats.total_passages - stats.passed_passages:,} ({100*(stats.total_passages - stats.passed_passages)/stats.total_passages:.1f}%)")

    print(f"\nDrop reasons (passages can have multiple):")
    print(f"  Has null segment:        {stats.dropped_has_null:,}")
    print(f"  Verification failed:     {stats.dropped_verification_failed:,}")
    print(f"  Too short (<{config['min_length']} chars):   {stats.dropped_too_short:,}")
    print(f"  Too similar (>={int(config['similarity']*100)}%):      {stats.dropped_too_similar:,}")
    print(f"    - Bodhi ~ Thanissaro:  {stats.similar_pairs['bodhi_thanissaro']:,}")
    print(f"    - Bodhi ~ Sujato:      {stats.similar_pairs['bodhi_sujato']:,}")
    print(f"    - Thanissaro ~ Sujato: {stats.similar_pairs['thanissaro_sujato']:,}")
    print(f"  Length ratio >{config['max_ratio']}:       {stats.dropped_length_ratio:,}")
    print(f"  Null char corruption:    {stats.dropped_null_char:,}")
    print(f"  Segment duplicates:      {stats.dropped_segment_duplicate:,}")

    print("\n" + "-" * 80)
    print("DEDUPLICATION (Pali-based)")
    print("-" * 80)

    print(f"\nPassages before dedup: {stats.pre_dedup_passages:,}")
    print(f"Duplicates removed: {stats.duplicates_removed:,}")
    print(f"  (Segments in duplicates: {stats.duplicate_segments_removed:,})")

    print("\n" + "-" * 80)
    print("FINAL BENCHMARK CORPUS")
    print("-" * 80)

    print(f"\nPassages graduating to benchmark: {stats.passed_passages:,}")
    print(f"  ({100*stats.passed_passages/stats.total_passages:.1f}% of {stats.total_passages:,} total passages)")

    print(f"\nSegments in passed passages: {stats.passed_segments:,}")
    print(f"  ({100*stats.passed_segments/stats.total_segments:.1f}% of {stats.total_segments:,} total segments)")

    # Estimate token count
    estimated_tokens = int(stats.passed_segments / stats.total_segments * 919825)
    print(f"\nEstimated tokens: ~{estimated_tokens:,}")
    print(f"  (based on original corpus of ~920k tokens)")

    print("\n" + "=" * 80)


def create_test_set(passages: dict[str, PassageData],
                    kept_passage_ids: list[str],
                    results: dict[str, PassageFilterResult],
                    duplicates: list[tuple[str, str, float]],
                    stats: FilterStats,
                    config: dict,
                    output_dir: Path):
    """Create the benchmark test set and metadata files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build test set data
    test_set = {}
    for pid in sorted(kept_passage_ids):
        passage = passages[pid]
        test_set[pid] = {
            "pali": passage.pali_text,
            "sujato": passage.sujato_text,
            "thanissaro": passage.thanissaro_text,
            "bodhi": passage.bodhi_text,
            "segment_count": len(passage.segment_ids),
            "segments": passage.segment_ids
        }

    # Write test set
    test_set_file = output_dir / "test_set.json"
    with open(test_set_file, 'w', encoding='utf-8') as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)
    print(f"\nTest set written to: {test_set_file}")

    # Build metadata
    # Categorize dropped passages by reason
    dropped_by_reason = {
        'has_null': [],
        'verification_failed': [],
        'too_short': [],
        'too_similar': [],
        'length_ratio': [],
        'null_char': [],
        'segment_duplicate': [],
        'duplicate': []
    }

    for pid, result in results.items():
        if not result.passed:
            for reason in result.drop_reasons:
                if isinstance(reason, tuple):
                    reason_type = reason[0]
                    if reason_type in dropped_by_reason:
                        dropped_by_reason[reason_type].append(pid)
                elif reason in dropped_by_reason:
                    dropped_by_reason[reason].append(pid)

    # Add duplicates
    for dup_id, kept_id, sim in duplicates:
        dropped_by_reason['duplicate'].append({
            'passage_id': dup_id,
            'duplicate_of': kept_id,
            'similarity': round(sim, 4)
        })

    metadata = {
        "description": "PaliBench - Pali Canon Translation Benchmark Test Set",
        "version": "1.0",
        "creation_date": str(Path(__file__).stat().st_mtime),
        "configuration": {
            "min_passage_length_chars": config['min_length'],
            "max_length_ratio": config['max_ratio'],
            "translation_similarity_threshold": config['similarity'],
            "pali_deduplication_threshold": config['dedup']
        },
        "statistics": {
            "original_corpus": {
                "passages": stats.total_passages,
                "segments": stats.total_segments,
                "estimated_tokens": 919825
            },
            "after_filtering": {
                "passages": stats.pre_dedup_passages,
                "segments": stats.pre_dedup_segments
            },
            "after_deduplication": {
                "passages": stats.passed_passages,
                "segments": stats.passed_segments,
                "estimated_tokens": int(stats.passed_segments / stats.total_segments * 919825)
            },
            "retention_rate": {
                "passages": round(stats.passed_passages / stats.total_passages, 4),
                "segments": round(stats.passed_segments / stats.total_segments, 4)
            }
        },
        "filtering_results": {
            "dropped_has_null": stats.dropped_has_null,
            "dropped_verification_failed": stats.dropped_verification_failed,
            "dropped_too_short": stats.dropped_too_short,
            "dropped_too_similar": stats.dropped_too_similar,
            "dropped_length_ratio": stats.dropped_length_ratio,
            "dropped_null_char": stats.dropped_null_char,
            "dropped_segment_duplicate": stats.dropped_segment_duplicate,
            "duplicates_removed": stats.duplicates_removed
        },
        "dropped_passages": dropped_by_reason
    }

    # Write metadata
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadata written to: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description='Filter corpus for benchmark - passage level')
    parser.add_argument('--min-length', type=int, default=DEFAULT_MIN_PASSAGE_LENGTH,
                        help=f'Minimum total passage length in chars (default: {DEFAULT_MIN_PASSAGE_LENGTH})')
    parser.add_argument('--max-ratio', type=float, default=DEFAULT_MAX_LENGTH_RATIO,
                        help=f'Max length ratio between translations (default: {DEFAULT_MAX_LENGTH_RATIO})')
    parser.add_argument('--similarity', type=float, default=DEFAULT_SIMILARITY_THRESHOLD,
                        help=f'Similarity threshold for "too similar" (default: {DEFAULT_SIMILARITY_THRESHOLD})')
    parser.add_argument('--dedup', type=float, default=DEFAULT_DEDUP_THRESHOLD,
                        help=f'Pali deduplication threshold (default: {DEFAULT_DEDUP_THRESHOLD})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only show statistics, do not create output files')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory for test set (default: data)')
    args = parser.parse_args()

    config = {
        'min_length': args.min_length,
        'max_ratio': args.max_ratio,
        'similarity': args.similarity,
        'dedup': args.dedup
    }

    repo_root = Path(__file__).resolve().parent.parent
    aligned_dir = repo_root / "source" / "aligned"
    verification_file = repo_root / "pipeline" / "verification_suspicious.json"
    output_dir = repo_root / args.output_dir

    print("Loading verification results...")
    suspicious_segments = load_suspicious_segments(verification_file)
    print(f"  Found {len(suspicious_segments)} suspicious segments")

    print("\nBuilding passage data...")
    passages = build_passages(aligned_dir, suspicious_segments)
    print(f"  Built {len(passages)} passages")

    print("\nAnalyzing corpus (passage-level filtering)...")
    stats, results = analyze_corpus(passages, config)

    # Store pre-dedup counts
    stats.pre_dedup_passages = stats.passed_passages
    stats.pre_dedup_segments = stats.passed_segments

    # Deduplicate
    print("\nDeduplicating passages based on Pali similarity...")
    kept_passages, duplicates = deduplicate_passages(passages, results, config['dedup'])

    # Update stats after deduplication
    stats.duplicates_removed = len(duplicates)
    stats.passed_passages = len(kept_passages)

    # Calculate segments in kept passages
    stats.passed_segments = sum(len(passages[pid].segment_ids) for pid in kept_passages)
    stats.duplicate_segments_removed = stats.pre_dedup_segments - stats.passed_segments

    print_report(stats, config)

    # Show some example duplicates
    if duplicates:
        print("\nSample duplicates found (first 10):")
        for dup_id, kept_id, sim in duplicates[:10]:
            print(f"  {dup_id} ~ {kept_id} ({sim*100:.1f}% similar)")

    # Create output files unless dry-run
    if not args.dry_run:
        print("\n" + "=" * 80)
        print("CREATING TEST SET")
        print("=" * 80)
        create_test_set(passages, kept_passages, results, duplicates, stats, config, output_dir)
        print("\nDone! Test set created successfully.")
    else:
        print("\n[Dry run - no files created. Remove --dry-run to create test set.]")


if __name__ == "__main__":
    main()

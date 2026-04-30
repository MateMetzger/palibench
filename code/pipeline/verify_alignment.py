#!/usr/bin/env python3
"""
Alignment Verification Script

Verifies that aligned Thanissaro and Bodhi segments can be found
in their respective source text files.

Categories of non-matches:
1. VERBATIM - found exactly as-is
2. NORMALIZED - found after text normalization (punctuation, whitespace, unicode)
3. EXPANDED - legitimate expansion of abbreviated source text
4. SUSPICIOUS - potential hallucination or alignment error
"""

import json
import os
import re
import unicodedata
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


@dataclass
class VerificationResult:
    segment_id: str
    translator: str
    aligned_text: str
    match_type: str  # VERBATIM, NORMALIZED, EXPANDED, CROSS_REF, SUSPICIOUS, NULL
    details: str = ""
    source_excerpt: str = ""  # nearby text in source if suspicious


def source_has_cross_reference(source: str) -> bool:
    """Check if source contains cross-reference patterns indicating content from another sutta."""
    cross_ref_patterns = [
        r'identical with those of \d+:\d+',
        r'cites.*entire discourse of \d+:\d+',
        r'see \d+:\d+',
        r'\[\d+[–-]\d+\].*\.\.\.',
        r'The verses? (?:is|are) (?:the same|identical)',
        r'as in \d+:\d+',
        r'omitted.*see',
    ]
    source_lower = source.lower()
    for pattern in cross_ref_patterns:
        if re.search(pattern, source_lower):
            return True
    return False


@dataclass
class VerificationStats:
    total: int = 0
    verbatim: int = 0
    normalized: int = 0
    expanded: int = 0
    cross_ref: int = 0
    suspicious: int = 0
    null_values: int = 0
    suspicious_segments: list = field(default_factory=list)


def normalize_text(text: str, strip_numbers: bool = False) -> str:
    """Normalize text for fuzzy matching."""
    if not text:
        return ""

    # Normalize unicode (NFKC normalizes compatibility characters)
    text = unicodedata.normalize('NFKC', text)

    # Convert various quote styles to simple quotes
    text = re.sub(r'[""„‟❝❞〝〞]', '"', text)
    text = re.sub(r"[''‚‛❛❜`]", "'", text)

    # Normalize dashes
    text = re.sub(r'[–—―‐‑‒]', '-', text)

    # Normalize ellipsis
    text = re.sub(r'\.{2,}|…', '...', text)

    # Normalize whitespace (including newlines)
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    # Optionally strip parenthetical numbers like (1), (2), etc.
    if strip_numbers:
        text = re.sub(r'\(\d+\)\s*', '', text)
        # Also strip standalone numbers at start of sentences after normalization
        text = re.sub(r'^\d+\.\s*', '', text)

    return text


def strip_punctuation(text: str) -> str:
    """Remove punctuation for even fuzzier matching."""
    if not text:
        return ""
    # Keep alphanumeric and spaces only
    return re.sub(r'[^\w\s]', '', text).lower()


def find_in_source(segment: str, source: str, normalized_source: str, stripped_source: str,
                   normalized_source_no_nums: str, has_cross_ref: bool) -> tuple[str, str]:
    """
    Try to find segment in source text with increasing fuzziness.
    Returns (match_type, details).
    """
    if not segment:
        return ("NULL", "")

    # 1. Try verbatim match
    if segment in source:
        return ("VERBATIM", "exact match")

    # 2. Try normalized match
    normalized_segment = normalize_text(segment)
    if normalized_segment in normalized_source:
        return ("NORMALIZED", "matched after normalization")

    # 3. Try stripped match (no punctuation, lowercase)
    stripped_segment = strip_punctuation(normalized_segment)
    if len(stripped_segment) > 10 and stripped_segment in stripped_source:
        return ("NORMALIZED", "matched after stripping punctuation")

    # 4. Try match after stripping parenthetical numbers from source
    # (source might have "(1) They grow" but aligned text has "They grow")
    normalized_segment_no_nums = normalize_text(segment, strip_numbers=True)
    if normalized_segment_no_nums in normalized_source_no_nums:
        return ("NORMALIZED", "matched after stripping list numbers")

    # 5. Check for expansion patterns (segment contains reconstructed text)
    # This happens when Pali is full but translator used "..." abbreviation
    if is_likely_expansion(segment, source, normalized_source):
        return ("EXPANDED", "appears to be legitimate expansion")

    # 6. Check for cross-reference expansions
    # If source references another sutta, content may be legitimately expanded
    if has_cross_ref:
        # For cross-reference files, be more lenient
        # Check if key words appear in source or if it's a short formulaic term
        words = normalized_segment.split()
        if len(words) <= 3:
            # Very short - likely a term from the referenced section
            return ("CROSS_REF", "short term from referenced sutta")
        # Check if at least some words match
        content_words = [w for w in words if len(w) > 3 and w.isalpha()]
        if content_words:
            found = sum(1 for w in content_words if w.lower() in normalized_source.lower())
            if found >= 1:  # At least one content word found
                return ("CROSS_REF", "content from referenced sutta")

    return ("SUSPICIOUS", "not found in source")


def is_likely_expansion(segment: str, source: str, normalized_source: str) -> bool:
    """
    Detect if segment is a legitimate expansion of abbreviated source text.

    The AI was instructed to expand when:
    - Pali is fully spelled out (no ellipsis)
    - But translator used "..." abbreviation
    - AI should find template and substitute terms

    Heuristics:
    1. If significant portions of the segment appear in source, it's likely expanded
    2. If segment follows a repetitive pattern seen elsewhere in source
    3. If the source contains "..." near similar content
    """
    normalized_segment = normalize_text(segment)
    source_has_abbrev = '...' in source or '…' in source

    # Check if substantial portions appear in source
    # Split into chunks and see how many match
    words = normalized_segment.split()

    # For very short segments (1-3 words)
    if len(words) < 4:
        if source_has_abbrev:
            # Check if key words from segment appear in source
            key_words = [w for w in words if len(w) > 3]
            if key_words:
                matches = sum(1 for w in key_words if w.lower() in normalized_source.lower())
                if matches >= len(key_words) * 0.5:
                    return True
            # For very short segments, also check if the exact phrase minus punctuation appears
            stripped_seg = strip_punctuation(normalized_segment)
            stripped_src = strip_punctuation(normalized_source)
            if stripped_seg and stripped_seg in stripped_src:
                return True
        return False

    # Check for overlapping word sequences
    window_sizes = [10, 7, 5, 4, 3]  # Try different window sizes
    for window in window_sizes:
        if len(words) >= window:
            matches = 0
            total_windows = len(words) - window + 1
            for i in range(total_windows):
                chunk = ' '.join(words[i:i+window])
                if chunk in normalized_source:
                    matches += 1

            # If more than 20% of windows match, likely expansion
            if total_windows > 0 and matches / total_windows > 0.20:
                return True

    # Check for key phrase matches (beginning and end of segment)
    beginning = ' '.join(words[:4]) if len(words) >= 4 else ' '.join(words)
    ending = ' '.join(words[-4:]) if len(words) >= 4 else ' '.join(words)

    beginning_match = beginning in normalized_source
    ending_match = ending in normalized_source

    if beginning_match or ending_match:
        # At least one end matches - likely a template expansion
        return True

    # Check if the segment looks like a repetitive formula that exists in source
    # with different terms (e.g., "mind with greed" pattern when source has "mind with X")
    if source_has_abbrev:
        # Source has abbreviations - check if segment matches a pattern
        # Extract unique content words
        content_words = [w for w in words if len(w) > 3 and w.isalpha()]
        if len(content_words) >= 2:
            # Check if at least 40% of content words appear in source
            found = sum(1 for w in content_words if w.lower() in normalized_source.lower())
            if found >= len(content_words) * 0.4:
                return True

    # Check for common Buddhist formulaic phrases that get expanded
    # These patterns are often abbreviated in translations
    formulaic_patterns = [
        "excellent",
        "righting the overturned",
        "revealing the hidden",
        "pointing out the path",
        "lighting a lamp",
        "that's what i said",
        "why did i say it",
    ]
    segment_lower = normalized_segment.lower()
    for pattern in formulaic_patterns:
        if pattern in segment_lower and source_has_abbrev:
            return True

    return False


def get_source_excerpt(segment: str, source: str, context_chars: int = 100) -> str:
    """Find the most similar region in source for debugging."""
    normalized_segment = normalize_text(segment)
    normalized_source = normalize_text(source)

    # Find first few words
    words = normalized_segment.split()[:4]
    if not words:
        return ""

    search_phrase = words[0]
    for i, word in enumerate(words[1:], 1):
        test_phrase = ' '.join(words[:i+1])
        if test_phrase.lower() in normalized_source.lower():
            search_phrase = test_phrase
        else:
            break

    # Find position in source
    pos = normalized_source.lower().find(search_phrase.lower())
    if pos >= 0:
        start = max(0, pos - 20)
        end = min(len(normalized_source), pos + len(search_phrase) + context_chars)
        return f"...{normalized_source[start:end]}..."

    return ""


def verify_file(aligned_path: Path, bodhi_raw_dir: Path, thanissaro_raw_dir: Path) -> dict[str, VerificationStats]:
    """Verify a single aligned JSON file."""

    # Determine collection (an, mn, sn) and filename
    collection = aligned_path.parent.name
    stem = aligned_path.stem

    # Load aligned data
    with open(aligned_path, 'r', encoding='utf-8') as f:
        aligned_data = json.load(f)

    # Load source files
    bodhi_path = bodhi_raw_dir / collection / f"{stem}.txt"
    thanissaro_path = thanissaro_raw_dir / collection / f"{stem}.txt"

    results = {
        'bodhi': VerificationStats(),
        'thanissaro': VerificationStats()
    }

    for translator, raw_path in [('bodhi', bodhi_path), ('thanissaro', thanissaro_path)]:
        stats = results[translator]

        if not raw_path.exists():
            print(f"  Warning: Source file not found: {raw_path}")
            continue

        with open(raw_path, 'r', encoding='utf-8') as f:
            source = f.read()

        normalized_source = normalize_text(source)
        stripped_source = strip_punctuation(normalized_source)
        normalized_source_no_nums = normalize_text(source, strip_numbers=True)
        has_cross_ref = source_has_cross_reference(source)

        for segment_id, segment_data in aligned_data.items():
            stats.total += 1

            aligned_text = segment_data.get(translator)

            if aligned_text is None:
                stats.null_values += 1
                continue

            match_type, details = find_in_source(
                aligned_text, source, normalized_source, stripped_source,
                normalized_source_no_nums, has_cross_ref
            )

            if match_type == "VERBATIM":
                stats.verbatim += 1
            elif match_type == "NORMALIZED":
                stats.normalized += 1
            elif match_type == "EXPANDED":
                stats.expanded += 1
            elif match_type == "CROSS_REF":
                stats.cross_ref += 1
            elif match_type == "SUSPICIOUS":
                stats.suspicious += 1
                excerpt = get_source_excerpt(aligned_text, source)
                stats.suspicious_segments.append(VerificationResult(
                    segment_id=segment_id,
                    translator=translator,
                    aligned_text=aligned_text,
                    match_type=match_type,
                    details=details,
                    source_excerpt=excerpt
                ))

    return results


def main():
    repo_root = Path(__file__).resolve().parent.parent
    aligned_dir = repo_root / "source" / "aligned"
    bodhi_raw_dir = repo_root / "source" / "bodhi"
    thanissaro_raw_dir = repo_root / "source" / "thanissaro"

    # Collect all aligned files
    aligned_files = []
    for collection in ['an', 'mn', 'sn']:
        collection_dir = aligned_dir / collection
        if collection_dir.exists():
            aligned_files.extend(sorted(collection_dir.glob("*.json")))

    print(f"Found {len(aligned_files)} aligned files to verify\n")

    # Aggregate stats
    total_stats = {
        'bodhi': VerificationStats(),
        'thanissaro': VerificationStats()
    }

    all_suspicious = {
        'bodhi': [],
        'thanissaro': []
    }

    for aligned_path in aligned_files:
        print(f"Verifying {aligned_path.relative_to(repo_root)}...")

        file_results = verify_file(aligned_path, bodhi_raw_dir, thanissaro_raw_dir)

        for translator in ['bodhi', 'thanissaro']:
            stats = file_results[translator]
            total = total_stats[translator]

            total.total += stats.total
            total.verbatim += stats.verbatim
            total.normalized += stats.normalized
            total.expanded += stats.expanded
            total.cross_ref += stats.cross_ref
            total.suspicious += stats.suspicious
            total.null_values += stats.null_values

            all_suspicious[translator].extend(stats.suspicious_segments)

    # Print summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    for translator in ['bodhi', 'thanissaro']:
        stats = total_stats[translator]
        print(f"\n{translator.upper()}:")
        print(f"  Total segments:  {stats.total:,}")
        print(f"  Null values:     {stats.null_values:,} ({100*stats.null_values/stats.total:.1f}%)")

        non_null = stats.total - stats.null_values
        if non_null > 0:
            print(f"  Verbatim:        {stats.verbatim:,} ({100*stats.verbatim/non_null:.1f}%)")
            print(f"  Normalized:      {stats.normalized:,} ({100*stats.normalized/non_null:.1f}%)")
            print(f"  Expanded:        {stats.expanded:,} ({100*stats.expanded/non_null:.1f}%)")
            print(f"  Cross-ref:       {stats.cross_ref:,} ({100*stats.cross_ref/non_null:.1f}%)")
            print(f"  Suspicious:      {stats.suspicious:,} ({100*stats.suspicious/non_null:.1f}%)")

    # Write suspicious segments to file for review
    output_path = repo_root / "pipeline" / "verification_suspicious.json"
    suspicious_output = {
        'summary': {
            'total_segments': total_stats['bodhi'].total + total_stats['thanissaro'].total,
            'total_suspicious': total_stats['bodhi'].suspicious + total_stats['thanissaro'].suspicious,
            'suspicious_rate': f"{100 * (total_stats['bodhi'].suspicious + total_stats['thanissaro'].suspicious) / (total_stats['bodhi'].total + total_stats['thanissaro'].total - total_stats['bodhi'].null_values - total_stats['thanissaro'].null_values):.2f}%",
            'categories': {
                'VERBATIM': 'Found exactly in source - no changes',
                'NORMALIZED': 'Found after normalizing quotes, whitespace, punctuation, or list numbers',
                'EXPANDED': 'Legitimate expansion of abbreviated source text (... in source)',
                'CROSS_REF': 'Content from a cross-referenced sutta (source says "see sutta X")',
                'SUSPICIOUS': 'Not found in source - requires manual review'
            },
            'note': 'Suspicious segments may include: (1) genuine hallucinations/errors from GPT, (2) text normalization edge cases, (3) translator using different phrasing than Pali structure'
        },
        'bodhi': {
            'stats': {
                'total': total_stats['bodhi'].total,
                'null_values': total_stats['bodhi'].null_values,
                'verbatim': total_stats['bodhi'].verbatim,
                'normalized': total_stats['bodhi'].normalized,
                'expanded': total_stats['bodhi'].expanded,
                'cross_ref': total_stats['bodhi'].cross_ref,
                'suspicious': total_stats['bodhi'].suspicious
            },
            'suspicious_segments': [
                {
                    'segment_id': r.segment_id,
                    'aligned_text': r.aligned_text,
                    'source_excerpt': r.source_excerpt,
                    'details': r.details
                }
                for r in all_suspicious['bodhi']
            ]
        },
        'thanissaro': {
            'stats': {
                'total': total_stats['thanissaro'].total,
                'null_values': total_stats['thanissaro'].null_values,
                'verbatim': total_stats['thanissaro'].verbatim,
                'normalized': total_stats['thanissaro'].normalized,
                'expanded': total_stats['thanissaro'].expanded,
                'cross_ref': total_stats['thanissaro'].cross_ref,
                'suspicious': total_stats['thanissaro'].suspicious
            },
            'suspicious_segments': [
                {
                    'segment_id': r.segment_id,
                    'aligned_text': r.aligned_text,
                    'source_excerpt': r.source_excerpt,
                    'details': r.details
                }
                for r in all_suspicious['thanissaro']
            ]
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(suspicious_output, f, indent=2, ensure_ascii=False)

    print(f"\nSuspicious segments written to: {output_path}")

    # Print first few suspicious for immediate review (handle encoding issues)
    print("\n" + "="*80)
    print("SAMPLE SUSPICIOUS SEGMENTS (first 10 per translator)")
    print("="*80)

    for translator in ['bodhi', 'thanissaro']:
        print(f"\n{translator.upper()}:")
        for i, result in enumerate(all_suspicious[translator][:10]):
            try:
                seg_id = result.segment_id
                aligned = result.aligned_text[:100] + ('...' if len(result.aligned_text) > 100 else '')
                # Encode/decode to handle console encoding issues
                aligned = aligned.encode('ascii', 'replace').decode('ascii')
                print(f"\n  [{i+1}] {seg_id}")
                print(f"      Aligned: {aligned}")
                if result.source_excerpt:
                    excerpt = result.source_excerpt[:100] + ('...' if len(result.source_excerpt) > 100 else '')
                    excerpt = excerpt.encode('ascii', 'replace').decode('ascii')
                    print(f"      Nearest: {excerpt}")
            except Exception as e:
                print(f"\n  [{i+1}] {result.segment_id} (encoding error: {e})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Alignment pipeline: segment-by-segment alignment with prompt caching.

Processes one passage at a time, one translator at a time.
Leverages OpenAI prompt caching for efficiency.
Supports resume capability.
"""
import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    print("ERROR: requests is required. Install with 'pip install requests'.")
    sys.exit(1)


def load_dotenv(path: Path):
    """Load environment variables from .env file."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = val


import re

def parse_uid_range(uid_spec: str, collection: str, available_uids: list[str]) -> list[str]:
    """
    Parse a UID specification which can be:
    - A single UID: "mn1"
    - A range: "mn1-mn10" (inclusive, returns only existing files)
    
    Returns list of UIDs to process (only those that exist).
    """
    if "-" not in uid_spec:
        # Single UID
        return [uid_spec] if uid_spec in available_uids else []
    
    # Range: extract start and end numbers
    parts = uid_spec.split("-", 1)
    if len(parts) != 2:
        return [uid_spec] if uid_spec in available_uids else []
    
    start_uid, end_uid = parts
    
    # Extract numeric parts (handles mn1, an3.1, sn12.1, etc.)
    # Match the number after the collection prefix
    start_match = re.match(rf"{collection}(\d+(?:\.\d+)?)", start_uid)
    end_match = re.match(rf"{collection}(\d+(?:\.\d+)?)", end_uid)
    
    if not start_match or not end_match:
        return [uid_spec] if uid_spec in available_uids else []
    
    start_num = start_match.group(1)
    end_num = end_match.group(1)
    
    # For simple numeric ranges (mn1-mn10), filter available UIDs
    result = []
    for uid in available_uids:
        match = re.match(rf"{collection}(\d+(?:\.\d+)?)", uid)
        if match:
            uid_num = match.group(1)
            # Compare as tuples of floats for proper ordering
            def to_tuple(s):
                parts = s.split(".")
                return tuple(int(p) for p in parts)
            
            try:
                uid_t = to_tuple(uid_num)
                start_t = to_tuple(start_num)
                end_t = to_tuple(end_num)
                if start_t <= uid_t <= end_t:
                    result.append(uid)
            except ValueError:
                continue
    
    return result


def get_passages(pali_json: dict) -> list[str]:
    """
    Extract unique passage prefixes from segment keys.
    E.g., "mn1:3.1" -> "mn1:3"
    """
    passages = []
    seen = set()
    for key in pali_json.keys():
        # Split by last dot to get passage prefix
        if "." in key:
            passage = key.rsplit(".", 1)[0]
        else:
            passage = key
        if passage not in seen:
            seen.add(passage)
            passages.append(passage)
    return passages


def get_passage_segments(pali_json: dict, sujato_json: dict, passage: str) -> dict:
    """
    Get all segments for a specific passage.
    Returns dict with pali and sujato for each segment.
    """
    segments = {}
    for key in pali_json.keys():
        if key.startswith(passage + ".") or key == passage:
            segments[key] = {
                "pali": pali_json[key],
                "sujato": sujato_json.get(key, ""),
            }
    return segments


def is_passage_processed(
    aligned_json: dict, passage: str, translator: str, force_if_all_null: bool = True
) -> bool:
    """
    Check if a passage has been processed for a specific translator.
    Returns True if ALL segments in the passage have the translator key with non-null values.
    If force_if_all_null is True, returns False if all values are null (allowing reprocessing).
    """
    has_any_key = False
    all_null = True
    
    for key in aligned_json.keys():
        if key.startswith(passage + ".") or key == passage:
            if translator not in aligned_json[key]:
                return False  # Key missing, not processed
            has_any_key = True
            if aligned_json[key][translator] is not None:
                all_null = False
    
    if not has_any_key:
        return False  # No keys found for this passage
    
    # If all values are null and force_if_all_null is enabled, treat as unprocessed
    if force_if_all_null and all_null:
        return False
    
    return True


def batch_passages(
    passages: list[str],
    pali_json: dict,
    sujato_json: dict,
    aligned_json: dict,
    translator: str,
    target_segments: int = 15,
    max_segments: int = 30,
) -> list[tuple[list[str], dict]]:
    """
    Group passages into batches targeting a certain segment count.
    Skips already-processed passages.
    
    Returns list of (passage_names, combined_segments) tuples.
    """
    batches = []
    current_passages = []
    current_segments = {}
    
    for passage in passages:
        # Skip already processed passages
        if is_passage_processed(aligned_json, passage, translator):
            continue
        
        # Get segments for this passage
        passage_segments = {}
        for key in pali_json.keys():
            if key.startswith(passage + ".") or key == passage:
                passage_segments[key] = {
                    "pali": pali_json[key],
                    "sujato": sujato_json.get(key, ""),
                }
        
        passage_size = len(passage_segments)
        
        # If adding this passage would exceed max, flush current batch first
        if current_segments and len(current_segments) + passage_size > max_segments:
            batches.append((current_passages.copy(), current_segments.copy()))
            current_passages = []
            current_segments = {}
        
        # Add passage to current batch
        current_passages.append(passage)
        current_segments.update(passage_segments)
        
        # If we've reached target size, flush batch
        if len(current_segments) >= target_segments:
            batches.append((current_passages.copy(), current_segments.copy()))
            current_passages = []
            current_segments = {}
    
    # Don't forget remaining passages
    if current_segments:
        batches.append((current_passages, current_segments))
    
    return batches


def extract_response_text(resp: dict) -> str:
    """Extract text from OpenAI API response."""
    if "output_text" in resp and isinstance(resp["output_text"], str):
        return resp["output_text"]
    for item in resp.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") in ("output_text", "text"):
                    return content.get("text", "")
    return ""


def call_alignment_api(
    system_prompt: str,
    segments: dict,
    translator_name: str,
    translator_text: str,
    api_key: str,
    model: str = "gpt-5-mini-2025-08-07",
    reasoning_effort: str = "medium",
) -> dict:
    """
    Call OpenAI API to align segments with translator text.
    Returns dict mapping segment IDs to extracted text.
    """
    user_prompt = (
        f"Extract {translator_name}'s translations for these segments.\n\n"
        "SEGMENTS:\n"
        f"{json.dumps(segments, ensure_ascii=False, indent=2)}\n\n"
        f"{translator_name.upper()}_FULL_TEXT:\n<<<\n"
        f"{translator_text}\n>>>\n"
    )

    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ],
        "text": {"format": {"type": "json_object"}},
        "reasoning": {"effort": reasoning_effort},
        "max_output_tokens": 16000,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(
        "https://api.openai.com/v1/responses",
        headers=headers,
        json=payload,
        timeout=300,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    text = extract_response_text(data).strip()

    if not text:
        raise RuntimeError("Empty response from API")

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON response: {e}\nResponse: {text[:500]}")


def merge_results(
    aligned_json: dict,
    segments: dict,
    translator: str,
    api_response: dict,
) -> dict:
    """
    Merge API response into aligned JSON.
    Ensures all segment keys exist with pali/sujato, adds translator field.
    """
    for key in segments.keys():
        if key not in aligned_json:
            aligned_json[key] = {
                "pali": segments[key]["pali"],
                "sujato": segments[key]["sujato"],
            }
        # Add translator result (could be string or null)
        aligned_json[key][translator] = api_response.get(key)
    return aligned_json


def process_sutta(
    collection: str,
    uid: str,
    translator: str,
    root: Path,
    api_key: str,
    dry_run: bool = False,
    verbose: bool = False,
    limit: int = 0,
    batch_target: int = 15,
    batch_max: int = 30,
    reasoning_effort: str = "medium",
) -> dict:
    """
    Process a single sutta for a single translator.
    Uses batched passage processing to reduce API calls.
    Returns stats dict.
    """
    stats = {"batches": 0, "passages": 0, "segments": 0, "extracted": 0, "null": 0, "skipped": 0}

    # Load source files
    pali_path = root / "source" / "pali" / collection / f"{uid}.json"
    sujato_path = root / "source" / "sujato" / collection / f"{uid}.json"
    translator_path = root / "source" / translator / collection / f"{uid}.txt"

    for p in [pali_path, sujato_path, translator_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    pali_json = json.loads(pali_path.read_text(encoding="utf-8"))
    sujato_json = json.loads(sujato_path.read_text(encoding="utf-8"))
    translator_text = translator_path.read_text(encoding="utf-8", errors="ignore")

    # Load system prompt
    system_prompt = (root / "pipeline" / "sysprompt_v2.txt").read_text(encoding="utf-8")

    # Load or initialize aligned file
    aligned_path = root / "source" / "aligned" / collection / f"{uid}.json"
    aligned_path.parent.mkdir(parents=True, exist_ok=True)

    if aligned_path.exists():
        aligned_json = json.loads(aligned_path.read_text(encoding="utf-8"))
    else:
        aligned_json = {}

    # Get all passages
    all_passages = get_passages(pali_json)
    
    # Count skipped passages for stats
    for passage in all_passages:
        if is_passage_processed(aligned_json, passage, translator):
            stats["skipped"] += 1
    
    # Create batches (already filters out processed passages)
    batches = batch_passages(
        passages=all_passages,
        pali_json=pali_json,
        sujato_json=sujato_json,
        aligned_json=aligned_json,
        translator=translator,
        target_segments=batch_target,
        max_segments=batch_max,
    )
    
    stats["passages"] = len(all_passages)
    
    if verbose and stats["skipped"] > 0:
        print(f"  Skipping {stats['skipped']} already-processed passages")

    for batch_idx, (passage_names, segments) in enumerate(batches):
        stats["batches"] += 1
        stats["segments"] += len(segments)
        
        passage_range = f"{passage_names[0]}" if len(passage_names) == 1 else f"{passage_names[0]}..{passage_names[-1]}"
        
        if verbose:
            print(f"  Batch {batch_idx + 1}/{len(batches)}: {passage_range} ({len(segments)} segments, {len(passage_names)} passages)...")

        # Check limit (on batches, not passages)
        if limit > 0 and stats["batches"] > limit:
            if verbose:
                print(f"  Limit reached ({limit} batches)")
            break

        if dry_run:
            continue

        # Call API
        try:
            api_response = call_alignment_api(
                system_prompt=system_prompt,
                segments=segments,
                translator_name=translator.capitalize(),
                translator_text=translator_text,
                api_key=api_key,
                reasoning_effort=reasoning_effort,
            )
        except Exception as e:
            print(f"  ERROR on batch {batch_idx + 1}: {e}")
            continue

        # Count results
        for key, value in api_response.items():
            if value is not None:
                stats["extracted"] += 1
            else:
                stats["null"] += 1

        # Merge and save
        aligned_json = merge_results(aligned_json, segments, translator, api_response)
        aligned_path.write_text(
            json.dumps(aligned_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Align Pali translations segment-by-segment"
    )
    parser.add_argument("collection", help="Collection (an, mn, sn)")
    parser.add_argument(
        "uid", 
        nargs="?", 
        help="Sutta ID (e.g., mn1) or range (e.g., mn1-mn10). Skips missing files."
    )
    parser.add_argument(
        "--translator",
        choices=["thanissaro", "bodhi", "both"],
        default="both",
        help="Which translator to process",
    )
    parser.add_argument(
        "--all", action="store_true", help="Process all suttas in collection"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without making API calls",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of batches to process (for testing, 0=no limit)",
    )
    parser.add_argument(
        "--batch-target",
        type=int,
        default=15,
        help="Target segments per batch (default: 15)",
    )
    parser.add_argument(
        "--batch-max",
        type=int,
        default=30,
        help="Maximum segments per batch (default: 30)",
    )
    parser.add_argument(
        "--reasoning",
        choices=["low", "medium", "high"],
        default="medium",
        help="Reasoning effort level (default: medium)",
    )

    args = parser.parse_args()

    # Determine root directory
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: OPENAI_API_KEY not found in environment or .env")
        sys.exit(1)

    # Determine translators to process
    translators = (
        ["thanissaro", "bodhi"] if args.translator == "both" else [args.translator]
    )

    # Determine suttas to process
    collection_path = root / "source" / "pali" / args.collection
    
    # Numeric-aware sorting for UIDs (mn1, mn2, ... mn10 instead of mn1, mn10, mn11, mn2)
    def uid_sort_key(uid: str):
        """Extract numeric parts for proper sorting (handles mn1, an3.1, sn12.23)."""
        match = re.match(rf"{args.collection}(\d+)(?:\.(\d+))?", uid)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2)) if match.group(2) else 0
            return (major, minor)
        return (999999, uid)  # Fallback for non-matching patterns
    
    available_uids = sorted([p.stem for p in collection_path.glob("*.json")], key=uid_sort_key)
    
    if args.all:
        uids = available_uids
    elif args.uid:
        uids = parse_uid_range(args.uid, args.collection, available_uids)
        if not uids:
            print(f"No matching suttas found for '{args.uid}' in {args.collection}")
            print(f"Available: {', '.join(available_uids[:10])}{'...' if len(available_uids) > 10 else ''}")
            sys.exit(1)
    else:
        parser.error("Either uid or --all is required")

    # Process each sutta
    for uid in uids:
        print(f"\n=== {args.collection}/{uid} ===")
        for translator in translators:
            print(f"\n[{translator.capitalize()}]")
            try:
                stats = process_sutta(
                    collection=args.collection,
                    uid=uid,
                    translator=translator,
                    root=root,
                    api_key=api_key,
                    dry_run=args.dry_run,
                    verbose=args.verbose,
                    limit=args.limit,
                    batch_target=args.batch_target,
                    batch_max=args.batch_max,
                    reasoning_effort=args.reasoning,
                )
                print(
                    f"  Batches: {stats['batches']}, "
                    f"Passages: {stats['passages']} (skipped: {stats['skipped']}), "
                    f"Segments: {stats['segments']}"
                )
                if not args.dry_run:
                    print(
                        f"  Results: {stats['extracted']} extracted, {stats['null']} null"
                    )
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")
            except Exception as e:
                print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()

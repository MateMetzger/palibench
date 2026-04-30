#!/usr/bin/env python3
"""
PaliBench Translation Benchmark Script

Translates Pali passages using OpenRouter API and saves results for evaluation.

Usage:
    python benchmark_translate.py --model "openai/gpt-4o" --reasoning none
    python benchmark_translate.py --model "openai/o3-mini" --reasoning high
    python benchmark_translate.py --model "anthropic/claude-3.7-sonnet" --reasoning medium

Required arguments:
    --model       OpenRouter model identifier (e.g., "openai/gpt-4o", "x-ai/grok-2")
    --reasoning   Reasoning effort level: xhigh, high, medium, low, minimal, or none

Optional arguments:
    --temperature    Sampling temperature (default: 0)
    --max-tokens     Max output tokens (default: 10000)
    --batch-tokens   Target tokens per batch (default: 3000)
    --no-json-mode   Disable JSON mode (for models that misbehave with it)
    --output-dir     Output directory (default: results/translations)
    --delay          Delay between batches in seconds (default: 0.5)
"""

import json
import os
import re
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

try:
    import tiktoken
except ImportError:
    print("Error: tiktoken not installed. Run: pip install tiktoken")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Error: requests not installed. Run: pip install requests")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def try_repair_json(content: str, debug: bool = False) -> str | None:
    """
    Attempt to repair common JSON issues in model output.
    Returns repaired string or None if repair not possible.
    """
    import re

    # Write debug to file for full visibility
    debug_file = open('debug.txt', 'w', encoding='utf-8') if debug else None
    def debug_log(msg):
        if debug_file:
            debug_file.write(msg + '\n')
            debug_file.flush()
        if debug:
            print(msg)

    original_len = len(content)

    # Remove markdown code blocks if present (handle newlines)
    content = re.sub(r'^```(?:json)?\s*\n?', '', content.strip())
    content = re.sub(r'\n?\s*```$', '', content)

    debug_log(f"    DEBUG repair: after markdown strip, len {original_len} -> {len(content)}")
    debug_log(f"    DEBUG repair: start = {repr(content[:100])}")

    # Remove any text before the first { or after the last }
    first_brace = content.find('{')
    last_brace = content.rfind('}')
    if first_brace == -1 or last_brace == -1:
        debug_log(f"    DEBUG repair: No braces found! first={first_brace}, last={last_brace}")
        if debug_file: debug_file.close()
        return None
    content = content[first_brace:last_brace + 1]

    debug_log(f"    DEBUG repair: extracted JSON len = {len(content)}")

    # Try parsing as-is first
    try:
        json.loads(content)
        if debug_file: debug_file.close()
        return content
    except json.JSONDecodeError as e1:
        debug_log(f"    DEBUG repair: initial parse failed at pos {e1.pos}: {e1.msg}")

    # Fix double-quoted values: ": ""value"" -> ": "value"
    # This happens when Claude wraps translations in extra quotes
    before = content
    # Simple approach: replace all occurrences of "" with " (but not at very start/end)
    # First fix the pattern ": "" at value start
    content = content.replace('": ""', '": "')
    # Then fix the pattern "" before comma, newline, or closing brace
    content = re.sub(r'""\s*,', '",', content)
    content = re.sub(r'""\s*\n', '"\n', content)
    content = re.sub(r'""\s*\}', '"}', content)

    # Fix split dialogue quotes: ?" "Sir -> ?' 'Sir (internal quotes as single quotes)
    # Claude outputs dialogue with unescaped internal quotes that break JSON
    # Use re.UNICODE flag and \w to match Unicode letters
    # Pattern: punctuation + quote + space + quote + letter (dialogue continuation)
    content = re.sub(r'([.?!,])" "(\w)', r"\1' '\2", content)
    # Also handle: word" "word (quote space quote between words)
    content = re.sub(r'(\w)" "(\w)', r"\1' '\2", content)
    # Fix: ," they -> ,' they (internal dialogue attribution like: "Yes, sir," they replied)
    content = re.sub(r'," (\w)', r",' \1", content)
    # Fix: ." He -> .' He (sentence ending inside dialogue)
    content = re.sub(r'\." (\w)', r".' \1", content)
    # Fix: . "He -> . 'He (period space quote word - different order)
    content = re.sub(r'\. "(\w)', r". '\1", content)
    # Fix: : "Gifts -> : 'Gifts (nested quotation introduced by colon)
    # IMPORTANT: Only match when preceded by a word char (inside a value), not after a quote (JSON key-value)
    content = re.sub(r'(\w): "(\w)', r"\1: '\2", content)
    # Note: removed "' -> '' fix as it breaks values starting with dialogue
    # With new prompt instructions, model should use single quotes correctly
    # Fix: ?'" "Sir -> ?' 'Sir (closing nested quote followed by opening nested quote)
    content = re.sub(r'\'\" \"(\w)', r"'' '\1", content)
    # Fix: ?" When -> ?' When (nested dialogue ends, narration continues - no new quote)
    content = re.sub(r'\?" (\w)', r"?' \1", content)
    if content != before:
        debug_log(f"    DEBUG repair: double-quote fix applied, {len(before)} -> {len(content)}")
        debug_log(f"    DEBUG repair: sample after fix: {repr(content[15:120])}")

    try:
        json.loads(content)
        if debug_file: debug_file.close()
        return content
    except json.JSONDecodeError as e2:
        debug_log(f"    DEBUG repair: still failing at pos {e2.pos}: {e2.msg}")
        ctx_start = max(0, e2.pos - 50)
        ctx_end = min(len(content), e2.pos + 50)
        debug_log(f"    DEBUG repair: context around pos {e2.pos}: {repr(content[ctx_start:ctx_end])}")
        # Also write full content to debug file for analysis
        debug_log(f"\n=== FULL REPAIRED CONTENT ===\n{content}\n=== END CONTENT ===")

    # Normalize various quote characters to standard ASCII quotes
    # Smart quotes, curly quotes, etc.
    quote_replacements = [
        ('"', '"'), ('"', '"'),  # Curly double quotes
        (''', "'"), (''', "'"),  # Curly single quotes
        ('„', '"'), ('‟', '"'),  # Other double quotes
        ('‚', "'"), ('‛', "'"),  # Other single quotes
    ]
    for old, new in quote_replacements:
        content = content.replace(old, new)

    try:
        json.loads(content)
        if debug_file: debug_file.close()
        return content
    except json.JSONDecodeError:
        pass

    # Common fix: escape unescaped newlines within strings
    # This is tricky - we need to be inside a string value
    # Simple approach: replace literal newlines with \n
    content = content.replace('\r\n', '\\n').replace('\r', '\\n')

    # Try again
    try:
        json.loads(content)
        if debug_file: debug_file.close()
        return content
    except json.JSONDecodeError:
        pass

    # Try removing trailing commas before } or ]
    content = re.sub(r',\s*}', '}', content)
    content = re.sub(r',\s*]', ']', content)

    try:
        json.loads(content)
        if debug_file: debug_file.close()
        return content
    except json.JSONDecodeError:
        pass

    if debug_file: debug_file.close()
    return None


VALID_REASONING_LEVELS = ["xhigh", "high", "medium", "low", "minimal", "none"]
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 10000
DEFAULT_BATCH_TOKENS = 3000
DEFAULT_DELAY = 0.5
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2  # Exponential backoff: 2^attempt seconds

# System prompt for translation
SYSTEM_PROMPT = """Translate the following Pali passages into English.
Return a JSON object with the same keys and the English translations as values.
Do not add text that is not present in the input.

IMPORTANT: For valid JSON, never use double quotes inside translation values.
- Use single quotes for dialogue: 'Hello,' he said.
- Use backticks for nested quotes: 'He said `hello` to me.'

Example:
Input: {"mn1:5": "Pathavī pathavīti sañjānāti.", "mn1:6": "Āpo āpoti sañjānāti."}
Output: {"mn1:5": "They perceive earth as earth.", "mn1:6": "They perceive water as water."}

Output ONLY the JSON object, no other text."""


# =============================================================================
# Tokenizer
# =============================================================================

def get_tokenizer():
    """Get tiktoken tokenizer for token counting."""
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using tiktoken."""
    return len(tokenizer.encode(text))


# =============================================================================
# Batching
# =============================================================================

def create_batches(passages: dict, tokenizer, max_tokens: int) -> list[list[str]]:
    """
    Create batches of passage IDs based on token count.

    Each batch targets max_tokens but won't exceed it (unless a single passage exceeds it).
    """
    batches = []
    current_batch = []
    current_tokens = 0

    for passage_id, passage_data in passages.items():
        pali_text = passage_data["pali"]
        passage_tokens = count_tokens(pali_text, tokenizer)

        # Check if adding this passage would exceed limit
        if current_tokens + passage_tokens > max_tokens and current_batch:
            # Close current batch and start new one
            batches.append(current_batch)
            current_batch = [passage_id]
            current_tokens = passage_tokens
        else:
            # Add to current batch
            current_batch.append(passage_id)
            current_tokens += passage_tokens

    # Don't forget the last batch
    if current_batch:
        batches.append(current_batch)

    return batches


# =============================================================================
# API Interaction
# =============================================================================

def make_api_request(
    api_key: str,
    model: str,
    passages_batch: dict,
    reasoning_effort: str,
    temperature: float,
    max_tokens: int,
    use_json_mode: bool = True
) -> dict:
    """
    Make a single API request to translate a batch of passages.

    Returns the parsed JSON response or raises an exception.
    """
    # Build the user message with passages to translate
    user_message = json.dumps(passages_batch, ensure_ascii=False, indent=2)

    # Build request payload
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    # Add JSON mode and response healing (can be disabled for models that misbehave)
    if use_json_mode:
        payload["response_format"] = {"type": "json_object"}
        payload["plugins"] = [{"id": "response-healing"}]

    # Add reasoning configuration (only if not "none" - some models reject this parameter entirely)
    if reasoning_effort != "none":
        payload["reasoning"] = {"effort": reasoning_effort}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/palibench",  # Optional but good practice
        "X-Title": "PaliBench Translation Benchmark"
    }

    response = requests.post(
        OPENROUTER_API_URL,
        headers=headers,
        json=payload,
        timeout=300  # 5 minute timeout for large batches
    )

    response.raise_for_status()
    return response.json()


def extract_translations(api_response: dict) -> tuple[dict, dict]:
    """
    Extract translations from API response.

    Returns (translations_dict, usage_info).
    Raises ValueError if response is invalid.
    """
    # Debug: print raw response structure if there's an issue
    choices = api_response.get("choices", [])
    if not choices:
        print(f"    DEBUG: Full API response: {json.dumps(api_response, indent=2)[:2000]}")
        raise ValueError("No choices in API response")

    message = choices[0].get("message", {})
    content = message.get("content", "")

    if not content:
        print(f"    DEBUG: choices[0] = {json.dumps(choices[0], indent=2)[:1000]}")
        print(f"    DEBUG: Full response keys: {list(api_response.keys())}")
        raise ValueError("Empty content in API response")

    # Parse JSON from content
    try:
        translations = json.loads(content)
    except json.JSONDecodeError as e:
        # Show the problematic area
        error_pos = e.pos if hasattr(e, 'pos') else 0
        context_start = max(0, error_pos - 100)
        context_end = min(len(content), error_pos + 100)
        print(f"    DEBUG: Content length: {len(content)} chars")
        print(f"    DEBUG: Error at position {error_pos}")
        print(f"    DEBUG: Context around error: ...{repr(content[context_start:context_end])}...")
        print(f"    DEBUG: Content end: ...{repr(content[-200:])}")

        # Try to repair common JSON issues
        repaired = try_repair_json(content, debug=True)
        if repaired:
            try:
                translations = json.loads(repaired)
                print(f"    JSON repaired successfully")
            except json.JSONDecodeError as e2:
                print(f"    DEBUG: Repair failed. Repaired content start: {repr(repaired[:300])}")
                print(f"    DEBUG: Repair failed. Repaired content end: {repr(repaired[-300:])}")
                raise ValueError(f"Invalid JSON in response (repair failed): {e2}")
        else:
            raise ValueError(f"Invalid JSON in response: {e}")

    if not isinstance(translations, dict):
        raise ValueError(f"Expected dict, got {type(translations).__name__}")

    # Extract usage info (including reasoning tokens if present)
    usage = api_response.get("usage", {})
    usage_info = {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "reasoning_tokens": usage.get("reasoning_tokens", usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0))
    }

    return translations, usage_info


def translate_batch(
    api_key: str,
    model: str,
    passages: dict,
    passage_ids: list[str],
    reasoning_effort: str,
    temperature: float,
    max_tokens: int,
    use_json_mode: bool = True,
    max_retries: int = MAX_RETRIES,
    failure_counts: dict = None
) -> tuple[dict, dict]:
    """
    Translate a batch of passages with retry logic.

    Returns (translations_dict, usage_info).
    Raises Exception after max_retries failures or if any passage fails 3 times.

    failure_counts: dict tracking per-passage failure counts (modified in place)
    """
    if failure_counts is None:
        failure_counts = {}

    # Build the batch payload (just passage_id -> pali_text)
    batch_payload = {pid: passages[pid]["pali"] for pid in passage_ids}

    all_translations = {}
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}
    remaining_ids = list(passage_ids)

    for attempt in range(max_retries):
        if not remaining_ids:
            break

        try:
            # Only request remaining passages
            current_payload = {pid: batch_payload[pid] for pid in remaining_ids}

            response = make_api_request(
                api_key=api_key,
                model=model,
                passages_batch=current_payload,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
                max_tokens=max_tokens,
                use_json_mode=use_json_mode
            )

            translations, usage_info = extract_translations(response)

            # Accumulate usage
            for key in total_usage:
                total_usage[key] += usage_info.get(key, 0)

            # Validate response: only keep keys that match requested passage IDs
            # and have sufficient content (not empty or truncated)
            MIN_TRANSLATION_LENGTH = 20  # Minimum chars for a valid translation

            valid_translations = {}
            invalid_keys = []
            too_short = []
            for key, value in translations.items():
                if key in remaining_ids:
                    if len(value.strip()) >= MIN_TRANSLATION_LENGTH:
                        valid_translations[key] = value
                    else:
                        too_short.append((key, value))
                else:
                    invalid_keys.append((key, value))

            if invalid_keys:
                print(f"    Warning: Discarding {len(invalid_keys)} invalid keys: {[k for k, v in invalid_keys[:3]]}...")

            if too_short:
                print(f"    Warning: Discarding {len(too_short)} too-short translations: {[k for k, v in too_short[:3]]}...")

            # Add valid translations to results
            all_translations.update(valid_translations)

            # Track missing passages
            missing = set(remaining_ids) - set(valid_translations.keys())

            if missing:
                for pid in missing:
                    failure_counts[pid] = failure_counts.get(pid, 0) + 1
                    if failure_counts[pid] >= max_retries:
                        raise Exception(f"Passage '{pid}' failed {max_retries} times. Stopping.")

                print(f"    Missing {len(missing)} passages, retrying... (attempt {attempt + 1}/{max_retries})")
                remaining_ids = list(missing)
                wait_time = RETRY_BACKOFF_BASE ** attempt
                time.sleep(wait_time)
            else:
                # All passages translated successfully
                remaining_ids = []

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Rate limited - wait longer
                wait_time = RETRY_BACKOFF_BASE ** (attempt + 2)
                print(f"    Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            elif e.response.status_code >= 500:
                # Server error - retry with backoff
                wait_time = RETRY_BACKOFF_BASE ** attempt
                print(f"    Server error ({e.response.status_code}). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                # Client error - don't retry
                raise

        except (ValueError, json.JSONDecodeError) as e:
            wait_time = RETRY_BACKOFF_BASE ** attempt
            print(f"    Invalid response: {e}. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait_time)

        except requests.exceptions.Timeout as e:
            wait_time = RETRY_BACKOFF_BASE ** attempt
            print(f"    Timeout. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait_time)

    # Check if we still have missing passages after all retries
    if remaining_ids:
        for pid in remaining_ids:
            failure_counts[pid] = failure_counts.get(pid, 0) + 1
            if failure_counts[pid] >= max_retries:
                raise Exception(f"Passage '{pid}' failed {max_retries} times. Stopping.")

    return all_translations, total_usage


# =============================================================================
# File I/O
# =============================================================================

def load_test_set(test_set_path: Path) -> dict:
    """Load the benchmark test set."""
    with open(test_set_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_existing_results(output_path: Path) -> dict:
    """Load existing results for resume capability."""
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_results(output_path: Path, results: dict):
    """Save results to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def sanitize_model_name(model: str) -> str:
    """Convert model name to safe filename."""
    # Replace slashes and other unsafe chars
    return re.sub(r'[/\\:*?"<>|]', '-', model)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Translate Pali passages using OpenRouter API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                        help='OpenRouter model identifier (e.g., "openai/gpt-4o")')
    parser.add_argument('--reasoning', type=str, required=True,
                        choices=VALID_REASONING_LEVELS,
                        help='Reasoning effort level')

    # Optional arguments
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                        help=f'Sampling temperature (default: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_TOKENS,
                        help=f'Max output tokens (default: {DEFAULT_MAX_TOKENS})')
    parser.add_argument('--batch-tokens', type=int, default=DEFAULT_BATCH_TOKENS,
                        help=f'Target tokens per batch (default: {DEFAULT_BATCH_TOKENS})')
    parser.add_argument('--no-json-mode', action='store_true',
                        help='Disable JSON mode (for models that misbehave with it)')
    parser.add_argument('--output-dir', type=str, default='results/translations',
                        help='Output directory (default: results/translations)')
    parser.add_argument('--delay', type=float, default=DEFAULT_DELAY,
                        help=f'Delay between batches in seconds (default: {DEFAULT_DELAY})')

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in environment or .env file")
        sys.exit(1)

    # Setup paths
    repo_root = Path(__file__).resolve().parent.parent
    test_set_path = repo_root / "data" / "test_set.json"
    output_dir = repo_root / args.output_dir

    # Create output filename from model name + reasoning level
    safe_model_name = sanitize_model_name(args.model)
    output_path = output_dir / f"{safe_model_name}_reasoning-{args.reasoning}.json"

    # Load test set
    print(f"Loading test set from {test_set_path}...")
    passages = load_test_set(test_set_path)
    total_passages = len(passages)
    print(f"  Loaded {total_passages} passages")

    # Load existing results for resume
    print(f"\nChecking for existing results at {output_path}...")
    existing_results = load_existing_results(output_path)
    completed_ids = set(existing_results.get("translations", {}).keys())
    print(f"  Found {len(completed_ids)} completed passages")

    # Filter out already completed passages
    remaining_passages = {pid: p for pid, p in passages.items() if pid not in completed_ids}
    remaining_count = len(remaining_passages)

    if remaining_count == 0:
        print("\nAll passages already translated. Nothing to do.")
        return

    print(f"  Remaining: {remaining_count} passages")

    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = get_tokenizer()

    # Create batches
    print(f"\nCreating batches (target: {args.batch_tokens} tokens per batch)...")
    batches = create_batches(remaining_passages, tokenizer, args.batch_tokens)
    print(f"  Created {len(batches)} batches")

    # Show batch size distribution
    batch_sizes = [len(b) for b in batches]
    print(f"  Passages per batch: min={min(batch_sizes)}, max={max(batch_sizes)}, avg={sum(batch_sizes)/len(batch_sizes):.1f}")

    # Initialize results structure
    if existing_results:
        results = existing_results
    else:
        results = {
            "metadata": {
                "model": args.model,
                "reasoning_effort": args.reasoning,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "batch_tokens": args.batch_tokens,
                "json_mode": not args.no_json_mode,
                "started_at": datetime.now().isoformat(),
                "test_set_passages": total_passages
            },
            "translations": {},
            "usage": {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_reasoning_tokens": 0,
                "total_tokens": 0
            }
        }

    # Process batches
    print(f"\n{'='*60}")
    print(f"Starting translation with {args.model}")
    print(f"Reasoning effort: {args.reasoning}")
    print(f"Temperature: {args.temperature}")
    print(f"{'='*60}\n")

    processed_in_session = 0
    failure_counts = {}  # Track per-passage failure counts across batches

    try:
        for batch_idx, batch_ids in enumerate(batches):
            batch_num = batch_idx + 1
            passages_so_far = len(completed_ids) + processed_in_session

            print(f"Batch {batch_num}/{len(batches)} ({len(batch_ids)} passages)")
            print(f"  Progress: {passages_so_far}/{total_passages} ({100*passages_so_far/total_passages:.1f}%)")

            # Translate batch
            try:
                translations, usage = translate_batch(
                    api_key=api_key,
                    model=args.model,
                    passages=remaining_passages,
                    passage_ids=batch_ids,
                    reasoning_effort=args.reasoning,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    use_json_mode=not args.no_json_mode,
                    failure_counts=failure_counts
                )

                # Update results
                results["translations"].update(translations)
                results["usage"]["total_prompt_tokens"] += usage["prompt_tokens"]
                results["usage"]["total_completion_tokens"] += usage["completion_tokens"]
                results["usage"]["total_reasoning_tokens"] += usage.get("reasoning_tokens", 0)
                results["usage"]["total_tokens"] += usage["total_tokens"]

                processed_in_session += len(translations)

                print(f"  Translated {len(translations)} passages")
                reasoning_info = f" (reasoning: {usage['reasoning_tokens']})" if usage.get('reasoning_tokens') else ""
                print(f"  Tokens: {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion{reasoning_info}")

                # Save after each batch for resume capability
                results["metadata"]["last_updated"] = datetime.now().isoformat()
                save_results(output_path, results)

            except Exception as e:
                print(f"\n  ERROR: {e}")
                print(f"\nStopping due to error. Progress saved to {output_path}")
                print(f"Resume by running the same command again.")
                sys.exit(1)

            # Delay between batches (except after last one)
            if batch_idx < len(batches) - 1 and args.delay > 0:
                time.sleep(args.delay)

    except KeyboardInterrupt:
        print(f"\n\nInterrupted. Progress saved to {output_path}")
        print(f"Resume by running the same command again.")
        sys.exit(1)

    # Final summary
    print(f"\n{'='*60}")
    print("TRANSLATION COMPLETE")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Total passages: {len(results['translations'])}")
    print(f"Total tokens used: {results['usage']['total_tokens']:,}")
    print(f"  Prompt: {results['usage']['total_prompt_tokens']:,}")
    print(f"  Completion: {results['usage']['total_completion_tokens']:,}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

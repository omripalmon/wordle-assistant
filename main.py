"""CLI for filtering 5-letter words with Wordle-style constraints.

Usage examples:
    python main.py --green 1=r 3=a --yellow s=0,2 e=4 --min s=1 e=1 --max e=1 s=1
    python main.py --guess crane,bgubb --guess audio,ubbbb
"""
from __future__ import annotations

import argparse
from collections import Counter

from wordle_filter import filter_words, load_nyt_wordlists, load_words


def parse_green(args: list[str]) -> dict[int, str]:
    """Parse '1=r 3=a' into {1: 'r', 3: 'a'}."""
    result = {}
    for arg in args:
        pos, letter = arg.split("=")
        result[int(pos)] = letter.lower()
    return result


def parse_yellow(args: list[str]) -> dict[str, set[int]]:
    """Parse 's=0,2 e=4' into {'s': {0, 2}, 'e': {4}}.

    Repeated letters are merged: 'a=1 a=3' -> {'a': {1, 3}}.
    """
    result: dict[str, set[int]] = {}
    for arg in args:
        letter, positions = arg.split("=")
        letter = letter.lower()
        result.setdefault(letter, set()).update(int(p) for p in positions.split(","))
    return result


def parse_min(args: list[str]) -> dict[str, int]:
    """Parse 'e=2 s=1' into {'e': 2, 's': 1}."""
    result = {}
    for arg in args:
        letter, count = arg.split("=")
        result[letter.lower()] = int(count)
    return result


def parse_max(args: list[str]) -> dict[str, int]:
    """Parse 'e=1 s=1' into {'e': 1, 's': 1}."""
    result = {}
    for arg in args:
        letter, count = arg.split("=")
        result[letter.lower()] = int(count)
    return result


def apply_yellow_minimums(
    excluded_positions: dict[str, set[int]] | None,
    min_occurrences: dict[str, int] | None,
) -> dict[str, int]:
    """Return a min_occurrences dict that guarantees every yellow letter appears
    at least once, while preserving any higher explicit minimum already set.

    Every letter in excluded_positions (yellow) is in the word by definition, so
    it needs a minimum of at least 1.  If --min already specifies a higher floor
    for that letter, the higher value wins.
    """
    result: dict[str, int] = dict(min_occurrences or {})
    for letter in (excluded_positions or {}):
        result[letter] = max(result.get(letter, 0), 1)
    return result or {}


def parse_guesses(
    guess_args: list[str],
) -> tuple[dict[int, str], dict[str, set[int]], dict[str, int], dict[str, int]]:
    """Parse a list of ``WORD,RESPONSE`` guess strings into Wordle filter constraints.

    Each entry is a 5-letter word paired with a 5-character response string where:
      - ``g`` = green  (correct letter, correct position)
      - ``y`` = yellow (correct letter, wrong position)
      - ``b`` = black  (letter not in the word, or over the allowed count)

    Multiple guesses are merged together correctly, handling repeated letters
    within and across guesses:

    - Green letters fix their position in ``known_positions``.
    - Yellow letters are added to ``excluded_positions`` (wrong positions) and
      raise the ``min_occurrences`` floor for that letter.
    - Black letters set ``max_occurrences`` to the number of green+yellow
      occurrences seen for that letter in the *same guess* (which may be > 0
      when a letter appears green/yellow elsewhere in the same guess).
    - ``min_occurrences`` is taken as the per-letter maximum across all guesses.

    Args:
        guess_args: List of strings like ``"crane,bgybb"`` or ``"audio,ybbbb"``.

    Returns:
        4-tuple of ``(known_positions, excluded_positions, min_occurrences,
        max_occurrences)`` ready to pass to ``filter_words``.

    Raises:
        ValueError: If any entry is malformed (wrong length, invalid response
            characters, etc.).

    Example::

        >>> parse_guesses(["crane,bgybb", "audio,ybbbb"])
        ({2: 'a'}, {'r': {1}}, {'a': 1, 'r': 1}, {'c': 0, 'n': 0, 'e': 0, 'd': 0, 'i': 0, 'o': 0})
    """
    known_positions: dict[int, str] = {}
    excluded_positions: dict[str, set[int]] = {}
    min_occurrences: dict[str, int] = {}
    max_occurrences: dict[str, int] = {}

    for entry in guess_args:
        if "," not in entry:
            raise ValueError(
                f"Invalid guess format {entry!r}: expected WORD,RESPONSE (e.g. crane,bgubb)"
            )
        word_part, response_part = entry.split(",", 1)
        word = word_part.strip().lower()
        response = response_part.strip().lower()

        if len(word) != 5 or not word.isalpha():
            raise ValueError(
                f"Invalid guess word {word!r}: must be exactly 5 alphabetic characters"
            )
        if len(response) != 5 or not all(c in "gyb" for c in response):
            raise ValueError(
                f"Invalid response {response!r}: must be 5 characters each being g/y/b"
            )

        # Count green+yellow occurrences per letter in this guess
        # (used to derive max_occurrences when a black tile appears)
        confirmed_counts: Counter[str] = Counter()
        for i, (ch, tile) in enumerate(zip(word, response)):
            if tile in ("g", "y"):
                confirmed_counts[ch] += 1

        for i, (ch, tile) in enumerate(zip(word, response)):
            if tile == "g":
                known_positions[i] = ch
            elif tile == "y":
                excluded_positions.setdefault(ch, set()).add(i)
                # This guess confirms at least confirmed_counts[ch] copies
                min_occurrences[ch] = max(
                    min_occurrences.get(ch, 0), confirmed_counts[ch]
                )
            else:  # black
                # The letter appears exactly confirmed_counts[ch] times total
                # (could be 0 if it never appeared green/yellow in this guess).
                # Only set/tighten max if it wouldn't contradict an existing lower max.
                cap = confirmed_counts[ch]
                if ch not in max_occurrences or cap < max_occurrences[ch]:
                    max_occurrences[ch] = cap

        # Ensure every yellow letter has at least min=1 even if only blacks
        # were processed first for the same letter in an earlier guess.
        for ch in excluded_positions:
            min_occurrences[ch] = max(min_occurrences.get(ch, 0), 1)

    return known_positions, excluded_positions, min_occurrences, max_occurrences


def merge_constraints(
    known_positions: dict[int, str],
    excluded_positions: dict[str, set[int]],
    min_occurrences: dict[str, int],
    max_occurrences: dict[str, int],
    extra_known: dict[int, str] | None,
    extra_excluded: dict[str, set[int]] | None,
    extra_min: dict[str, int] | None,
    extra_max: dict[str, int] | None,
) -> tuple[dict[int, str], dict[str, set[int]], dict[str, int], dict[str, int]]:
    """Merge two sets of constraints, taking the stricter value for each key."""
    merged_known = {**known_positions, **(extra_known or {})}

    merged_excluded: dict[str, set[int]] = {k: set(v) for k, v in excluded_positions.items()}
    for letter, positions in (extra_excluded or {}).items():
        merged_excluded.setdefault(letter, set()).update(positions)

    merged_min = dict(min_occurrences)
    for letter, val in (extra_min or {}).items():
        merged_min[letter] = max(merged_min.get(letter, 0), val)

    merged_max = dict(max_occurrences)
    for letter, val in (extra_max or {}).items():
        if letter not in merged_max or val < merged_max[letter]:
            merged_max[letter] = val

    return merged_known, merged_excluded, merged_min, merged_max


def main():
    parser = argparse.ArgumentParser(description="Filter 5-letter words (Wordle helper)")
    parser.add_argument(
        "--green", nargs="*", default=[],
        help="Known positions: POS=LETTER (e.g. 1=r 3=a)",
    )
    parser.add_argument(
        "--yellow", nargs="*", default=[],
        help="Letter present but forbidden at positions: LETTER=POS,POS (e.g. s=0,2)",
    )
    parser.add_argument(
        "--min", nargs="*", default=[],
        help="Minimum occurrences: LETTER=COUNT (e.g. e=2 means at least two e's)",
    )
    parser.add_argument(
        "--max", nargs="*", default=[],
        help="Maximum occurrences: LETTER=COUNT (e.g. e=1)",
    )
    parser.add_argument(
        "--guess", action="append", default=[], metavar="WORD,RESPONSE",
        help=(
            "A past guess and its result, e.g. crane,bgybb  "
            "Response is 5 chars: g=green, y=yellow, b=black. "
            "Repeat --guess for multiple past guesses. "
            "Can be combined with --green/--yellow/--min/--max."
        ),
    )
    parser.add_argument(
        "--answers", default=None,
        help="Path to possible answers file (default: bundled wordle-answers.txt)",
    )
    parser.add_argument(
        "--wordlist", default=None,
        help="Path to a single word list file (legacy; uses it as the answers list)",
    )
    args = parser.parse_args()

    # Build constraints from --guess entries first, then merge manual flags on top
    if args.guess:
        kp, ep, mn, mx = parse_guesses(args.guess)
    else:
        kp, ep, mn, mx = {}, {}, {}, {}

    extra_kp = parse_green(args.green) if args.green else None
    extra_ep = parse_yellow(args.yellow) if args.yellow else None
    extra_mn = apply_yellow_minimums(extra_ep, parse_min(args.min) if args.min else None) or None
    extra_mx = parse_max(args.max) if args.max else None

    known_positions, excluded_positions, min_occurrences, max_occurrences = merge_constraints(
        kp, ep, mn, mx, extra_kp, extra_ep, extra_mn, extra_mx
    )

    if args.wordlist:
        words = load_words(args.wordlist)
    else:
        answers_list, _ = load_nyt_wordlists(args.answers)
        words = answers_list
    results = filter_words(
        words,
        known_positions=known_positions or None,
        excluded_positions=excluded_positions or None,
        min_occurrences=min_occurrences or None,
        max_occurrences=max_occurrences or None,
    )

    for word in results:
        print(word)
    print(f"\n{len(results)} words found")


if __name__ == "__main__":
    main()

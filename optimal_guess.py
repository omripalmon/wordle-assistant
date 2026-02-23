"""Find the optimal first Wordle guess by entropy of the pattern distribution.

For each candidate guess, computes the distribution of Wordle feedback patterns
across all possible secret words, then calculates the Shannon entropy of that
distribution.  The guess that maximises entropy provides the most information
(splits the word space most evenly).

Usage:
    python optimal_guess.py
    python optimal_guess.py --top 20
    python optimal_guess.py --wordlist /path/to/words.txt --workers 4
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from multiprocessing import Pool, cpu_count

from wordle_filter import filter_words, load_words

# Pattern tile values (base-3 digits)
GREEN = 2
YELLOW = 1
GRAY = 0


def compute_pattern(guess: str, secret: str) -> int:
    """Compute the Wordle feedback pattern for a guess against a secret word.

    Returns an integer in [0, 242] encoding the 5-tile pattern in base 3,
    where GREEN=2, YELLOW=1, GRAY=0, most-significant trit first.

    The matching uses Wordle's official two-pass rules:
    1. First pass: mark exact positional matches as GREEN.
    2. Second pass: for remaining positions, mark YELLOW if the guessed
       letter appears among unmatched secret letters (respecting frequency),
       otherwise GRAY.
    """
    result = [GRAY, GRAY, GRAY, GRAY, GRAY]
    secret_counts: dict[str, int] = {}

    # Build letter frequency for the secret
    for ch in secret:
        secret_counts[ch] = secret_counts.get(ch, 0) + 1

    # Pass 1: greens (exact matches)
    for i in range(5):
        if guess[i] == secret[i]:
            result[i] = GREEN
            secret_counts[guess[i]] -= 1

    # Pass 2: yellows (present but wrong position)
    for i in range(5):
        if result[i] == GREEN:
            continue
        ch = guess[i]
        if secret_counts.get(ch, 0) > 0:
            result[i] = YELLOW
            secret_counts[ch] -= 1

    # Encode as base-3 integer
    val = 0
    for r in result:
        val = val * 3 + r
    return val


def pattern_distribution(guess: str, word_list: list[str]) -> Counter[int]:
    """Compute the distribution of patterns for a guess against all secrets."""
    return Counter(compute_pattern(guess, secret) for secret in word_list)


def entropy(distribution: Counter[int], total: int) -> float:
    """Compute Shannon entropy (in bits) of a pattern count distribution.

    Args:
        distribution: Counter mapping pattern IDs to counts.
        total: Total number of words (sum of counts).

    Returns:
        Entropy H = -sum(p * log2(p)) where p = count/total for each bucket.
    """
    if total == 0:
        return 0.0
    h = 0.0
    for count in distribution.values():
        if count > 0:
            p = count / total
            h -= p * math.log2(p)
    return h


def score_guess(guess: str, word_list: list[str]) -> float:
    """Compute the entropy score for a single guess against the full word list."""
    dist = pattern_distribution(guess, word_list)
    return entropy(dist, len(word_list))


# ---------------------------------------------------------------------------
# Multiprocessing helpers
# ---------------------------------------------------------------------------
# Module-level variable set by worker initialiser so each child process holds
# its own copy of the word list without re-serialising it per task.
_word_list: list[str] = []


def _init_worker(word_list: list[str]) -> None:
    """Initialise the word list in each worker process."""
    global _word_list
    _word_list = word_list


def _score_word(guess: str) -> tuple[str, float]:
    """Score a single guess word (multiprocessing target)."""
    return (guess, score_guess(guess, _word_list))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def valid_word_bonus(n: int) -> float:
    """Compute the bonus awarded to a guess that is itself a valid answer.

    For a candidate pool of size *n*, the bonus is ``(1/n) * log2(n)``,
    which reflects the residual information gained if the guess happens to
    be the secret word.  Returns 0 when *n* <= 1 (log2 undefined / zero).
    """
    if n <= 1:
        return 0.0
    return math.log2(n) / n


def find_optimal_guesses(
    word_list: list[str],
    top_n: int = 10,
    workers: int | None = None,
    valid_words: set[str] | None = None,
) -> list[tuple[str, float, float]]:
    """Find the guesses with highest entropy (most informative first guesses).

    Args:
        word_list: List of unique 5-letter words to score as candidates.
        top_n: Number of top results to return.
        workers: Number of parallel processes (default: CPU count).
        valid_words: Set of words that are valid answers.  Candidates in this
            set receive a bonus of ``(1/N)*log2(N)`` (where N is the size of
            the candidate pool) added to their adjusted score used for
            ranking.  If ``None``, all words are treated as valid.

    Returns:
        List of (word, raw_entropy, adjusted_entropy) tuples sorted by
        adjusted entropy descending.
    """
    if workers is None:
        workers = cpu_count()

    total = len(word_list)
    if total == 0:
        return []

    if valid_words is None:
        valid_words = set(word_list)

    bonus = valid_word_bonus(total)
    scored: list[tuple[str, float, float]] = []

    with Pool(processes=workers, initializer=_init_worker, initargs=(word_list,)) as pool:
        results = pool.imap_unordered(_score_word, word_list, chunksize=64)
        for i, (word, raw_score) in enumerate(results, 1):
            word_bonus = bonus if word in valid_words else 0.0
            scored.append((word, raw_score, raw_score + word_bonus))
            if i % 500 == 0 or i == total:
                print(
                    f"\r  Progress: {i}/{total} words scored "
                    f"({100 * i / total:.1f}%)",
                    end="",
                    flush=True,
                    file=sys.stderr,
                )

    print(file=sys.stderr)  # newline after progress bar

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_n]


def best_guess_from_constraints(
    known_positions: dict[int, str] | None = None,
    excluded_positions: dict[str, set[int]] | None = None,
    min_occurrences: dict[str, int] | None = None,
    max_occurrences: dict[str, int] | None = None,
    wordlist: str = "/usr/share/dict/words",
) -> tuple[str, float] | None:
    """Given Wordle constraints, return the highest-entropy guess from the remaining candidates.

    Filters the word list using the supplied constraints (identical semantics to
    ``filter_words``), then scores every surviving word by the Shannon entropy of
    its Wordle feedback pattern distribution against that same candidate set.

    Args:
        known_positions: Green letters — map of position (0-4) to letter.
            e.g. ``{1: 'r', 3: 'a'}``
        excluded_positions: Yellow letters — map of letter to forbidden positions.
            e.g. ``{'s': {0, 2}}`` means 's' cannot be at index 0 or 2.
        min_occurrences: Minimum required count per letter.
            e.g. ``{'e': 2}`` means 'e' must appear at least twice.
        max_occurrences: Maximum allowed count per letter.
            e.g. ``{'e': 1}`` means 'e' can appear at most once.
        wordlist: Path to the dictionary file (default: ``/usr/share/dict/words``).

    Returns:
        ``(word, entropy_bits)`` for the best guess, or ``None`` if no words survive
        the constraints.

    Example::

        >>> best_guess_from_constraints(
        ...     known_positions={2: 'a'},
        ...     excluded_positions={'r': {0}},
        ...     min_occurrences={'r': 1},
        ...     max_occurrences={'e': 1},
        ... )
        ('crane', 3.17)   # illustrative — actual value depends on word list
    """
    all_words = load_words(wordlist)
    candidates = filter_words(
        all_words,
        known_positions=known_positions,
        excluded_positions=excluded_positions,
        min_occurrences=min_occurrences,
        max_occurrences=max_occurrences,
    )

    if not candidates:
        return None

    best_word, best_score = max(
        ((word, score_guess(word, candidates)) for word in candidates),
        key=lambda t: t[1],
    )
    return best_word, best_score


def main() -> None:
    from main import (
        apply_yellow_minimums,
        merge_constraints,
        parse_green,
        parse_guesses,
        parse_max,
        parse_min,
        parse_yellow,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Find the optimal Wordle guess by pattern entropy. "
            "Without constraints, scores all words (first-guess mode). "
            "With constraints, filters to legal candidates first then finds the best guess."
        ),
    )
    parser.add_argument(
        "--wordlist",
        default="/usr/share/dict/words",
        help="Path to word list file (default: /usr/share/dict/words)",
    )
    # High-level guess interface
    parser.add_argument(
        "--guess", action="append", default=[], metavar="WORD,RESPONSE",
        help=(
            "A past guess and Wordle response, e.g. --guess crane,bgybb  "
            "Response is 5 chars: g=green, y=yellow, b=black. "
            "Repeat for multiple past guesses: --guess crane,bgybb --guess audio,ybbgb. "
            "Can be combined with --green/--yellow/--min/--max."
        ),
    )
    # Manual constraint arguments — same syntax as main.py
    parser.add_argument(
        "--green", nargs="*", default=[],
        help="Known positions: POS=LETTER (e.g. 1=r 3=a)",
    )
    parser.add_argument(
        "--yellow", nargs="*", default=[],
        help="Letter forbidden at positions: LETTER=POS,POS (e.g. s=0,2)",
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
        "--image", default=None, metavar="PATH",
        help=(
            "Path to a Wordle screenshot. Tile colours are auto-detected. "
            "Use --image-words to supply the guessed words shown in the image. "
            "Can be combined with --guess / --green / --yellow / --min / --max."
        ),
    )
    parser.add_argument(
        "--image-words", nargs="*", default=None, metavar="WORD",
        help=(
            "Words guessed in the --image screenshot, in order "
            "(e.g. --image-words crane audio). Required when using --image."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top guesses to display (default: 10)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="(Unconstrained mode) Number of parallel worker processes (default: all CPUs)",
    )
    args = parser.parse_args()

    # Build constraints from --image screenshot first
    image_guesses: list[str] = []
    if args.image:
        from wordle_image import parse_wordle_image
        try:
            parsed = parse_wordle_image(args.image, words=args.image_words)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error reading image: {exc}", file=sys.stderr)
            sys.exit(1)
        print("Guesses read from image:", file=sys.stderr)
        for word, response in parsed:
            entry = f"{word},{response}"
            print(f"  {entry}", file=sys.stderr)
            image_guesses.append(entry)

    # Build constraints from --guess entries + image guesses, then merge manual flags on top
    all_guess_args = image_guesses + args.guess
    if all_guess_args:
        kp, ep, mn, mx = parse_guesses(all_guess_args)
    else:
        kp, ep, mn, mx = {}, {}, {}, {}

    extra_kp = parse_green(args.green) if args.green else None
    extra_ep = parse_yellow(args.yellow) if args.yellow else None
    extra_mn = apply_yellow_minimums(extra_ep, parse_min(args.min) if args.min else None) or None
    extra_mx = parse_max(args.max) if args.max else None

    known_positions, excluded_positions, min_occurrences, max_occurrences = merge_constraints(
        kp, ep, mn, mx, extra_kp, extra_ep, extra_mn, extra_mx
    )

    # Normalise empties to None for the constrained-mode check below
    known_positions = known_positions or None
    excluded_positions = excluded_positions or None
    min_occurrences = min_occurrences or None
    max_occurrences = max_occurrences or None
    constrained = any(x is not None for x in (known_positions, excluded_positions, min_occurrences or None, max_occurrences))

    raw_words = load_words(args.wordlist)

    if constrained:
        # ------------------------------------------------------------------ #
        # Constrained mode: filter first, then find single best guess        #
        # ------------------------------------------------------------------ #
        candidates = filter_words(
            raw_words,
            known_positions=known_positions,
            excluded_positions=excluded_positions,
            min_occurrences=min_occurrences,
            max_occurrences=max_occurrences,
        )
        if not candidates:
            print("No words match the given constraints.", file=sys.stderr)
            sys.exit(1)

        for word in candidates:
            print(word)
        print(f"\n{len(candidates)} words found\n")

        if len(candidates) == 1:
            print(f"Only one possible word remaining: {candidates[0]}")
            return

        candidate_set = set(candidates)
        bonus = valid_word_bonus(len(candidates))
        scored = sorted(
            (
                (word, raw := score_guess(word, candidates), raw + (bonus if word in candidate_set else 0.0))
                for word in raw_words
            ),
            key=lambda t: t[2],
            reverse=True,
        )
        top_n = min(args.top, len(scored))
        max_entropy = math.log2(len(candidates))
        print(f"\nTop {top_n} guesses (highest entropy = most informative):\n")
        print(f"{'Rank':<6} {'Word':<10} {'Entropy (bits)':<16} {'Adjusted':<16} {'% of max':<12} {'Note'}")
        print("-" * 68)
        for rank, (word, raw_score, adj_score) in enumerate(scored[:top_n], 1):
            pct = 100 * raw_score / max_entropy if max_entropy > 0 else 0
            note = "" if word in candidate_set else "*"
            print(f"{rank:<6} {word:<10} {raw_score:<16.4f} {adj_score:<16.4f} {pct:<12.1f} {note}")

        best_overall = scored[0] if scored else None
        best_valid = next((t for t in scored if t[0] in candidate_set), None)

        print()
        if best_overall:
            print(f"Best overall (adjusted score):  {best_overall[0]}  "
                  f"(raw={best_overall[1]:.4f}, adjusted={best_overall[2]:.4f})")
        if best_valid:
            print(f"Best valid-word candidate:      {best_valid[0]}  "
                  f"(raw={best_valid[1]:.4f}, adjusted={best_valid[2]:.4f})")

        print(f"\nRemaining candidates: {len(candidates)}")
        print(f"* = guess is not itself a valid candidate under current constraints")
        print(f"Valid candidates receive a +{bonus:.4f} bit bonus ((1/N)*log2(N), N={len(candidates)}) in adjusted score.")
    else:
        # ------------------------------------------------------------------ #
        # Unconstrained mode: score all words (original first-guess finder)  #
        # ------------------------------------------------------------------ #
        word_list = sorted(set(raw_words))
        print(f"Loaded {len(word_list)} unique words.", file=sys.stderr)

        max_entropy = math.log2(len(word_list))
        print(f"Maximum possible entropy: {max_entropy:.4f} bits", file=sys.stderr)
        print(
            f"Scoring all words using {args.workers or cpu_count()} workers...",
            file=sys.stderr,
        )

        valid_words_set = set(word_list)
        results = find_optimal_guesses(
            word_list,
            top_n=args.top,
            workers=args.workers,
            valid_words=valid_words_set,
        )

        print(f"\nTop {len(results)} guesses (highest entropy = most informative):\n")
        print(f"{'Rank':<6} {'Word':<10} {'Entropy (bits)':<16} {'Adjusted':<16} {'% of max':<12} {'Valid?'}")
        print("-" * 68)
        for rank, (word, raw_score, adj_score) in enumerate(results, 1):
            pct = 100 * raw_score / max_entropy if max_entropy > 0 else 0
            valid_mark = "yes" if word in valid_words_set else ""
            print(f"{rank:<6} {word:<10} {raw_score:<16.4f} {adj_score:<16.4f} {pct:<12.1f} {valid_mark}")

        # Best overall (by adjusted score) and best valid-word candidate
        best_overall = results[0] if results else None
        best_valid = next(
            ((w, r, a) for w, r, a in results if w in valid_words_set), None
        )
        if best_valid is None:
            # Not in top_n — search full scored list would require re-running;
            # in unconstrained mode all words are valid so this won't happen.
            pass

        print()
        if best_overall:
            print(f"Best overall (adjusted score):  {best_overall[0]}  "
                  f"(raw={best_overall[1]:.4f}, adjusted={best_overall[2]:.4f})")
        if best_valid:
            print(f"Best valid-word candidate:      {best_valid[0]}  "
                  f"(raw={best_valid[1]:.4f}, adjusted={best_valid[2]:.4f})")

        n = len(word_list)
        bonus_display = valid_word_bonus(n)
        print(
            f"\nNote: Ranking maximises entropy of the pattern distribution.\n"
            f"Valid-word candidates receive a +{bonus_display:.4f} bit bonus ((1/N)*log2(N), N={n}) in adjusted score.\n"
            f"In unconstrained mode all words are valid, so adjusted = raw + bonus."
        )


if __name__ == "__main__":
    main()

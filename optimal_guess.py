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

from wordle_filter import filter_words, load_nyt_wordlists, load_words

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
# its own copy of the answer pool without re-serialising it per task.
_answer_pool: list[str] = []


def _init_worker(answer_pool: list[str]) -> None:
    """Initialise the answer pool in each worker process."""
    global _answer_pool
    _answer_pool = answer_pool


def _score_word(guess: str) -> tuple[str, float]:
    """Score a single guess word (multiprocessing target)."""
    return (guess, score_guess(guess, _answer_pool))


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
    guess_candidates: list[str],
    answer_pool: list[str] | None = None,
    top_n: int = 10,
    workers: int | None = None,
    valid_answers: set[str] | None = None,
) -> list[tuple[str, float, float]]:
    """Find the guesses with highest entropy (most informative first guesses).

    Args:
        guess_candidates: All words to evaluate as potential guesses.
        answer_pool: The set of possible answer words used to compute entropy.
            Each guess is scored by the entropy of its pattern distribution
            across these words.  If ``None``, defaults to ``guess_candidates``
            (backward-compatible single-list mode).
        top_n: Number of top results to return.
        workers: Number of parallel processes (default: CPU count).
        valid_answers: Set of words that are valid answers.  Words in this
            set receive a bonus of ``(1/N)*log2(N)`` (where N = len(answer_pool))
            added to their adjusted score.  If ``None``, defaults to
            ``set(answer_pool)``.

    Returns:
        List of (word, raw_entropy, adjusted_entropy) tuples sorted by
        adjusted entropy descending.
    """
    if answer_pool is None:
        answer_pool = guess_candidates

    if workers is None:
        workers = cpu_count()

    n_answers = len(answer_pool)
    n_guesses = len(guess_candidates)
    if n_guesses == 0 or n_answers == 0:
        return []

    if valid_answers is None:
        valid_answers = set(answer_pool)

    bonus = valid_word_bonus(n_answers)
    scored: list[tuple[str, float, float]] = []

    with Pool(processes=workers, initializer=_init_worker, initargs=(answer_pool,)) as pool:
        results = pool.imap_unordered(_score_word, guess_candidates, chunksize=64)
        for i, (word, raw_score) in enumerate(results, 1):
            word_bonus = bonus if word in valid_answers else 0.0
            scored.append((word, raw_score, raw_score + word_bonus))
            if i % 500 == 0 or i == n_guesses:
                print(
                    f"\r  Progress: {i}/{n_guesses} words scored "
                    f"({100 * i / n_guesses:.1f}%)",
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
    answers_path: str | None = None,
    guesses_path: str | None = None,
) -> tuple[str, float] | None:
    """Given Wordle constraints, return the highest-entropy guess from the remaining candidates.

    Filters the possible *answers* using the supplied constraints, then scores
    every valid guess (answers + allowed guesses) by the Shannon entropy of its
    Wordle feedback pattern distribution against the filtered answer candidates.

    Args:
        known_positions: Green letters — map of position (0-4) to letter.
            e.g. ``{1: 'r', 3: 'a'}``
        excluded_positions: Yellow letters — map of letter to forbidden positions.
            e.g. ``{'s': {0, 2}}`` means 's' cannot be at index 0 or 2.
        min_occurrences: Minimum required count per letter.
            e.g. ``{'e': 2}`` means 'e' must appear at least twice.
        max_occurrences: Maximum allowed count per letter.
            e.g. ``{'e': 1}`` means 'e' can appear at most once.
        answers_path: Path to the answers file (default: bundled wordle-answers.txt).
        guesses_path: Path to the allowed-guesses file (default: bundled wordle-allowed-guesses.txt).

    Returns:
        ``(word, entropy_bits)`` for the best guess, or ``None`` if no words survive
        the constraints.
    """
    answers, allowed_guesses = load_nyt_wordlists(answers_path, guesses_path)
    all_guesses = answers + allowed_guesses

    candidates = filter_words(
        answers,
        known_positions=known_positions,
        excluded_positions=excluded_positions,
        min_occurrences=min_occurrences,
        max_occurrences=max_occurrences,
    )

    if not candidates:
        return None

    best_word, best_score = max(
        ((word, score_guess(word, candidates)) for word in all_guesses),
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
        "--answers",
        default=None,
        help="Path to possible answers file (default: bundled wordle-answers.txt)",
    )
    parser.add_argument(
        "--guesses",
        default=None,
        help="Path to allowed guesses file (default: bundled wordle-allowed-guesses.txt)",
    )
    parser.add_argument(
        "--wordlist",
        default=None,
        help="Path to a single word list file (uses it as both answers and guesses; "
             "overrides --answers/--guesses)",
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

    if args.wordlist:
        # Legacy single-file mode: use the file as both answers and guesses.
        raw = load_words(args.wordlist)
        answers = raw
        all_guesses = raw
    else:
        answers, allowed_guesses = load_nyt_wordlists(args.answers, args.guesses)
        all_guesses = answers + allowed_guesses

    if constrained:
        # ------------------------------------------------------------------ #
        # Constrained mode: filter answers, score all valid guesses          #
        # ------------------------------------------------------------------ #
        # Only possible answers are listed as candidates; all valid guesses
        # are scored against those candidates.
        candidates = filter_words(
            answers,
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
                for word in all_guesses
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
        print(f"* = guess is not itself a valid answer candidate")
        print(f"Valid answer candidates receive a +{bonus:.4f} bit bonus ((1/N)*log2(N), N={len(candidates)}) in adjusted score.")
    else:
        # ------------------------------------------------------------------ #
        # Unconstrained mode: score all guesses, entropy vs all answers      #
        # ------------------------------------------------------------------ #
        answers_deduped = sorted(set(answers))
        all_guesses_deduped = sorted(set(all_guesses))
        print(f"Answers pool: {len(answers_deduped)} words.", file=sys.stderr)
        print(f"Guess candidates: {len(all_guesses_deduped)} words.", file=sys.stderr)

        max_entropy = math.log2(len(answers_deduped))
        print(f"Maximum possible entropy: {max_entropy:.4f} bits", file=sys.stderr)
        print(
            f"Scoring all {len(all_guesses_deduped)} guess candidates using "
            f"{args.workers or cpu_count()} workers...",
            file=sys.stderr,
        )

        answers_set = set(answers_deduped)
        results = find_optimal_guesses(
            all_guesses_deduped,
            answer_pool=answers_deduped,
            top_n=args.top,
            workers=args.workers,
            valid_answers=answers_set,
        )

        print(f"\nTop {len(results)} guesses (highest entropy = most informative):\n")
        print(f"{'Rank':<6} {'Word':<10} {'Entropy (bits)':<16} {'Adjusted':<16} {'% of max':<12} {'Answer?'}")
        print("-" * 68)
        for rank, (word, raw_score, adj_score) in enumerate(results, 1):
            pct = 100 * raw_score / max_entropy if max_entropy > 0 else 0
            answer_mark = "yes" if word in answers_set else ""
            print(f"{rank:<6} {word:<10} {raw_score:<16.4f} {adj_score:<16.4f} {pct:<12.1f} {answer_mark}")

        best_overall = results[0] if results else None
        best_valid = next(
            ((w, r, a) for w, r, a in results if w in answers_set), None
        )

        print()
        if best_overall:
            print(f"Best overall (adjusted score):  {best_overall[0]}  "
                  f"(raw={best_overall[1]:.4f}, adjusted={best_overall[2]:.4f})")
        if best_valid:
            print(f"Best answer-word candidate:     {best_valid[0]}  "
                  f"(raw={best_valid[1]:.4f}, adjusted={best_valid[2]:.4f})")

        n = len(answers_deduped)
        bonus_display = valid_word_bonus(n)
        print(
            f"\nNote: Entropy computed over {n} possible answers.\n"
            f"Answer words receive a +{bonus_display:.4f} bit bonus ((1/N)*log2(N), N={n}) in adjusted score."
        )


if __name__ == "__main__":
    main()

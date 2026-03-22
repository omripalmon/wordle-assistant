"""Wordle-style word filter for 5-letter English words."""
from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).parent
_NYT_ANSWERS = _HERE / "wordle-answers.txt"
_NYT_GUESSES = _HERE / "wordle-allowed-guesses.txt"


def load_words(path: str = "/usr/share/dict/words") -> list[str]:
    """Load 5-letter lowercase alpha-only words from a dictionary file."""
    text = Path(path).read_text()
    return [
        w.lower()
        for w in text.splitlines()
        if len(w) == 5 and w.isalpha()
    ]


def load_nyt_wordlists(
    answers_path: str | None = None,
    guesses_path: str | None = None,
) -> tuple[list[str], list[str]]:
    """Load the NYT Wordle word lists.

    Returns:
        ``(answers, allowed_guesses)`` where *answers* are the ~14 855 valid
        Wordle words (answers + guesses combined; NYT now selects answers
        server-side) and *allowed_guesses* is an empty list.
    """
    ap = Path(answers_path) if answers_path else _NYT_ANSWERS
    gp = Path(guesses_path) if guesses_path else _NYT_GUESSES

    def _read(p: Path) -> list[str]:
        return [
            w.strip().lower()
            for w in p.read_text().splitlines()
            if len(w.strip()) == 5 and w.strip().isalpha()
        ]

    return _read(ap), _read(gp)


def filter_words(
    words: list[str],
    known_positions: dict[int, str] | None = None,
    excluded_positions: dict[str, set[int]] | None = None,
    min_occurrences: dict[str, int] | None = None,
    max_occurrences: dict[str, int] | None = None,
) -> list[str]:
    """Filter 5-letter words based on positional constraints.

    Args:
        words: List of 5-letter words to filter.
        known_positions: Map of position (0-4) to letter that must be there.
            e.g. {1: 'r'} means position 1 must be 'r'.
        excluded_positions: Map of letter to positions where it cannot appear.
            e.g. {'s': {3, 4}} means 's' cannot be at index 3 or 4.
            Note: does NOT imply the letter must be present — use min_occurrences
            to require a minimum count.
        min_occurrences: Map of letter to minimum required count in the word.
            e.g. {'e': 2} means 'e' must appear at least twice.
            Use this to express that a yellow letter must be present (min 1),
            or that a double-yellow implies at least 2 copies, etc.
        max_occurrences: Map of letter to maximum allowed count in the word.
            e.g. {'e': 1} means 'e' can appear at most once.

    Returns:
        List of words matching all constraints.
    """
    known_positions = known_positions or {}
    excluded_positions = excluded_positions or {}
    min_occurrences = min_occurrences or {}
    max_occurrences = max_occurrences or {}

    results = []
    for word in words:
        if len(word) != 5:
            continue
        w = word.lower()

        # Check known positions (green)
        if any(w[pos] != letter for pos, letter in known_positions.items()):
            continue

        # Check excluded positions (yellow): letter must NOT be at those spots
        if any(
            w[pos] == letter
            for letter, positions in excluded_positions.items()
            for pos in positions
        ):
            continue

        # Check minimum occurrences
        if any(w.count(letter) < minimum for letter, minimum in min_occurrences.items()):
            continue

        # Check maximum occurrences
        if any(w.count(letter) > limit for letter, limit in max_occurrences.items()):
            continue

        results.append(word)
    return results

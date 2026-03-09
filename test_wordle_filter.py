"""Tests for wordle_filter.py."""
import pytest
from wordle_filter import load_words, filter_words


# -- load_words tests --

def test_load_words_filters_to_five_letters(tmp_path):
    wordfile = tmp_path / "words.txt"
    wordfile.write_text("apple\nhi\nbanana\ngrape\nABCDE\n")
    words = load_words(str(wordfile))
    assert words == ["apple", "grape", "abcde"]


def test_load_words_excludes_non_alpha(tmp_path):
    wordfile = tmp_path / "words.txt"
    wordfile.write_text("he11o\nwo-ld\nhello\n")
    words = load_words(str(wordfile))
    assert words == ["hello"]


def test_load_words_empty_file(tmp_path):
    wordfile = tmp_path / "words.txt"
    wordfile.write_text("")
    assert load_words(str(wordfile)) == []


# -- filter_words: no constraints --

def test_filter_no_constraints():
    words = ["apple", "grape", "lemon"]
    assert filter_words(words) == ["apple", "grape", "lemon"]


def test_filter_skips_non_five_letter():
    words = ["hi", "apple", "banana"]
    assert filter_words(words) == ["apple"]


# -- filter_words: known_positions (green) --

def test_filter_known_positions_single():
    words = ["apple", "amber", "about"]
    result = filter_words(words, known_positions={0: "a"})
    assert result == ["apple", "amber", "about"]


def test_filter_known_positions_excludes():
    words = ["apple", "grape", "plane"]
    result = filter_words(words, known_positions={0: "a"})
    assert result == ["apple"]


def test_filter_known_positions_multiple():
    words = ["crane", "crate", "craze", "grace"]
    result = filter_words(words, known_positions={0: "c", 1: "r"})
    assert result == ["crane", "crate", "craze"]


# -- filter_words: excluded_positions (yellow) --

def test_filter_excluded_positions_letter_present_not_at_position():
    words = ["raise", "arise", "spare"]
    # 'r' must be in word but NOT at position 0
    result = filter_words(words, excluded_positions={"r": {0}})
    assert result == ["arise", "spare"]


def test_filter_excluded_positions_letter_missing_from_word():
    words = ["hello", "world"]
    # 'r' must be in the word (min_occurrences) AND not at position 0 (excluded_positions)
    result = filter_words(words, excluded_positions={"r": {0}}, min_occurrences={"r": 1})
    assert result == ["world"]


# -- filter_words: max_occurrences --

def test_filter_max_occurrences():
    words = ["geese", "grape", "greed"]
    # At most 1 'e'
    result = filter_words(words, max_occurrences={"e": 1})
    assert result == ["grape"]


def test_filter_max_occurrences_zero():
    words = ["apple", "grape", "plumb"]
    # No 'a' allowed
    result = filter_words(words, max_occurrences={"a": 0})
    assert result == ["plumb"]


# -- filter_words: combined constraints --

def test_filter_combined():
    words = ["crane", "crate", "craze", "grace", "trace"]
    result = filter_words(
        words,
        known_positions={0: "c", 1: "r"},
        excluded_positions={"e": {4}},
        max_occurrences={"a": 1},
    )
    # "crane" has e at 4 -> excluded
    # "crate" has e at 4 -> excluded
    # "craze" starts c,r, 'e' at 4 -> excluded
    # "grace" doesn't start with c -> excluded
    # "trace" doesn't start with c -> excluded
    assert result == []


def test_filter_combined_with_matches():
    words = ["crane", "crate", "crimp", "crush"]
    result = filter_words(
        words,
        known_positions={0: "c", 1: "r"},
        max_occurrences={"e": 0},
    )
    assert result == ["crimp", "crush"]


# -- edge case: empty word list --

def test_filter_empty_list():
    assert filter_words([]) == []


# -- yellow + green interaction --

def test_filter_yellow_requires_extra_copy_beyond_green():
    words = ["error", "crane", "river"]
    # 'r' is green at position 2, and yellow (not at position 0)
    # This means the word needs r at pos 2 AND at least one more r elsewhere (not pos 0)
    result = filter_words(
        words,
        known_positions={2: "r"},
        excluded_positions={"r": {0}},
    )
    # "error": pos2='r' ✓, r not at 0 ✓, count('r')>=2 ✓ -> match
    # "crane": pos2='a' ✗
    # "river": pos2='v' ✗
    assert result == ["error"]

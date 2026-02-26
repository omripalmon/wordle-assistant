"""Tests for optimal_guess.py."""
from __future__ import annotations

import math
from collections import Counter

import pytest

from optimal_guess import (
    GREEN,
    YELLOW,
    GRAY,
    compute_pattern,
    pattern_distribution,
    entropy,
    score_guess,
    find_optimal_guesses,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _encode(*tiles: int) -> int:
    """Encode a 5-tile pattern list into a base-3 integer."""
    val = 0
    for t in tiles:
        val = val * 3 + t
    return val


# ---------------------------------------------------------------------------
# compute_pattern
# ---------------------------------------------------------------------------

class TestComputePattern:
    def test_all_green(self):
        assert compute_pattern("crane", "crane") == _encode(2, 2, 2, 2, 2)

    def test_all_gray(self):
        assert compute_pattern("crane", "plumb") == _encode(0, 0, 0, 0, 0)

    def test_all_yellow(self):
        # guess="abcde", secret="eabcd" -> every letter present but wrong pos
        assert compute_pattern("abcde", "eabcd") == _encode(1, 1, 1, 1, 1)

    def test_mixed_green_yellow_gray(self):
        # guess="crane", secret="charm"
        # c->GREEN, r->YELLOW, a->GREEN, n->GRAY, e->GRAY
        assert compute_pattern("crane", "charm") == _encode(2, 1, 2, 0, 0)

    def test_simple_yellow_and_green(self):
        # guess="crane", secret="nacre"
        # c:YELLOW r:YELLOW a:YELLOW n:YELLOW e:GREEN
        assert compute_pattern("crane", "nacre") == _encode(1, 1, 1, 1, 2)

    def test_duplicate_letter_in_guess_one_in_secret(self):
        # guess="speed", secret="abide"
        # s:GRAY p:GRAY e(pos2):YELLOW(consumes 1 'e') e(pos3):GRAY d:YELLOW
        assert compute_pattern("speed", "abide") == _encode(0, 0, 1, 0, 1)

    def test_duplicate_letter_in_secret_one_in_guess(self):
        # guess="crane", secret="error"
        # c:GRAY r:GREEN(pos1, consumes 1 r) a:GRAY n:GRAY e:YELLOW(e at pos0)
        assert compute_pattern("crane", "error") == _encode(0, 2, 0, 0, 1)

    def test_green_takes_priority_over_yellow(self):
        # guess="kneel", secret="steel"
        # k:GRAY n:GRAY e(pos2):GREEN e(pos3):GREEN l:GREEN
        assert compute_pattern("kneel", "steel") == _encode(0, 0, 2, 2, 2)

    def test_double_letter_both_green(self):
        # guess="geese", secret="geese"
        assert compute_pattern("geese", "geese") == _encode(2, 2, 2, 2, 2)

    def test_double_letter_limited_yellow(self):
        # guess="llama", secret="world"
        # l(0): not at 0 in 'world', but l at pos 3 in 'world' -> YELLOW (consumes l)
        # l(1): no more l's left -> GRAY
        # a(2): GRAY  m(3): GRAY  a(4): GRAY
        assert compute_pattern("llama", "world") == _encode(1, 0, 0, 0, 0)

    def test_pattern_in_valid_range(self):
        assert 0 <= compute_pattern("aaaaa", "bbbbb") <= 242
        assert 0 <= compute_pattern("abcde", "abcde") <= 242
        assert 0 <= compute_pattern("zzzzz", "aaaaa") <= 242

    def test_all_green_is_242(self):
        assert compute_pattern("hello", "hello") == 242

    def test_all_gray_is_zero(self):
        assert compute_pattern("aaaaa", "bbbbb") == 0


# ---------------------------------------------------------------------------
# entropy
# ---------------------------------------------------------------------------

class TestEntropy:
    def test_uniform_four_buckets(self):
        dist = Counter({0: 25, 1: 25, 2: 25, 3: 25})
        assert entropy(dist, 100) == pytest.approx(2.0)

    def test_single_bucket_zero_entropy(self):
        dist = Counter({0: 100})
        assert entropy(dist, 100) == pytest.approx(0.0)

    def test_two_equal_buckets(self):
        dist = Counter({0: 50, 1: 50})
        assert entropy(dist, 100) == pytest.approx(1.0)

    def test_two_unequal_buckets(self):
        dist = Counter({0: 75, 1: 25})
        expected = -0.75 * math.log2(0.75) - 0.25 * math.log2(0.25)
        assert entropy(dist, 100) == pytest.approx(expected)

    def test_empty_distribution(self):
        assert entropy(Counter(), 0) == pytest.approx(0.0)

    def test_many_singleton_buckets(self):
        # 100 buckets each with count 1 -> H = log2(100)
        dist = Counter({i: 1 for i in range(100)})
        assert entropy(dist, 100) == pytest.approx(math.log2(100))


# ---------------------------------------------------------------------------
# pattern_distribution
# ---------------------------------------------------------------------------

class TestPatternDistribution:
    def test_single_secret_all_green(self):
        dist = pattern_distribution("crane", ["crane"])
        assert dist[242] == 1
        assert sum(dist.values()) == 1

    def test_total_equals_word_count(self):
        words = ["crane", "slate", "plumb", "house", "tiger"]
        dist = pattern_distribution("crane", words)
        assert sum(dist.values()) == 5

    def test_identical_secrets_same_pattern(self):
        dist = pattern_distribution("crane", ["plumb", "plumb", "plumb"])
        assert len(dist) == 1
        assert sum(dist.values()) == 3


# ---------------------------------------------------------------------------
# score_guess
# ---------------------------------------------------------------------------

class TestScoreGuess:
    def test_singleton_zero_entropy(self):
        assert score_guess("crane", ["crane"]) == pytest.approx(0.0)

    def test_identical_words_zero_entropy(self):
        assert score_guess("crane", ["crane", "crane", "crane"]) == pytest.approx(0.0)

    def test_varied_list_positive_entropy(self):
        words = ["crane", "slate", "plumb", "house", "tiger"]
        assert score_guess("crane", words) > 0.0

    def test_entropy_at_most_log2_n(self):
        words = ["crane", "slate", "plumb", "house", "tiger"]
        score = score_guess("crane", words)
        assert score <= math.log2(len(words)) + 1e-9


# ---------------------------------------------------------------------------
# find_optimal_guesses
# ---------------------------------------------------------------------------

class TestFindOptimalGuesses:
    def test_returns_correct_count(self):
        words = ["crane", "slate", "plumb", "house", "tiger"]
        results = find_optimal_guesses(words, top_n=3, workers=1)
        assert len(results) == 3

    def test_sorted_descending(self):
        words = ["crane", "slate", "plumb", "house", "tiger"]
        results = find_optimal_guesses(words, top_n=5, workers=1)
        # Returns (word, raw_entropy, adjusted_entropy) 3-tuples; sort by adjusted
        scores = [adj for _, _, adj in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_larger_than_list(self):
        words = ["crane", "slate", "plumb"]
        results = find_optimal_guesses(words, top_n=10, workers=1)
        assert len(results) == 3

    def test_all_words_present_single_pool(self):
        # When guess_candidates == answer_pool all guess words appear in results
        words = ["crane", "slate", "plumb", "house", "tiger"]
        results = find_optimal_guesses(words, top_n=5, workers=1)
        assert {w for w, _, _ in results} == set(words)

    def test_separate_guess_and_answer_pools(self):
        # Guess candidates is a superset; entropy computed against answer_pool only
        answers = ["crane", "slate", "plumb"]
        guesses = answers + ["zzzzz", "aaaaa"]
        results = find_optimal_guesses(guesses, answer_pool=answers, top_n=5, workers=1)
        assert len(results) == 5
        guess_words = {w for w, _, _ in results}
        assert guess_words == set(guesses)

    def test_bonus_only_for_answer_words(self):
        # "zzzzz" and "aaaaa" are not answers so their adjusted == raw
        answers = ["crane", "slate", "plumb"]
        non_answers = ["zzzzz", "aaaaa"]
        results = find_optimal_guesses(
            answers + non_answers,
            answer_pool=answers,
            top_n=5,
            workers=1,
        )
        result_map = {w: (raw, adj) for w, raw, adj in results}
        for w in non_answers:
            raw, adj = result_map[w]
            assert raw == pytest.approx(adj), f"{w} should have no bonus"
        for w in answers:
            raw, adj = result_map[w]
            assert adj > raw, f"{w} should have bonus"

    def test_empty_list(self):
        assert find_optimal_guesses([], top_n=5, workers=1) == []

    def test_single_worker_matches_multi_worker(self):
        words = [
            "crane", "slate", "plumb", "house", "tiger",
            "bread", "flame", "ghost", "drink", "spoon",
        ]
        r1 = find_optimal_guesses(words, top_n=10, workers=1)
        r2 = find_optimal_guesses(words, top_n=10, workers=2)
        # Same words and scores regardless of worker count
        assert set(r1) == set(r2)

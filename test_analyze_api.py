"""Regression tests for the FastAPI /analyze endpoint.

Uses FastAPI's TestClient to POST each fixture image and assert that the
JSON response matches the confirmed baseline stored in expected.json.

Test layers
-----------
1. HTTP contract    — status 200, required top-level keys always present.
2. Colour fidelity  — tile colours in the response match expected codes.
3. Candidate count  — exact count locked to baseline (detects filter regressions).
4. Best-guess word  — best_overall and best_valid words locked to baseline.
5. Structural       — suggestions sorted descending, adjusted ≥ raw entropy.
6. Blank-image path — blank grid returns all words as candidates + valid best guess.

Run:
    pytest test_analyze_api.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app, RAW_WORDS

# ── Setup ─────────────────────────────────────────────────────────────────────
client = TestClient(app)

FIXTURES_DIR = Path(__file__).parent / "tests" / "fixtures"

with open(FIXTURES_DIR / "expected.json") as _f:
    EXPECTED: dict[str, dict] = json.load(_f)

TOTAL_WORDS = len(RAW_WORDS)

REQUIRED_KEYS = {
    "guesses", "candidates", "candidate_count",
    "top_suggestions", "best_overall", "best_valid", "bonus",
}

_COLOR_MAP = {"g": "green", "y": "yellow", "b": "black"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _post(name: str) -> dict:
    """POST a fixture image to /analyze and return the parsed JSON."""
    path = FIXTURES_DIR / name
    with open(path, "rb") as f:
        resp = client.post("/analyze", files={"image": (name, f, "image/png")})
    assert resp.status_code == 200, (
        f"{name}: HTTP {resp.status_code} — {resp.text[:200]}"
    )
    return resp.json()


def _all_names() -> list[str]:
    return list(EXPECTED.keys())


# ── 1. HTTP contract ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", _all_names())
def test_status_200(name: str) -> None:
    """Every fixture must return HTTP 200."""
    _post(name)  # assertion inside _post


@pytest.mark.parametrize("name", _all_names())
def test_required_keys_present(name: str) -> None:
    """Response must contain all required top-level keys."""
    data = _post(name)
    missing = REQUIRED_KEYS - data.keys()
    assert not missing, f"{name}: missing keys {missing}"


# ── 2. Colour fidelity ────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", _all_names())
def test_tile_colours_match_baseline(name: str) -> None:
    """Tile colours in guesses[].tiles[].color match the confirmed baseline."""
    data     = _post(name)
    expected = EXPECTED[name]["responses"]
    assert len(data["guesses"]) == len(expected), (
        f"{name}: {len(data['guesses'])} guesses but {len(expected)} expected"
    )
    for row_i, (guess, resp_code) in enumerate(zip(data["guesses"], expected)):
        for col_i, (tile, code) in enumerate(zip(guess["tiles"], resp_code)):
            want = _COLOR_MAP[code]
            got  = tile["color"]
            assert got == want, (
                f"{name} row {row_i} col {col_i}: colour '{got}' ≠ '{want}'"
            )


# ── 3. Candidate count locked to baseline ────────────────────────────────────

@pytest.mark.parametrize("name", _all_names())
def test_candidate_count_matches_baseline(name: str) -> None:
    """Candidate count is locked to the baseline value captured at confirmation.

    A change here means either the wordlist or the filtering logic changed.
    """
    data     = _post(name)
    expected = EXPECTED[name]["candidate_count"]
    assert data["candidate_count"] == expected, (
        f"{name}: candidate_count {data['candidate_count']} ≠ baseline {expected}"
    )


# ── 4. Best-guess words locked to baseline ───────────────────────────────────

@pytest.mark.parametrize("name", _all_names())
def test_best_overall_word_matches_baseline(name: str) -> None:
    """best_overall.word is locked to the baseline."""
    data     = _post(name)
    expected = EXPECTED[name]["best_overall_word"]
    got      = data["best_overall"]["word"] if data["best_overall"] else None
    assert got == expected, (
        f"{name}: best_overall '{got}' ≠ baseline '{expected}'"
    )


@pytest.mark.parametrize("name", _all_names())
def test_best_valid_word_matches_baseline(name: str) -> None:
    """best_valid.word is locked to the baseline."""
    data     = _post(name)
    expected = EXPECTED[name]["best_valid_word"]
    got      = data["best_valid"]["word"] if data["best_valid"] else None
    assert got == expected, (
        f"{name}: best_valid '{got}' ≠ baseline '{expected}'"
    )


# ── 5. Structural invariants ─────────────────────────────────────────────────

@pytest.mark.parametrize("name", _all_names())
def test_suggestions_sorted_descending(name: str) -> None:
    """top_suggestions must be sorted descending by adjusted_entropy."""
    data   = _post(name)
    scores = [s["adjusted_entropy"] for s in data["top_suggestions"]]
    assert scores == sorted(scores, reverse=True), (
        f"{name}: suggestions not sorted descending: {scores}"
    )


@pytest.mark.parametrize("name", _all_names())
def test_adjusted_entropy_gte_raw(name: str) -> None:
    """adjusted_entropy must be ≥ raw_entropy for every suggestion (bonus ≥ 0)."""
    data = _post(name)
    for s in data["top_suggestions"]:
        assert s["adjusted_entropy"] >= s["raw_entropy"] - 1e-6, (
            f"{name}: adjusted {s['adjusted_entropy']} < raw {s['raw_entropy']}"
        )


@pytest.mark.parametrize("name", _all_names())
def test_candidates_list_length_matches_count(name: str) -> None:
    """len(candidates) must equal candidate_count (or be capped at list limit)."""
    data = _post(name)
    # The API may truncate the returned list but the count must be ≥ list length
    assert data["candidate_count"] >= len(data["candidates"]), (
        f"{name}: candidate_count {data['candidate_count']} < "
        f"len(candidates) {len(data['candidates'])}"
    )


# ── 6. Blank-image specific tests ────────────────────────────────────────────

def test_blank_returns_zero_guesses() -> None:
    """Blank grid must yield an empty guesses list (no constraint is applied)."""
    data = _post("blank_dark.png")
    assert data["guesses"] == []


def test_blank_candidate_count_equals_full_wordlist() -> None:
    """With no constraints every word in the list is a candidate."""
    data = _post("blank_dark.png")
    assert data["candidate_count"] == TOTAL_WORDS, (
        f"blank: candidate_count {data['candidate_count']} ≠ "
        f"total words {TOTAL_WORDS}"
    )


def test_blank_best_overall_is_not_null() -> None:
    """Blank grid must return a valid best_overall suggestion."""
    data = _post("blank_dark.png")
    assert data["best_overall"] is not None
    assert data["best_valid"]   is not None
    assert len(data["top_suggestions"]) > 0


# ── 7. Cross-fixture sanity checks ───────────────────────────────────────────

def test_more_guesses_means_fewer_candidates() -> None:
    """Adding a second guess must not increase the candidate pool."""
    one_guess  = _post("raise_ybbby_v1.png")["candidate_count"]
    two_guesses = _post("raise_chain.png")["candidate_count"]
    assert two_guesses <= one_guess, (
        f"Two guesses ({two_guesses}) produced more candidates than one ({one_guess})"
    )


def test_variant_images_produce_identical_candidate_counts() -> None:
    """Two crops of the same game must yield the same candidate_count."""
    for base in ("smelt", "raise_ybbby", "raise_bybbb"):
        c1 = _post(f"{base}_v1.png")["candidate_count"]
        c2 = _post(f"{base}_v2.png")["candidate_count"]
        assert c1 == c2, (
            f"{base}: v1 candidates={c1} ≠ v2 candidates={c2}"
        )


# ── 8. Health endpoint ────────────────────────────────────────────────────────

def test_health_endpoint() -> None:
    """GET /health must return status=ok and report the loaded word count."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert int(data["words_loaded"]) == TOTAL_WORDS

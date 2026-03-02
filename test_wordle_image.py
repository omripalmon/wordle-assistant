"""Regression tests for wordle_image.parse_wordle_image.

Each fixture is a real Wordle screenshot captured during development.
The expected baseline is stored in tests/fixtures/expected.json and was
produced by running the code against the images when all known-good
behaviours were verified.

Test layers
-----------
1. Guess count      — how many played rows were detected (no Tesseract needed).
2. Colour codes     — g/y/b per tile; the primary regression signal (no Tesseract).
3. OCR words        — parametrised over *reliable* fixtures only; skipped when
                      Tesseract is not installed.
4. Known OCR failures — marked xfail so they remain visible in CI output without
                      blocking the suite.  Update when the underlying bug is fixed.
5. Targeted regression tests — one assertion per historical bug fix.

Run:
    pytest test_wordle_image.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wordle_image import parse_wordle_image, _tesseract_available

# ── Paths & baseline ─────────────────────────────────────────────────────────
FIXTURES_DIR = Path(__file__).parent / "tests" / "fixtures"

with open(FIXTURES_DIR / "expected.json") as _f:
    EXPECTED: dict[str, dict] = json.load(_f)

HAS_TESSERACT = _tesseract_available()
REQUIRES_TESSERACT = pytest.mark.skipif(
    not HAS_TESSERACT, reason="Tesseract not installed"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fixture(name: str) -> Path:
    return FIXTURES_DIR / name


def _all_names() -> list[str]:
    return list(EXPECTED.keys())


def _reliable_names() -> list[str]:
    """Fixtures where every OCR word result is currently reliable."""
    return [
        n for n, exp in EXPECTED.items()
        if exp["words"] and all(exp["word_ocr_reliable"])
    ]


def _unreliable_names() -> list[str]:
    """Fixtures with at least one word where OCR is known to be wrong."""
    return [
        n for n, exp in EXPECTED.items()
        if any(not r for r in exp.get("word_ocr_reliable", []))
    ]


# ── 1. Guess-count tests (no Tesseract) ──────────────────────────────────────

@pytest.mark.parametrize("name", _all_names())
def test_guess_count(name: str) -> None:
    """Correct number of played rows is detected."""
    result = parse_wordle_image(_fixture(name))
    expected = EXPECTED[name]["guess_count"]
    assert len(result) == expected, (
        f"{name}: expected {expected} guess(es), got {len(result)}"
    )


# ── 2. Colour-code tests (no Tesseract) ──────────────────────────────────────

@pytest.mark.parametrize("name", _all_names())
def test_response_codes(name: str) -> None:
    """Tile colour codes (g/y/b) exactly match the confirmed baseline."""
    result = parse_wordle_image(_fixture(name))
    actual   = [resp for _, resp in result]
    expected = EXPECTED[name]["responses"]
    assert actual == expected, (
        f"{name}: colours {actual!r} ≠ expected {expected!r}"
    )


# ── 3. OCR word tests — reliable fixtures ────────────────────────────────────

@REQUIRES_TESSERACT
@pytest.mark.parametrize("name", _reliable_names())
def test_ocr_words_reliable(name: str) -> None:
    """OCR correctly reads all words for fixtures where output is reliable."""
    result = parse_wordle_image(_fixture(name))
    for i, ((actual_word, _), expected_word) in enumerate(
        zip(result, EXPECTED[name]["words"])
    ):
        assert actual_word == expected_word, (
            f"{name} row {i}: OCR '{actual_word}' ≠ expected '{expected_word}'"
        )


# ── 4. OCR word tests — known failures (xfail) ───────────────────────────────

@REQUIRES_TESSERACT
@pytest.mark.parametrize("name", _unreliable_names())
@pytest.mark.xfail(
    strict=True,
    reason="Known OCR failure — word_ocr_reliable=False in expected.json. "
           "Flip to True and update 'words' when the underlying bug is fixed.",
)
def test_ocr_words_known_failures(name: str) -> None:
    """Documents known OCR word failures; xfail keeps them visible without
    blocking CI.  These will become xpass once the bug is fixed, at which
    point the fixture entry should be updated and this mark removed."""
    result = parse_wordle_image(_fixture(name))
    actual_words = [w for w, _ in result]
    # expected["words"] records the *human-correct* word ("smelt"), not the
    # current buggy OCR output ("snelt").  The test xfails because they differ.
    assert actual_words == EXPECTED[name]["words"]


# ── 5. Targeted regression tests ─────────────────────────────────────────────

def test_blank_grid_returns_empty_list() -> None:
    """Regression: blank/unplayed grid returns [] instead of raising ValueError."""
    assert parse_wordle_image(_fixture("blank_dark.png")) == []


@REQUIRES_TESSERACT
def test_i_detection_in_chain_row() -> None:
    """Regression: 'I' at position 3 of CHAIN is read correctly (was '?').

    Commit: 3e26097 — Fix OCR returning '?' for 'I' using shape-based detection.
    """
    result = parse_wordle_image(_fixture("raise_chain.png"))
    assert len(result) == 2, "raise_chain.png must have exactly 2 guesses"
    word, resp = result[1]   # second guess is CHAIN
    assert resp[3] == "g",  f"CHAIN tile 3 colour should be green, got '{resp[3]}'"
    assert word[3] == "i",  f"CHAIN tile 3 letter should be 'i', got '{word[3]}'"


@REQUIRES_TESSERACT
def test_smelt_m_read_correctly() -> None:
    """Regression: bold M in SMELT misread as N (older Tesseract) or I (5.5.0).

    Fixed by extending the M/N CoM-symmetry disambiguator to also trigger
    when Tesseract returns 'I' but the stroke span is wide (> 20 % of the
    binarised-image width), indicating the glyph is too wide to be a real 'I'.

    Covers both known confusions:
      - Tesseract <5.5: M → N  (original fix: CoM drift check)
      - Tesseract 5.5.0: M → I  (new: wide-span guard + same CoM check)
    """
    for fixture in ("smelt_v1.png", "smelt_v2.png"):
        result = parse_wordle_image(_fixture(fixture))
        assert len(result) == 1, f"{fixture}: expected 1 guess, got {len(result)}"
        word, resp = result[0]
        assert resp == "bbygg", f"{fixture}: colour codes changed unexpectedly: '{resp}'"
        assert word == "smelt", (
            f"{fixture}: expected 'smelt', got '{word}' — M/N/I fix may have regressed"
        )


@REQUIRES_TESSERACT
def test_two_guess_game_both_rows_detected() -> None:
    """Regression: second played row in dark mode was sometimes missed.

    Commit: 57759d2 — Fix dark-mode second row not detected.
    """
    result = parse_wordle_image(_fixture("raise_chain.png"))
    assert len(result) == 2, (
        f"raise_chain.png must detect 2 guesses, got {len(result)}"
    )


@REQUIRES_TESSERACT
def test_flume_f_read_correctly() -> None:
    """Regression: bold 'F' in FLUME misread as 'E' by cloud Tesseract 5.x.

    Fixed by checking lower-middle right-half pixel density: 'E' (bottom bar)
    has dense dark pixels in rows 55–75 % of the right half of the letter,
    while 'F' (no bottom bar) has none there.
    """
    result = parse_wordle_image(_fixture("raise_could_flume.png"))
    assert len(result) == 3, "raise_could_flume.png must have exactly 3 guesses"
    word, resp = result[2]  # third guess is FLUME
    assert resp == "gggbg", f"FLUME colour codes changed: '{resp}'"
    assert word == "flume", (
        f"expected 'flume', got '{word}' — F/E fix may have regressed"
    )


@REQUIRES_TESSERACT
def test_could_o_read_correctly() -> None:
    """Regression: bold 'O' in COULD misread as 'C' by Tesseract.

    Fixed by checking equatorial right-side pixel density: 'O' (closed ring)
    has dense dark pixels in the centre-right region, while 'C' (open arc)
    has none there.
    """
    result = parse_wordle_image(_fixture("raise_could.png"))
    assert len(result) == 2, "raise_could.png must have exactly 2 guesses"
    word, resp = result[1]  # second guess is COULD
    assert resp == "bbgyb", f"COULD colour codes changed: '{resp}'"
    assert word == "could", (
        f"expected 'could', got '{word}' — O/C fix may have regressed"
    )


def test_variant_images_produce_identical_results() -> None:
    """Two crops of the same game state must produce the same colour codes."""
    for base in ("smelt", "raise_ybbby", "raise_bybbb"):
        v1 = parse_wordle_image(_fixture(f"{base}_v1.png"))
        v2 = parse_wordle_image(_fixture(f"{base}_v2.png"))
        r1 = [r for _, r in v1]
        r2 = [r for _, r in v2]
        assert r1 == r2, (
            f"{base}: v1 colours {r1!r} ≠ v2 colours {r2!r}"
        )

"""FastAPI backend for the Wordle Helper web service.

Endpoints:
    POST /analyze   — accepts a multipart image upload, returns JSON analysis
    GET  /health    — liveness probe for Railway

Pipeline (sequential, no multiprocessing):
    1. parse_wordle_image()  — detect grid, classify tile colours, OCR letters
    2. parse_guesses()       — convert (word, response) pairs into constraints
    3. filter_words()        — narrow the word list to valid candidates
    4. score_guess() loop    — rank every word in the full list by Shannon entropy
    5. Serialise results     — return JSON matching the agreed schema

Run locally (from the repo root):
    uvicorn api.main:app --reload --port 8000 --app-dir .
"""
from __future__ import annotations

import asyncio
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — allow imports from /app (Docker WORKDIR) or the repo root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from main import parse_guesses
from optimal_guess import score_guess, valid_word_bonus
from wordle_filter import filter_words, load_words
from wordle_image import parse_wordle_image

# ---------------------------------------------------------------------------
# Word list — loaded once at startup to avoid per-request disk I/O
# ---------------------------------------------------------------------------
_WORDLIST_CANDIDATES = [
    ROOT / "wordlist.txt",          # bundled curated list (preferred)
    Path("/app/wordlist.txt"),       # Docker WORKDIR path
    Path("/usr/share/dict/words"),   # OS fallback
]
_wordlist_path = next(
    (p for p in _WORDLIST_CANDIDATES if p.exists()), None
)
if _wordlist_path is None:
    raise RuntimeError(
        "No word list found. Expected wordlist.txt in the project root or "
        "/usr/share/dict/words on the system."
    )

RAW_WORDS: list[str] = load_words(str(_wordlist_path))
WORD_SET: set[str] = set(RAW_WORDS)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Wordle Helper API",
    description=(
        "Analyses a Wordle screenshot and suggests the best next guess "
        "using Shannon entropy scoring."
    ),
    version="1.0.0",
)

# CORS — allow the Vercel frontend and local development.
# Set the CORS_ORIGINS env var in Railway to your Vercel URL.
_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS",
        "https://wordle-helper.vercel.app,http://localhost:3000,http://localhost:5500",
    ).split(",")
    if origin.strip()
]
_ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] ,
    #_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe — returns OK plus word list size."""
    return {"status": "ok", "words_loaded": str(len(RAW_WORDS))}


@app.get("/debug")
async def debug() -> dict[str, Any]:
    """Diagnostic endpoint — reports Tesseract availability and path."""
    import shutil
    tess_path = shutil.which("tesseract")
    tess_version: str | None = None
    tessdata_prefix = os.environ.get("TESSDATA_PREFIX", "not set")
    if tess_path:
        import subprocess
        try:
            result = subprocess.run(
                ["tesseract", "--version"], capture_output=True, text=True, timeout=5
            )
            tess_version = (result.stdout or result.stderr).strip().splitlines()[0]
        except Exception as exc:
            tess_version = f"error: {exc}"
    # Check if pytesseract can see it
    try:
        import pytesseract
        pytesseract_version = pytesseract.get_tesseract_version()
        pytesseract_ok = True
    except Exception as exc:
        pytesseract_version = str(exc)
        pytesseract_ok = False
    return {
        "tesseract_path": tess_path,
        "tesseract_version": tess_version,
        "tessdata_prefix": tessdata_prefix,
        "pytesseract_ok": pytesseract_ok,
        "pytesseract_version": str(pytesseract_version),
        "words_loaded": len(RAW_WORDS),
    }


@app.post("/diagnose")
async def diagnose(image: UploadFile = File(...)) -> dict[str, Any]:
    """Deep diagnostic — returns per-tile RGB, colour, OCR letter, and preprocessed tile b64.

    Upload the same image you'd send to /analyze. Returns a row-by-row,
    tile-by-tile breakdown including the base64-encoded preprocessed image
    that Tesseract actually receives, so OCR issues can be inspected.
    """
    import base64
    import colorsys
    import io
    import colorsys
    from wordle_image import (
        _background_color, _find_tile_rows, _find_tile_cols,
        _sample_tile_color, _ocr_tile_letter, classify_tile_color,
        _color_distance, _tesseract_available,
    )
    from PIL import Image as PILImage, ImageOps, ImageFilter

    suffix = Path(image.filename or "upload.png").suffix or ".png"
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(await image.read())

        img = PILImage.open(tmp_path).convert("RGB")
        bg = _background_color(img)
        tile_rows = _find_tile_rows(img)

        rows_out = []
        reference_cols = None
        for row_i, (top, bottom) in enumerate(tile_rows[:2]):  # only first 2 played rows
            cols = _find_tile_cols(img, top, bottom, reference_cols=reference_cols)
            if len(cols) == 5:
                reference_cols = cols

            tiles_out = []
            for col_i, (left, right) in enumerate(cols):
                rgb = _sample_tile_color(img, left, right, top, bottom)
                tile_color = classify_tile_color(rgb)
                letter = _ocr_tile_letter(img, left, right, top, bottom)
                h, s, v = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)

                # Build the preprocessed image Tesseract actually receives
                tile_b64 = None
                try:
                    w, wh = right - left, bottom - top
                    factor = 4
                    scaled = img.resize((img.width * factor, img.height * factor), PILImage.LANCZOS)
                    inset_x = max(factor, (w // 10) * factor)
                    inset_y = max(factor, (wh // 10) * factor)
                    crop = scaled.crop((left*factor+inset_x, top*factor+inset_y,
                                        right*factor-inset_x, bottom*factor-inset_y))
                    grey = crop.convert("L")
                    pixels = list(grey.getdata())
                    lo, hi = min(pixels), max(pixels)
                    if hi > lo:
                        grey = grey.point(lambda p: int((p - lo) * 255 / (hi - lo)))
                    inv = ImageOps.invert(grey)
                    # Otsu binarisation — makes letter pure black on pure white
                    inv_pixels = list(inv.getdata())
                    hist = [0] * 256
                    for p in inv_pixels:
                        hist[p] += 1
                    total_px = len(inv_pixels)
                    sum_all = sum(i * hist[i] for i in range(256))
                    sum_bg = wb = 0
                    otsu_t = 128
                    max_var = 0.0
                    for t in range(256):
                        wb += hist[t]
                        if wb == 0: continue
                        wf = total_px - wb
                        if wf == 0: break
                        sum_bg += t * hist[t]
                        mb = sum_bg / wb
                        mf = (sum_all - sum_bg) / wf
                        var = wb * wf * (mb - mf) ** 2
                        if var > max_var:
                            max_var = var
                            otsu_t = t
                    inv = inv.point(lambda p: 0 if p < otsu_t else 255)
                    sharp = inv.filter(ImageFilter.SHARPEN)
                    pad = max(8, sharp.width // 8)
                    padded = PILImage.new("L", (sharp.width + 2*pad, sharp.height + 2*pad), 255)
                    padded.paste(sharp, (pad, pad))
                    buf = io.BytesIO()
                    padded.save(buf, format="PNG")
                    tile_b64 = base64.b64encode(buf.getvalue()).decode()
                except Exception as ex:
                    tile_b64 = f"error:{ex}"

                tiles_out.append({
                    "col": col_i + 1,
                    "bounds": {"left": left, "right": right},
                    "rgb": list(rgb),
                    "hue": round(h * 360, 1),
                    "sat": round(s, 3),
                    "val": round(v, 3),
                    "color": tile_color.label,
                    "color_code": tile_color.code,
                    "ocr_letter": letter,
                    "tile_b64": tile_b64,
                })
            rows_out.append({
                "row": row_i + 1,
                "bounds": {"top": top, "bottom": bottom},
                "col_count": len(cols),
                "tiles": tiles_out,
            })

        return {
            "image_size": {"width": img.width, "height": img.height},
            "background_rgb": list(bg),
            "row_count": len(tile_rows),
            "rows": rows_out,
        }
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Regression-test runner
# ---------------------------------------------------------------------------

_SUITE_FILES: dict[str, list[str]] = {
    "image": ["test_wordle_image.py"],
    "api":   ["test_analyze_api.py"],
    "all":   ["test_wordle_image.py", "test_analyze_api.py"],
}


def _parse_pytest_output(stdout: str, elapsed: float, suite: str) -> dict[str, Any]:
    """Parse ``pytest -v --tb=line`` output into structured JSON."""
    tests: list[dict[str, Any]] = []

    # Per-test result lines:  "path::name OUTCOME   [XX%]"
    RESULT_RE = re.compile(
        r"^(\S+::\S+)\s+(PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)\s+\[",
        re.MULTILINE,
    )
    for m in RESULT_RE.finditer(stdout):
        nodeid  = m.group(1)
        raw     = m.group(2)
        outcome = (
            raw.lower()
               .replace("xfail", "xfailed")
               .replace("xpass", "xpassed")
        )
        file_part, _, name_part = nodeid.partition("::")
        tests.append({
            "id":      nodeid,
            "file":    file_part,
            "name":    name_part,
            "outcome": outcome,
            "message": "",
        })

    # --tb=line failure messages: "FAILED path::name - AssertionError: ..."
    FAIL_RE = re.compile(
        r"^(?:FAILED|ERROR)\s+(\S+::\S+)\s+-\s+(.+)$", re.MULTILINE
    )
    messages = {m.group(1): m.group(2).strip() for m in FAIL_RE.finditer(stdout)}
    for t in tests:
        if t["outcome"] in ("failed", "error"):
            t["message"] = messages.get(t["id"], "")

    counts: dict[str, int] = {
        k: 0 for k in ("passed", "failed", "xfailed", "xpassed", "skipped", "error")
    }
    for t in tests:
        if t["outcome"] in counts:
            counts[t["outcome"]] += 1

    return {
        "suite":      suite,
        "total":      len(tests),
        "duration":   elapsed,
        "tests":      tests,
        # summary line from pytest (last non-empty line starting with "=")
        "summary":    next(
            (ln.strip("= \n") for ln in reversed(stdout.splitlines()) if ln.startswith("=") and "passed" in ln),
            "",
        ),
        **counts,
    }


@app.get("/tests/expected")
async def get_expected() -> dict[str, Any]:
    """Return the expected fixture metadata from tests/fixtures/expected.json."""
    import json
    path = ROOT / "tests" / "fixtures" / "expected.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="expected.json not found")
    return json.loads(path.read_text())


@app.get("/tests/fixture-ocr")
async def fixture_ocr_results() -> dict[str, Any]:
    """Run parse_wordle_image on every fixture and return actual OCR results.

    Returns a mapping of fixture filename → {words, responses} (or {error}).
    Used by the frontend to compare expected vs identified characters/colours.
    """
    import json

    fixtures_dir = ROOT / "tests" / "fixtures"
    expected_path = fixtures_dir / "expected.json"
    if not expected_path.exists():
        raise HTTPException(status_code=404, detail="expected.json not found")

    expected: dict[str, Any] = json.loads(expected_path.read_text())

    def _process_all() -> dict[str, Any]:
        results: dict[str, Any] = {}
        for filename in expected:
            fixture_path = fixtures_dir / filename
            if not fixture_path.exists():
                results[filename] = {"error": "fixture not found", "words": [], "responses": []}
                continue
            try:
                guesses = parse_wordle_image(str(fixture_path))
                results[filename] = {
                    "words":     [g[0] for g in guesses],
                    "responses": [g[1] for g in guesses],
                }
            except Exception as exc:
                results[filename] = {"error": str(exc), "words": [], "responses": []}
        return results

    return await asyncio.to_thread(_process_all)


@app.get("/tests/run")
async def run_tests(suite: str = "image") -> dict[str, Any]:
    """Run the regression test suite and return structured results.

    Parameters
    ----------
    suite:
        ``image`` — image-parsing tests only (fast, ~1 min).
        ``api``   — API tests with entropy scoring (~20 min).
        ``all``   — both suites.
    """
    if suite not in _SUITE_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"suite must be one of: {list(_SUITE_FILES)}",
        )

    test_paths = [ROOT / f for f in _SUITE_FILES[suite]]
    missing = [p.name for p in test_paths if not p.exists()]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Test files not found: {missing}. "
                   "Ensure the repo is deployed with test files present.",
        )

    fixtures_dir = ROOT / "tests" / "fixtures"
    if not fixtures_dir.exists():
        raise HTTPException(
            status_code=404,
            detail="Test fixtures not found at tests/fixtures/.",
        )

    def _run() -> dict[str, Any]:
        t0 = time.monotonic()
        proc = subprocess.run(
            [sys.executable, "-m", "pytest",
             *[str(p) for p in test_paths],
             "-v", "--tb=line", "--no-header", "-p", "no:warnings"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=2700,          # 45-minute hard stop
        )
        elapsed = round(time.monotonic() - t0, 2)
        result  = _parse_pytest_output(proc.stdout, elapsed, suite)
        result["returncode"] = proc.returncode
        # Attach tail of raw output so callers can debug unexpected failures
        combined = proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr.strip() else "")
        result["output_tail"] = combined[-4000:]
        return result

    return await asyncio.to_thread(_run)


# ---------------------------------------------------------------------------
# Analysis endpoint
# ---------------------------------------------------------------------------

@app.post("/analyze")
async def analyze(image: UploadFile = File(...)) -> dict[str, Any]:
    """Analyse a Wordle screenshot and return the best next guesses.

    Accepts a multipart/form-data upload with a single field named ``image``.

    Response schema::

        {
          "guesses": [
            {
              "word": "raise",
              "response": "bgybb",
              "tiles": [{"letter": "R", "color": "black"}, ...]
            }
          ],
          "candidates":      ["matin", "cabin", ...],
          "candidate_count": 142,
          "top_suggestions": [
            {
              "word": "tunic",
              "raw_entropy": 4.0628,
              "adjusted_entropy": 4.0628,
              "is_candidate": false
            },
            ...
          ],
          "best_overall": {"word": "tunic", "raw": 4.0628, "adjusted": 4.0628},
          "best_valid":   {"word": "matin", "raw": 3.9712, "adjusted": 4.0216},
          "bonus": 0.0504
        }
    """
    # ------------------------------------------------------------------
    # 1. Write upload to a temp file — parse_wordle_image needs a path
    # ------------------------------------------------------------------
    suffix = Path(image.filename or "upload.png").suffix or ".png"
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            contents = await image.read()
            tmp.write(contents)

        # ------------------------------------------------------------------
        # 2. Parse the Wordle grid from the image
        # ------------------------------------------------------------------
        try:
            raw_guesses: list[tuple[str, str]] = parse_wordle_image(tmp_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=f"Temp file error: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Check if OCR produced any real letters — '?' means Tesseract couldn't
        # read the tile.  We still return colour-based results but cannot derive
        # letter constraints from unknown words.
        # Empty guess list (blank grid) is valid — treat as no constraints.
        ocr_failed = bool(raw_guesses) and all("?" in word for word, _ in raw_guesses)

        # ------------------------------------------------------------------
        # 3. Convert (word, response) pairs to Wordle constraints
        #    Skip guesses whose letters are unknown (OCR returned '?????').
        # ------------------------------------------------------------------
        known_guesses = [
            (word, response)
            for word, response in raw_guesses
            if word.replace("?", "").strip() != "" and word.isalpha()
        ]
        guess_args = [f"{word},{response}" for word, response in known_guesses]
        try:
            known_positions, excluded_positions, min_occurrences, max_occurrences = (
                parse_guesses(guess_args) if guess_args else ({}, {}, {}, {})
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Constraint error: {exc}") from exc

        # ------------------------------------------------------------------
        # 4. Filter the word list to valid candidates
        # ------------------------------------------------------------------
        candidates: list[str] = filter_words(
            RAW_WORDS,
            known_positions=known_positions or None,
            excluded_positions=excluded_positions or None,
            min_occurrences=min_occurrences or None,
            max_occurrences=max_occurrences or None,
        )
        candidate_set = set(candidates)
        n = len(candidates)

        # ------------------------------------------------------------------
        # 5. Sequential entropy scoring
        #    Score every word in RAW_WORDS against the candidate pool.
        #    This replicates the constrained-mode logic from optimal_guess.py
        #    without multiprocessing (safe inside an async request handler).
        # ------------------------------------------------------------------
        bonus = valid_word_bonus(n)
        top_n = 10

        if n == 0:
            scored: list[tuple[str, float, float]] = []
        elif n == 1:
            only = candidates[0]
            raw = score_guess(only, candidates)
            scored = [(only, raw, raw + bonus)]
        else:
            scored = sorted(
                (
                    (
                        word,
                        (raw_score := score_guess(word, candidates)),
                        raw_score + (bonus if word in candidate_set else 0.0),
                    )
                    for word in RAW_WORDS
                ),
                key=lambda t: t[2],
                reverse=True,
            )

        top_suggestions = [
            {
                "word": word,
                "raw_entropy": round(raw_s, 4),
                "adjusted_entropy": round(adj_s, 4),
                "is_candidate": word in candidate_set,
            }
            for word, raw_s, adj_s in scored[:top_n]
        ]

        best_overall = scored[0] if scored else None
        best_valid = next((t for t in scored if t[0] in candidate_set), None)

        def _fmt(t: tuple[str, float, float] | None) -> dict[str, Any] | None:
            if t is None:
                return None
            return {"word": t[0], "raw": round(t[1], 4), "adjusted": round(t[2], 4)}

        # ------------------------------------------------------------------
        # 6. Build per-tile detail for the guesses section
        # ------------------------------------------------------------------
        _COLOR_MAP = {"g": "green", "y": "yellow", "b": "black"}

        guesses_payload = [
            {
                "word": word,
                "response": response,
                "tiles": [
                    {
                        "letter": letter.upper() if letter != "?" else "?",
                        "color": _COLOR_MAP.get(code, "black"),
                    }
                    for letter, code in zip(word, response)
                ],
            }
            for word, response in raw_guesses
        ]

        return {
            "guesses": guesses_payload,
            "candidates": candidates,
            "candidate_count": n,
            "top_suggestions": top_suggestions,
            "best_overall": _fmt(best_overall),
            "best_valid": _fmt(best_valid),
            "bonus": round(bonus, 4),
            "ocr_failed": ocr_failed,
            "ocr_warning": (
                "Letter OCR unavailable on this server — tile colours were read "
                "correctly but letter constraints could not be applied. "
                "Results are based on colour patterns only."
                if ocr_failed else None
            ),
        }

    finally:
        # Always clean up the temp file
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

"""Parse a Wordle screenshot and extract guesses with their tile responses.

Supports the NYT Wordle (light + dark mode), high-contrast mode, and most
common Wordle clones that use the standard green / yellow / dark-grey palette.

Color classification uses HSV so it is robust to JPEG compression artefacts,
display scaling, and minor colour shifts between themes:

  Green  — hue 100-165 °, saturation ≥ 25 %   (correct letter & position)
  Yellow — hue  15-75  °, saturation ≥ 25 %   (correct letter, wrong pos)
  Black  — everything else                     (letter not in word / absent)

High-contrast mode maps orange (hue 15-35°, sat ≥ 25 %) → yellow and
cyan/blue (hue 165-210°, sat ≥ 35 %) → green; those ranges are also covered.

Letter OCR
----------
When Tesseract is installed (``brew install tesseract`` / ``apt install
tesseract-ocr``) and the ``pytesseract`` Python wrapper is available
(``pip install pytesseract``), each tile is cropped, contrast-inverted so the
letter appears black-on-white, and fed to Tesseract in single-character mode.
The result is then filtered to A-Z; any non-letter output falls back to '?'.

If Tesseract is not installed, letter extraction is skipped and '?' is used
for each tile (or words can be supplied explicitly via the ``words`` argument).

Algorithm
---------
1. Convert image to RGB, then scan rows to find horizontal bands of tiles.
   Each tile row is a horizontal stripe where the image contains a grid of
   same-sized coloured squares separated by thin gaps.
2. Auto-detect tile size and grid layout by looking for the repeating pattern
   of coloured regions (using colour distance from background, not brightness,
   so dark tiles are detected correctly).
3. For each tile: sample its centre patch → classify colour; crop the inner
   region → OCR the letter.
4. Return list of (word, response) tuples.

Usage::

    from wordle_image import parse_wordle_image

    # Full auto: colours + letters via OCR (requires Tesseract)
    guesses = parse_wordle_image('screenshot.png')
    for word, response in guesses:
        print(f'{word},{response}')

    # Supply words explicitly (no OCR needed):
    guesses = parse_wordle_image('screenshot.png', words=['crane', 'audio'])

CLI::

    python wordle_image.py screenshot.png
    python wordle_image.py screenshot.png --words crane audio
    python wordle_image.py screenshot.png --diagnose
"""
from __future__ import annotations

import argparse
import colorsys
import sys
from pathlib import Path
from typing import NamedTuple

try:
    from PIL import Image, ImageFilter, ImageOps
except ImportError:
    sys.exit("Pillow is required: pip install Pillow")

# pytesseract is optional — OCR is skipped gracefully if not available.
try:
    import os
    import shutil

    import pytesseract
    from pytesseract import TesseractNotFoundError as _TesseractNotFoundError

    # Tesseract may be installed in a non-standard location (e.g. ~/homebrew on
    # macOS when Homebrew was installed without sudo).  Try common paths and
    # point pytesseract at the first one that exists.
    _TESSERACT_SEARCH_PATHS = [
        "tesseract",                                        # already on PATH
        os.path.expanduser("~/homebrew/bin/tesseract"),    # non-root Homebrew
        "/opt/homebrew/bin/tesseract",                      # Apple-Silicon Homebrew
        "/usr/local/bin/tesseract",                         # Intel-Mac Homebrew
        "/usr/bin/tesseract",                               # Linux system
    ]
    for _tess_path in _TESSERACT_SEARCH_PATHS:
        if shutil.which(_tess_path) or os.path.isfile(_tess_path):
            pytesseract.pytesseract.tesseract_cmd = _tess_path
            break

    def _tesseract_available() -> bool:
        """Return True if the Tesseract binary is reachable."""
        try:
            pytesseract.get_tesseract_version()
            return True
        except _TesseractNotFoundError:
            return False

except ImportError:
    pytesseract = None  # type: ignore[assignment]

    def _tesseract_available() -> bool:  # type: ignore[misc]
        return False


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class TileColor(NamedTuple):
    """The Wordle feedback colour for a single tile."""
    code: str          # 'g', 'y', or 'b'
    label: str         # 'green', 'yellow', or 'black'
    rgb: tuple[int, int, int]


# ---------------------------------------------------------------------------
# Colour classification
# ---------------------------------------------------------------------------

def _rgb_to_hsv_degrees(r: int, g: int, b: int) -> tuple[float, float, float]:
    """Return (hue_degrees, saturation_0_1, value_0_1) for an RGB triplet."""
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return h * 360, s, v


def classify_tile_color(rgb: tuple[int, int, int]) -> TileColor:
    """Classify a tile's centre pixel colour into green / yellow / black.

    Uses HSV so the classifier is robust to JPEG artefacts and theme variants
    (light mode, dark mode, high-contrast mode).

    Args:
        rgb: (R, G, B) tuple sampled from the tile centre.

    Returns:
        A :class:`TileColor` with ``code`` in ``{'g', 'y', 'b'}``.
    """
    r, g, b = rgb
    hue, sat, val = _rgb_to_hsv_degrees(r, g, b)

    # Require meaningful saturation AND brightness to distinguish coloured
    # tiles from grey/black ones.
    # - Saturation threshold is lowered to 0.15 because averaging a patch
    #   that includes white letter pixels can wash out saturation.
    # - Value (brightness) must be > 0.35 so dark-grey "absent" tiles
    #   (NYT dark: val≈0.24) are never mistaken for coloured tiles even
    #   if their hue happens to fall in the green/yellow range.
    if sat >= 0.15 and val > 0.35:
        # Green: standard NYT green (hue ~115°) and high-contrast cyan (~185°)
        if (100 <= hue <= 165) or (165 < hue <= 210 and sat >= 0.25):
            return TileColor("g", "green", rgb)
        # Yellow: standard NYT yellow (~55°) and high-contrast orange (~25°)
        if (15 <= hue <= 75):
            return TileColor("y", "yellow", rgb)

    # Everything else: absent / black tile
    return TileColor("b", "black", rgb)


# ---------------------------------------------------------------------------
# Grid detection
# ---------------------------------------------------------------------------

def _background_color(image: Image.Image) -> tuple[int, int, int]:
    """Estimate the background colour by taking the most common corner pixel.

    Samples the four corner regions (5×5 px each) and returns the median RGB.
    This works for both light-mode (white) and dark-mode (near-black) Wordle.
    """
    samples: list[tuple[int, int, int]] = []
    corners = [
        (0, 0), (image.width - 5, 0),
        (0, image.height - 5), (image.width - 5, image.height - 5),
    ]
    for cx, cy in corners:
        for dx in range(5):
            for dy in range(5):
                px = min(cx + dx, image.width - 1)
                py = min(cy + dy, image.height - 1)
                samples.append(image.getpixel((px, py))[:3])
    # Use channel-wise median
    samples.sort()
    mid = samples[len(samples) // 2]
    return mid


def _color_distance(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    """Euclidean distance between two RGB colours (0–441)."""
    return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2) ** 0.5


def _row_tile_score(
    image: Image.Image,
    y: int,
    bg: tuple[int, int, int],
    min_dist: float = 30.0,
) -> float:
    """Return the fraction of pixels in row ``y`` that differ from background.

    A high score means this row is likely inside a tile.  A low score means
    it is background or a gap between tiles.
    """
    total = image.width
    different = sum(
        1 for x in range(total)
        if _color_distance(image.getpixel((x, y))[:3], bg) >= min_dist
    )
    return different / total


def _find_played_bands(
    image: Image.Image,
    min_tile_height: int = 20,
) -> list[tuple[int, int]]:
    """Find contiguous vertical bands where tiles have been played (filled).

    A played row has many pixels that differ clearly from the background.
    Empty rows (not yet guessed) have only thin border pixels — their score
    is very low, so they are not returned here.

    Returns:
        List of (top, bottom) y-coordinate pairs for *played* tile rows only.
    """
    bg = _background_color(image)
    scores = [_row_tile_score(image, y, bg) for y in range(image.height)]

    # Threshold: >15 % of the row's pixels must differ from background.
    #
    # We use 15 % (not a higher value like 40 %) because on mobile screenshots
    # the Wordle grid can be a relatively narrow band in the centre of a full-
    # width image (e.g. 330 px of grid in a 1170 px wide screenshot = 28 %).
    # In dark mode the tile fill (grey ≈ (58,58,60) on bg ≈ (18,18,19)) gives a
    # colour-distance of ~70 — well above min_dist=30 — but only over the tile
    # fraction of the row.  A purely-grey row like "HEDER" (all absent tiles)
    # may therefore score ~28 %, which is above 15 % but below the old 40 %.
    #
    # Empty tiles in dark mode have only a very thin (~1-2 px) dark border whose
    # colour is barely different from the background; they score < 5 %, safely
    # below this threshold.
    threshold = 0.15
    bands: list[tuple[int, int]] = []
    in_band = False
    start = 0
    for y, s in enumerate(scores):
        if not in_band and s >= threshold:
            in_band = True
            start = y
        elif in_band and s < threshold:
            in_band = False
            if y - start >= min_tile_height:
                bands.append((start, y - 1))
    if in_band and image.height - start >= min_tile_height:
        bands.append((start, image.height - 1))
    return bands


def _find_tile_rows(
    image: Image.Image,
    min_tile_height: int = 20,
) -> list[tuple[int, int]]:
    """Detect all tile rows in the grid, including empty (unplayed) rows.

    Strategy
    --------
    1. Find all *played* bands (rows with solid tile fills) using colour-
       distance from background.
    2. Derive the tile height (``h``) and gap (``g``) from the spacing
       between consecutive played bands.  If there is only one played band
       (common mid-game), estimate tile height from the band's own height.
    3. Find the left/right extent of the grid from the played band(s).
    4. Walk the full grid — from the first played row upward to row 0 and
       downward to the bottom — projecting rows at pitch ``h + g``.
    5. Keep only rows whose centre-column pixel differs from background by
       at least a small amount (border or fill), so we don't return rows
       that are purely outside the grid.

    This correctly handles:
    - Light mode: all played tiles bright, empty tiles white (same as bg).
    - Dark mode: played tiles coloured/grey, empty tiles ≈ background with
      only a thin dark border.
    - Partial games (1–5 rows played out of 6).

    Args:
        image: PIL Image in RGB mode.
        min_tile_height: Minimum pixel height to count as a tile row.

    Returns:
        List of (top, bottom) pixel-row pairs for every tile row in the grid,
        ordered top to bottom.  Only rows that belong to the Wordle grid are
        returned; purely empty rows outside the grid are excluded.
    """
    played = _find_played_bands(image, min_tile_height)
    if not played:
        return []

    # --- Estimate tile height and pitch from played bands ---
    # Tile height: median height of played bands
    heights = [b - t + 1 for t, b in played]
    tile_h = sorted(heights)[len(heights) // 2]

    # Gap between bands: estimate from spacing between consecutive bands
    if len(played) >= 2:
        pitches = [played[i+1][0] - played[i][0] for i in range(len(played)-1)]
        pitch = sorted(pitches)[len(pitches) // 2]
        gap = max(0, pitch - tile_h)
    else:
        # Only one played band — gap is typically ~10 % of tile height on NYT
        gap = max(4, tile_h // 8)
        pitch = tile_h + gap

    # --- Find the column span from the first played band ---
    first_top, first_bottom = played[0]

    # --- Project all grid rows using the pitch, starting from first played row ---
    # Walk upward from first_top (in case earlier rows were played but missed)
    # and downward
    rows: list[tuple[int, int]] = []

    # Anchor: first played band
    anchor_top = first_top

    # Go upward from anchor
    t = anchor_top
    while t >= 0:
        b = t + tile_h - 1
        if b < image.height:
            rows.append((t, b))
        t -= pitch

    # Go downward from anchor (skip the anchor itself, already added)
    t = anchor_top + pitch
    while t + tile_h - 1 < image.height:
        rows.append((t, t + tile_h - 1))
        t += pitch

    # Sort top-to-bottom
    rows.sort()

    # --- Filter: keep only rows that are inside the visible grid ---
    # A row belongs to the grid if at least one pixel along its midline
    # differs from background (played fill, border, or letter ink).
    #
    # We scan the *full width* of the midline rather than only the centre
    # column, because on light-mode screenshots the centre column can fall
    # in a gap between tiles (all white) even though the tile borders are
    # clearly visible just a few pixels to either side.
    bg = _background_color(image)
    valid: list[tuple[int, int]] = []
    for top, bottom in rows:
        mid_y = (top + bottom) // 2
        if not (0 <= mid_y < image.height):
            continue
        # Sample evenly across the full width; accept the row as soon as any
        # pixel clearly differs from background (dist ≥ 5 catches thin borders).
        step = max(1, image.width // 100)
        in_grid = any(
            _color_distance(image.getpixel((x, mid_y))[:3], bg) >= 5
            for x in range(0, image.width, step)
        )
        if in_grid:
            valid.append((top, bottom))

    return valid


def _find_tile_cols(
    image: Image.Image,
    top: int,
    bottom: int,
    min_tile_width: int = 20,
    reference_cols: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Detect tile columns within a tile row band.

    Strategy (three-pass):

    1. **Solid-fill pass** — scan the midline of the band for contiguous runs
       of pixels that clearly differ from the background (threshold 15).  Any
       run that is at least ``min_tile_width`` pixels wide is recorded as a
       tile column.  This works for played rows whose tiles are coloured/grey.

    2. **Border-reconstruction pass** — when pass 1 finds no wide segments
       (e.g. a light-mode empty row where only thin 1–4 px borders differ from
       the white background), collect all narrow segments and pair consecutive
       ones that are separated by a gap consistent with being the left- and
       right-border of the same tile.  The reconstructed tile span runs from
       the left edge of the left border to the right edge of the right border.
       This handles both clean 2 px PNG borders and JPEG-blurred edges.

    3. **Reference fallback** — if both passes fail (e.g. a row whose entire
       midline is pure background), return ``reference_cols`` unchanged.

    Args:
        image: PIL Image in RGB mode.
        top: Top y-coordinate of the tile row.
        bottom: Bottom y-coordinate of the tile row.
        min_tile_width: Minimum pixel width to count as a tile column.
        reference_cols: Column layout from a played row, used as fallback
            when this row cannot be detected independently.

    Returns:
        List of (left, right) pixel-column pairs for each detected tile.
    """
    bg = _background_color(image)
    tile_h = bottom - top + 1

    # -----------------------------------------------------------------------
    # Build a column "presence" profile by sampling MULTIPLE horizontal scan
    # lines across the tile height, not just the midline.
    #
    # Scanning only the midline fails when a large bold letter (M, W, A, B…)
    # places a white stroke exactly at mid-height, creating false gaps.
    # By aggregating many scanlines, each column x gets a vote count equal to
    # the number of scan rows where it differs from the background.  Columns
    # that are part of a tile fill register votes on nearly every scan row,
    # while letter-stroke gaps only miss a few rows — so the profile stays
    # high across the entire tile width regardless of the letter shape.
    #
    # For empty tiles (border-only), the profile is near-zero everywhere
    # except at the thin left/right border columns.
    # -----------------------------------------------------------------------
    # Sample ~20 evenly-spaced rows within the tile, skipping top/bottom 10%
    # to avoid border pixels affecting the profile.
    margin_y = max(1, tile_h // 10)
    scan_ys = list(range(top + margin_y, bottom - margin_y + 1,
                         max(1, (tile_h - 2 * margin_y) // 20)))
    if not scan_ys:
        scan_ys = [(top + bottom) // 2]

    threshold = 15.0
    votes = [0] * image.width
    for y in scan_ys:
        for x in range(image.width):
            if _color_distance(image.getpixel((x, y))[:3], bg) >= threshold:
                votes[x] += 1

    # A column belongs to a tile if it has votes on at least 30% of scan rows
    min_votes = max(1, len(scan_ys) * 3 // 10)
    active = [v >= min_votes for v in votes]

    # -----------------------------------------------------------------------
    # Pass 1: extract contiguous active-column runs → tile segments
    # -----------------------------------------------------------------------
    segments: list[tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    for x, a in enumerate(active):
        if not in_seg and a:
            in_seg = True
            seg_start = x
        elif in_seg and not a:
            in_seg = False
            segments.append((seg_start, x - 1))
    if in_seg:
        segments.append((seg_start, image.width - 1))

    wide = [(l, r) for l, r in segments if r - l + 1 >= min_tile_width]

    # If we have the right number of tiles, return immediately.
    # If not (e.g. JPEG blur merges two adjacent tiles into one wide segment),
    # do NOT return yet — fall through to try the reference layout instead.
    expected = len(reference_cols) if reference_cols else None
    if wide and (expected is None or len(wide) == expected):
        return wide

    # -----------------------------------------------------------------------
    # Pass 1b: try splitting any over-wide segment using reference tile widths.
    #
    # This handles the case where two adjacent same-coloured tiles (e.g. two
    # green tiles side by side) get merged by JPEG compression into one wide
    # active run.  If we have a reference column layout we can check each
    # segment: if it is roughly twice the expected tile width, split it at
    # the midpoint.  We keep the result only if it yields the right count.
    # -----------------------------------------------------------------------
    if wide and reference_cols and len(wide) != expected:
        ref_widths = [r - l + 1 for l, r in reference_cols]
        avg_ref_w = sum(ref_widths) / len(ref_widths) if ref_widths else min_tile_width
        split_result: list[tuple[int, int]] = []
        for l, r in wide:
            w = r - l + 1
            # How many tiles does this segment span (rounded)?
            n_tiles = max(1, round(w / avg_ref_w))
            if n_tiles > 1:
                # Split evenly into n_tiles sub-segments
                sub_w = w // n_tiles
                for k in range(n_tiles):
                    sl = l + k * sub_w
                    sr = l + (k + 1) * sub_w - 1 if k < n_tiles - 1 else r
                    split_result.append((sl, sr))
            else:
                split_result.append((l, r))
        if len(split_result) == expected:
            return split_result

    # If we still have the wrong count but have ANY wide segments, fall
    # through to the reference fallback below rather than returning bad data.
    if wide and (expected is None):
        return wide

    # -----------------------------------------------------------------------
    # Pass 2: border reconstruction for empty light-mode rows.
    #
    # Empty tiles have only thin (1–4 px) border pixels that differ from the
    # white background; the interior is white.  The multi-row vote profile
    # will show narrow spikes at each border edge.  We pair consecutive
    # left-border and right-border spikes to reconstruct full tile spans.
    # -----------------------------------------------------------------------
    if segments:
        max_interior = image.width // 2
        reconstructed: list[tuple[int, int]] = []
        i = 0
        while i < len(segments):
            left_seg = segments[i]
            if i + 1 < len(segments):
                right_seg = segments[i + 1]
                gap = right_seg[0] - left_seg[1] - 1
                if gap <= max_interior:
                    reconstructed.append((left_seg[0], right_seg[1]))
                    i += 2
                    continue
            i += 1

        valid_reconstructed = [
            (l, r) for l, r in reconstructed if r - l + 1 >= min_tile_width
        ]
        if valid_reconstructed:
            return valid_reconstructed

    # -----------------------------------------------------------------------
    # Pass 3: fall back to reference layout (e.g. pure-white midline)
    # -----------------------------------------------------------------------
    if reference_cols:
        return reference_cols

    return []


def _sample_tile_color(
    image: Image.Image,
    left: int,
    right: int,
    top: int,
    bottom: int,
) -> tuple[int, int, int]:
    """Sample the dominant fill colour of a Wordle tile.

    Wordle tiles have a bold letter rendered in white at the centre, which
    dilutes the tile colour when averaging a dead-centre patch.  To avoid
    this, we sample four small patches near the **corners of the inner
    region** (25–40 % inset from each edge) where tile fill dominates, then
    return their combined average.

    Args:
        image: PIL Image in RGB mode.
        left, right, top, bottom: Pixel bounds of the tile.

    Returns:
        (R, G, B) average colour representative of the tile's fill.
    """
    w = right - left
    h = bottom - top

    # Inset 25 % from each edge to avoid borders, but stay away from centre
    ix = max(2, w // 4)
    iy = max(2, h // 4)

    # Four corner-region anchors (top-left, top-right, bottom-left, bottom-right)
    anchors = [
        (left + ix,     top + iy),
        (right - ix,    top + iy),
        (left + ix,     bottom - iy),
        (right - ix,    bottom - iy),
    ]

    patch = max(2, w // 8)
    patch_h = max(2, h // 8)

    r_total = g_total = b_total = count = 0
    for ax, ay in anchors:
        for py in range(ay - patch_h, ay + patch_h + 1):
            for px in range(ax - patch, ax + patch + 1):
                if 0 <= px < image.width and 0 <= py < image.height:
                    pixel = image.getpixel((px, py))
                    r_total += pixel[0]
                    g_total += pixel[1]
                    b_total += pixel[2]
                    count += 1
    if count == 0:
        return (0, 0, 0)
    return (r_total // count, g_total // count, b_total // count)


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def _ocr_tile_letter(
    image: Image.Image,
    left: int,
    right: int,
    top: int,
    bottom: int,
) -> str:
    """Extract the single letter shown on a Wordle tile using Tesseract OCR.

    Pre-processing steps applied to maximise accuracy on bold white-on-colour
    Wordle letters:

    1. Crop a slightly inset region (10 % margin) to avoid tile borders.
    2. Scale up to at least 80 × 80 px (Tesseract performs poorly on small glyphs).
    3. Convert to greyscale and invert so the letter is dark on a light background.
    4. Apply a slight sharpening pass to crisp up JPEG-compressed edges.
    5. Run Tesseract with ``--psm 10`` (single character) and
       ``tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ``.
    6. Return the first alphabetic character found, or ``'?'`` on failure.

    Args:
        image: Full PIL Image in RGB mode.
        left, right, top, bottom: Pixel bounds of the tile.

    Returns:
        Single uppercase letter string, or ``'?'`` if OCR failed or Tesseract
        is not installed.
    """
    if not _tesseract_available():
        return "?"

    w = right - left
    h = bottom - top

    # 1. Upscale the source image BEFORE cropping so that sub-pixel information
    #    in thin letter strokes (e.g. 'I') is preserved by LANCZOS resampling.
    #    We scale the whole image by a fixed integer factor (4×) so coordinate
    #    mapping stays exact and no floating-point rounding occurs.
    factor = 4
    scaled_img = image.resize(
        (image.width * factor, image.height * factor), Image.LANCZOS
    )

    # 2. Inset crop at the scaled coordinates — skip outer 10 % of the tile
    #    to avoid border artefacts; the inset is also scaled by the same factor.
    inset_x = max(factor, (w // 10) * factor)
    inset_y = max(factor, (h // 10) * factor)
    crop = scaled_img.crop((
        left  * factor + inset_x,
        top   * factor + inset_y,
        right * factor - inset_x,
        bottom * factor - inset_y,
    ))

    # 3. Greyscale + contrast stretch so the full 0-255 range is used.
    #    Wordle tiles have low contrast after JPEG/PNG compression and retina
    #    downscaling; stretching ensures the letter stands out regardless of
    #    tile colour.
    grey = crop.convert("L")
    pixels = list(grey.getdata())
    lo, hi = min(pixels), max(pixels)
    if hi > lo:
        grey = grey.point(lambda p: int((p - lo) * 255 / (hi - lo)))

    # 4. Invert so the letter (light on coloured tile) becomes dark on light bg,
    #    which is the orientation Tesseract expects.
    inverted = ImageOps.invert(grey)

    # 4b. Binarise with Otsu's method so the image is pure black/white.
    #     The contrast-stretched inverted image has a grey background (~200–230)
    #     and a near-black letter (~0–40). A simple fixed threshold of 128
    #     works well here and avoids Tesseract misreading E as I due to the
    #     grey midtone background confusing the anti-aliased letter edges.
    inv_pixels = list(inverted.getdata())
    # Compute Otsu threshold over the pixel histogram
    hist = [0] * 256
    for p in inv_pixels:
        hist[p] += 1
    total = len(inv_pixels)
    sum_all = sum(i * hist[i] for i in range(256))
    sum_bg = wb = 0
    otsu_thresh = 128  # sensible default
    max_var = 0.0
    for t in range(256):
        wb += hist[t]
        if wb == 0:
            continue
        wf = total - wb
        if wf == 0:
            break
        sum_bg += t * hist[t]
        mb = sum_bg / wb
        mf = (sum_all - sum_bg) / wf
        var = wb * wf * (mb - mf) ** 2
        if var > max_var:
            max_var = var
            otsu_thresh = t
    binarised = inverted.point(lambda p: 0 if p < otsu_thresh else 255)

    # 5. Sharpen
    sharpened = binarised.filter(ImageFilter.SHARPEN)

    # 5b. Add a white border (padding) around the image.
    #     Tesseract performs poorly when glyphs touch the image edge; a small
    #     margin of background colour gives it room to find the character.
    pad = max(8, sharpened.width // 8)
    padded = Image.new("L", (sharpened.width + 2 * pad, sharpened.height + 2 * pad), 255)
    padded.paste(sharpened, (pad, pad))
    sharpened = padded

    # Mapping of visually-similar non-alpha characters that Tesseract sometimes
    # produces instead of the correct letter (especially for thin glyphs like 'I').
    _SIMILAR: dict[str, str] = {
        "l": "I", "1": "I", "|": "I", "!": "I",
        "0": "O",
        "5": "S",
        "6": "G",
        "8": "B",
    }

    def _extract_letter(raw: str) -> str:
        """Return the first valid A-Z character from raw Tesseract output,
        applying visually-similar substitutions for common OCR confusions.

        Similarity map is checked *before* the generic alpha test so that
        characters like lowercase 'l' are mapped to 'I' rather than 'L'.
        """
        for ch in raw:
            if ch in _SIMILAR:
                return _SIMILAR[ch]
            if ch.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                return ch.upper()
        return ""

    # 5c. Shape-based 'I' detector.
    #
    # The Wordle sans-serif 'I' is a plain vertical bar with no serifs.
    # Tesseract frequently fails or returns '?' for it because the glyph is
    # ambiguous with 'l', '1', and '|'.  Rather than relying on OCR alone,
    # we detect 'I' by shape: count the black pixels in each column of the
    # binarised image.  If there is exactly ONE contiguous group of columns
    # that contains many black pixels (letter ink), and that group is narrow
    # relative to the image width (< 30 %), the letter is almost certainly 'I'.
    bw_pixels = list(sharpened.getdata())
    bw_w, bw_h = sharpened.width, sharpened.height
    col_black = [0] * bw_w
    for idx, p in enumerate(bw_pixels):
        if p < 128:
            col_black[idx % bw_w] += 1
    # Columns with > 5 % of rows filled = part of a stroke
    min_stroke = max(1, bw_h // 20)
    stroke_cols = [x for x, c in enumerate(col_black) if c >= min_stroke]
    if stroke_cols:
        groups: list[tuple[int, int]] = []
        g_start = stroke_cols[0]
        g_prev  = stroke_cols[0]
        for x in stroke_cols[1:]:
            if x - g_prev > 3:
                groups.append((g_start, g_prev))
                g_start = x
            g_prev = x
        groups.append((g_start, g_prev))
        # Single narrow group centred in the image → 'I'
        if len(groups) == 1:
            g_l, g_r = groups[0]
            g_w = g_r - g_l + 1
            centre_x = (g_l + g_r) / 2
            img_centre = bw_w / 2
            # Use a tight threshold: I has ~8% stroke width; the next narrowest
            # letter (E/L) is ~24%.  15% safely separates them.
            if g_w < bw_w * 0.15 and abs(centre_x - img_centre) < bw_w * 0.20:
                return "I"

    # 6. Tesseract — multi-stage strategy, most-specific first:
    #    Stages (a)-(b): whitelist A-Z, psm 10 then 8. Catches most letters.
    #    Stages (c)-(d): no whitelist, psm 10 then 8. _extract_letter maps
    #       'l'/'1'/'|' → 'I' so sans-serif I is caught even when the whitelist
    #       blocks all output.
    #    Stages (e)-(f): extend the whitelist to include common look-alike
    #       digits/punctuation so Tesseract is not forced to output nothing.
    #       This is the last resort for very thin glyphs (I, l, 1).
    whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    extended  = whitelist + "l1|!"   # let Tesseract emit look-alikes for I
    tess_letter = ""
    for config in (
        f"--psm 10 -c tessedit_char_whitelist={whitelist}",
        f"--psm 8  -c tessedit_char_whitelist={whitelist}",
        "--psm 10",                                          # no whitelist
        "--psm 8",                                           # no whitelist
        f"--psm 10 -c tessedit_char_whitelist={extended}",  # extended whitelist
        f"--psm 7  -c tessedit_char_whitelist={extended}",  # single-line mode
    ):
        try:
            raw = pytesseract.image_to_string(sharpened, config=config).strip()
        except Exception:
            continue
        found = _extract_letter(raw)
        if found:
            tess_letter = found
            break

    # 6b. M ↔ N/I disambiguation.
    # Tesseract sometimes misreads the bold Wordle 'M' as 'N' (older builds)
    # or as 'I' (observed on Tesseract 5.5.0 in the Railway deployment).
    # Both confusions are caught by the same centre-of-mass symmetry test:
    # M is left-right symmetric (per-row CoM is constant), while N has a
    # diagonal stroke that causes the CoM to drift linearly top-to-bottom.
    #
    # For the M→I case we add a stroke-span guard so that a genuine 'I'
    # (narrow single-column stroke, already handled by the shape detector
    # above and therefore never reaching this point with wide span) is never
    # promoted to 'M'.  A true 'I' has stroke_cols spanning < 10 % of the
    # image width; an 'M' misread as 'I' spans > 30 %.  The 20 % threshold
    # gives a comfortable safety margin.
    #
    # Fixture measurements:  M ≈ 0.001,  N ≈ 0.083  →  threshold 0.05 safe.
    _m_i_candidate = (
        tess_letter == "I"
        and bool(stroke_cols)
        and (stroke_cols[-1] - stroke_cols[0] + 1) > bw_w * 0.20
    )
    if tess_letter == "N" or _m_i_candidate:
        row_coms: list[tuple[int, float]] = []
        for row in range(bw_h):
            row_px = bw_pixels[row * bw_w : (row + 1) * bw_w]
            blacks = [x for x, p in enumerate(row_px) if p < 128]
            if len(blacks) >= 3:
                row_coms.append((row, sum(blacks) / len(blacks)))
        if len(row_coms) >= 5:
            rows_v = [r for r, _ in row_coms]
            coms_v = [c for _, c in row_coms]
            n_r = len(rows_v)
            mr = sum(rows_v) / n_r
            mc = sum(coms_v) / n_r
            slope_num = sum((r - mr) * (c - mc) for r, c in zip(rows_v, coms_v))
            slope_den = sum((r - mr) ** 2 for r in rows_v)
            if slope_den > 0:
                slope = slope_num / slope_den
                total_drift = abs(slope) * bw_h / bw_w
                if total_drift < 0.05:
                    tess_letter = "M"

    # 6c. O ↔ C disambiguation.
    # Tesseract sometimes misreads the bold Wordle 'O' as 'C' because both
    # glyphs are circular arcs.  The key difference is closure: 'O' is a
    # complete ring — its right side has dark pixels all the way through
    # the equator.  The bold Wordle 'C' has two tips (top and bottom) that
    # extend to the right, but its equatorial band is empty (the opening).
    #
    # We use stroke_cols (already computed for the I-detector) to locate the
    # right edge of the letter, then sample the rightmost 25 % of its
    # horizontal extent inside the *equatorial* row band (42–58 % of the
    # padded image height).  That narrow band captures the gap of C but the
    # solid arc of O.
    #
    # Measured densities (equatorial, right-25 %) across all fixtures:
    #   Real O (misread as C):  0.95   →  should be corrected to O
    #   Real C (various fonts): 0.21 – 0.40  →  should stay as C
    #   Threshold 0.80 for C→O (was 0.60): Linux/Docker LANCZOS resampling
    #   produces slightly more blur at C's right opening, pushing density into
    #   the 0.60–0.75 range for genuine C tiles on deployed Railway.  Real O
    #   measures 0.90+ so 0.80 still gives a clear gap.
    #   Threshold 0.60 kept for O→C: unchanged, gives ample separation.
    #   Equatorial band widened to 38–62 % (was 42–58 %) for a larger, more
    #   stable sample that is less sensitive to per-pixel rendering differences.
    if tess_letter in ("C", "O") and stroke_cols:
        _co_right = stroke_cols[-1]
        _co_lw    = _co_right - stroke_cols[0] + 1    # letter width in px
        _co_r0    = int(_co_right - _co_lw * 0.25)    # rightmost 25 % of letter
        _co_r1    = _co_right + 1
        _co_m0    = int(bw_h * 0.38)                  # equatorial band start (widened)
        _co_m1    = int(bw_h * 0.62)                  # equatorial band end   (widened)
        _co_dark = _co_total = 0
        for _row in range(_co_m0, _co_m1):
            for _col in range(_co_r0, _co_r1):
                _co_total += 1
                if bw_pixels[_row * bw_w + _col] < 128:
                    _co_dark += 1
        if _co_total > 0:
            _co_density = _co_dark / _co_total
            # Density-based final decision — ignores Tesseract's C/O reading
            # entirely so the result is the same regardless of which letter
            # Tesseract returned first.
            #   Real O: density ≥ 0.85  →  "O"
            #   Real C: density < 0.80  →  "C"  (even if Tesseract said "O")
            # The gap between 0.75 (Docker C upper bound) and 0.90 (O lower
            # bound) is wide enough that 0.80 cleanly separates them.
            tess_letter = "O" if _co_density >= 0.80 else "C"

    # 6d. F ↔ E disambiguation.
    # Cloud Tesseract 5.x sometimes misreads the bold Wordle 'F' as 'E'.
    # Both glyphs share a vertical stroke and two horizontal bars (top and
    # middle), but only 'E' has a third horizontal bar at the bottom.
    #
    # We check dark-pixel density in the lower-middle row band (55–75 %) of
    # the padded image, in the right half of the letter extent.  A genuine
    # 'E' has dense dark pixels there (its bottom bar); a genuine 'F' has
    # none (its stem ends after the middle bar).
    #
    # Measured densities (rows 55–75 %, right half of letter):
    #   E ≈ 0.45+,  F ≈ 0.00  →  threshold 0.10 gives ample margin.
    if tess_letter == "E" and stroke_cols:
        _ef_left  = stroke_cols[0]
        _ef_right = stroke_cols[-1]
        _ef_lw    = _ef_right - _ef_left + 1
        _ef_c0    = int(_ef_left + _ef_lw * 0.50)   # right half of letter
        _ef_c1    = _ef_right + 1
        _ef_m0    = int(bw_h * 0.55)                 # lower-middle rows
        _ef_m1    = int(bw_h * 0.75)
        _ef_dark = _ef_total = 0
        for _row in range(_ef_m0, _ef_m1):
            for _col in range(_ef_c0, _ef_c1):
                _ef_total += 1
                if bw_pixels[_row * bw_w + _col] < 128:
                    _ef_dark += 1
        if _ef_total > 0 and _ef_dark / _ef_total < 0.10:
            tess_letter = "F"

    return tess_letter or "?"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_wordle_image(
    image_path: str | Path,
    words: list[str] | None = None,
    expected_cols: int = 5,
) -> list[tuple[str, str]]:
    """Parse a Wordle screenshot and extract (word, response) pairs.

    The function auto-detects the grid of tiles, samples each tile's colour
    (green / yellow / black), and — when Tesseract is available — reads the
    letter on each tile via OCR.

    Letter resolution priority:
      1. ``words`` argument — explicit words supplied by the caller (most
         reliable; skips OCR entirely).
      2. Tesseract OCR — attempted automatically when ``words`` is ``None``
         and the Tesseract binary is installed.
      3. ``'?'`` placeholders — used per-tile when OCR fails or is unavailable.

    Args:
        image_path: Path to the screenshot (PNG, JPEG, WebP, etc.).
        words: Optional list of guessed words in order (e.g. ``['crane',
            'audio']``).  When supplied the response strings are paired with
            these words and OCR is skipped.  When omitted, OCR is attempted;
            if Tesseract is not installed each word is ``'?????'``.
        expected_cols: Expected number of tiles per row (default 5 for standard
            Wordle).  Used to filter out spurious rows with a different column
            count.

    Returns:
        List of ``(word, response)`` tuples where ``response`` is a 5-character
        string of ``'g'``/``'y'``/``'b'`` characters, e.g.::

            [('crane', 'bgybb'), ('audio', 'ybbgb')]

        If OCR is unavailable, words will be ``'?????'`` (or contain ``'?'``
        for individual letters that could not be read).

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If no Wordle grid could be detected in the image, or if
            ``words`` is supplied but its length does not match the number of
            detected rows.

    Example::

        >>> guesses = parse_wordle_image('my_wordle.png', words=['crane', 'slate'])
        >>> for word, resp in guesses:
        ...     print(f'{word},{resp}')
        crane,bgybb
        slate,gbgbg
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    image = Image.open(path).convert("RGB")

    # --- Detect tile rows ---
    tile_rows = _find_tile_rows(image)
    if not tile_rows:
        return []  # empty / unplayed grid — no guesses yet

    # --- For each tile row, detect columns, sample colours, and OCR letters ---
    use_ocr = (words is None)
    responses: list[str] = []
    ocr_words: list[str] = []
    bg = _background_color(image)

    # Build a reference column layout from the first row that self-detects
    # correctly (a played row). Empty rows reuse this layout.
    reference_cols: list[tuple[int, int]] | None = None

    for top, bottom in tile_rows:
        cols = _find_tile_cols(image, top, bottom, reference_cols=reference_cols)
        # Only keep rows that match the expected tile count
        if len(cols) != expected_cols:
            continue

        # Update reference from any well-detected played row
        if reference_cols is None or len(cols) == expected_cols:
            reference_cols = cols

        # Skip rows where the tile centres are all background-coloured
        # (i.e. empty/unplayed rows — nothing to extract from them)
        centre_dists = [
            _color_distance(_sample_tile_color(image, l, r, top, bottom), bg)
            for l, r in cols
        ]
        if all(d < 20 for d in centre_dists):
            continue  # empty row — no colour/letter data

        row_response = ""
        row_word = ""
        for left, right in cols:
            rgb = _sample_tile_color(image, left, right, top, bottom)
            tile = classify_tile_color(rgb)
            row_response += tile.code
            if use_ocr:
                row_word += _ocr_tile_letter(image, left, right, top, bottom).lower()
        responses.append(row_response)
        if use_ocr:
            ocr_words.append(row_word)

    if not responses:
        return []  # all detected rows were empty/unplayed — no guesses yet

    # --- Resolve final word list ---
    if words is not None:
        if len(words) != len(responses):
            raise ValueError(
                f"Supplied {len(words)} word(s) but detected {len(responses)} "
                f"tile row(s) in the image."
            )
        paired_words = [w.lower().strip() for w in words]
    else:
        paired_words = ocr_words  # may contain '?' chars if OCR failed

    return list(zip(paired_words, responses))


# ---------------------------------------------------------------------------
# Diagnostic helper
# ---------------------------------------------------------------------------

def describe_image_grid(
    image_path: str | Path,
    expected_cols: int = 5,
) -> None:
    """Print a diagnostic description of the detected grid (for debugging).

    Shows the detected tile rows/columns and the sampled RGB + classified
    colour for each tile.

    Args:
        image_path: Path to the screenshot.
        expected_cols: Expected tiles per row.
    """
    path = Path(image_path)
    image = Image.open(path).convert("RGB")
    bg = _background_color(image)
    print(f"Image size: {image.width} × {image.height} px")
    print(f"Background colour (estimated): rgb{bg}")

    tile_rows = _find_tile_rows(image)
    print(f"Detected {len(tile_rows)} candidate tile row(s).")

    reference_cols: list[tuple[int, int]] | None = None
    for row_idx, (top, bottom) in enumerate(tile_rows):
        cols = _find_tile_cols(image, top, bottom, reference_cols=reference_cols)
        if len(cols) == expected_cols:
            reference_cols = cols
        marker = "" if len(cols) == expected_cols else f"  ← skipped ({len(cols)} cols)"
        print(f"\n  Row {row_idx + 1}: y={top}–{bottom}, {len(cols)} tile(s){marker}")
        if len(cols) != expected_cols:
            continue
        for col_idx, (left, right) in enumerate(cols):
            rgb = _sample_tile_color(image, left, right, top, bottom)
            tile = classify_tile_color(rgb)
            h, s, v = _rgb_to_hsv_degrees(*rgb)
            letter = _ocr_tile_letter(image, left, right, top, bottom)
            ocr_note = f"  letter='{letter}'" if _tesseract_available() else "  (OCR unavailable)"
            print(
                f"    Tile {col_idx + 1}: rgb={rgb}  "
                f"hue={h:.0f}° sat={s:.2f} val={v:.2f}  "
                f"→ {tile.label} ({tile.code}){ocr_note}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse a Wordle screenshot, print the identified guesses, "
            "and (by default) suggest the best next guess using entropy scoring. "
            "Response chars: g=green, y=yellow, b=black."
        )
    )
    parser.add_argument("image", help="Path to the Wordle screenshot")
    parser.add_argument(
        "--words", nargs="*", default=None,
        help="Guessed words in order (e.g. crane audio). "
             "If omitted, letters are extracted via OCR (requires Tesseract).",
    )
    parser.add_argument(
        "--cols", type=int, default=5,
        help="Expected number of tiles per row (default: 5)",
    )
    parser.add_argument(
        "--diagnose", action="store_true",
        help="Print a detailed grid diagnostic instead of the guess list",
    )
    parser.add_argument(
        "--no-suggest", action="store_true",
        help="Skip the optimal-guess suggestion step; just print the identified guesses.",
    )
    parser.add_argument(
        "--wordlist", default="/usr/share/dict/words",
        help="Word list for the optimal-guess search (default: /usr/share/dict/words)",
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="Number of top suggestions to display (default: 5)",
    )
    args = parser.parse_args()

    if args.diagnose:
        describe_image_grid(args.image, expected_cols=args.cols)
        return

    try:
        guesses = parse_wordle_image(args.image, words=args.words, expected_cols=args.cols)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- Print identified guesses ---
    print("Guesses identified from image:")
    for word, response in guesses:
        print(f"  {word},{response}")
    sys.stdout.flush()

    if args.no_suggest:
        return

    # --- Run optimal-guess suggestion ---
    # Import and call the optimal_guess main() logic directly so that output
    # ordering is preserved (no subprocess buffering race).
    print()  # blank line separator
    sys.stdout.flush()

    # Build a sys.argv that optimal_guess.main() will parse.
    import optimal_guess as _og  # noqa: PLC0415
    orig_argv = sys.argv[:]
    sys.argv = [
        "optimal_guess.py",
        "--top", str(args.top),
        "--wordlist", args.wordlist,
    ]
    for word, response in guesses:
        sys.argv += ["--guess", f"{word},{response}"]
    try:
        _og.main()
    finally:
        sys.argv = orig_argv


if __name__ == "__main__":
    main()

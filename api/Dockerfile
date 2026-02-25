# syntax=docker/dockerfile:1
FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────────────────────
# tesseract-ocr        — OCR engine binary
# tesseract-ocr-eng    — English language data pack
# libgl1               — Pillow indirect dependency on slim images
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────────────
WORKDIR /app
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ───────────────────────────────────────────────────────
# Shared Python modules from the repo root
COPY wordle_filter.py  ./wordle_filter.py
COPY wordle_image.py   ./wordle_image.py
COPY optimal_guess.py  ./optimal_guess.py
COPY main.py           ./main.py
COPY wordlist.txt      ./wordlist.txt

# Test suite and fixtures
COPY conftest.py           ./conftest.py
COPY test_wordle_image.py  ./test_wordle_image.py
COPY test_analyze_api.py   ./test_analyze_api.py
COPY tests/                ./tests/

# API package (copied last — only invalidates cache on API changes)
COPY api/main.py       ./api/main.py

# ── Runtime configuration ────────────────────────────────────────────────────
# Tell pytesseract where the Debian tessdata lives
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Railway injects PORT at runtime; default to 8000 for local docker run
ENV PORT=8000
EXPOSE 8000

# ── Entrypoint ───────────────────────────────────────────────────────────────
# --app-dir=. puts /app on sys.path so `from wordle_filter import ...` works
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --app-dir ."]

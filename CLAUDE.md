# CLAUDE.md

## Project Overview

A collection of Python utility scripts:

- **Wordle helper** (`main.py`, `wordle_filter.py`) — CLI tool that filters 5-letter words using Wordle-style constraints (green/yellow/max occurrences). Uses `/usr/share/dict/words` by default.
- **Prime factorization API** (`prime_api.py`) — HTTP server (stdlib `http.server`) exposing `GET /factorize?n=<number>`.
- **Prime client** (`prime_client.py`) — CLI client for the prime API.
- **Optimal Wordle guess** (`optimal_guess.py`) — Finds the best first Wordle guess by maximising Shannon entropy of the feedback pattern distribution. Uses multiprocessing for speed.
- **Wordle image parser** (`wordle_image.py`) — Extracts past guesses and tile colours (g/y/b) from a Wordle screenshot using Pillow pixel analysis.

## Tech Stack

- Python 3.10+ (uses `X | Y` union syntax)
- Stdlib only, **except**:
  - `Pillow` (`pip install Pillow`) — required by `wordle_image.py` for screenshot parsing
  - `pytesseract` (`pip install pytesseract`) + Tesseract binary (`brew install tesseract`) — optional; enables automatic letter extraction from screenshots in `wordle_image.py`

## Running

```sh
# Wordle helper
python main.py --green 1=r 3=a --yellow s=0,2 --max e=1

# Prime API server
python prime_api.py 8080

# Prime client
python prime_client.py localhost 8080 360

# Optimal Wordle guess finder
python optimal_guess.py --top 10
python optimal_guess.py --wordlist /path/to/words.txt --workers 4
python optimal_guess.py --image screenshot.png

# Wordle image parser (standalone)
python wordle_image.py screenshot.png
```

## Conventions

- Type hints on function signatures
- Module-level docstrings with usage examples
- `argparse` for all CLI interfaces

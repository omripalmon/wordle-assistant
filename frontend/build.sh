#!/usr/bin/env bash
# build.sh — injected at Vercel build time.
# Replaces the __RAILWAY_API_URL__ placeholder in every HTML file with the
# value of the RAILWAY_API_URL environment variable.
#
# Vercel sets this variable per environment:
#   Production (production branch) → Railway production URL
#   Preview    (main branch)       → Railway staging URL
set -euo pipefail

: "${RAILWAY_API_URL:?RAILWAY_API_URL env var is required}"

for f in index.html test_image.html tests.html fixtures.html; do
  [ -f "$f" ] || continue
  sed -i "s|__RAILWAY_API_URL__|${RAILWAY_API_URL}|g" "$f"
  echo "✓ $f → ${RAILWAY_API_URL}"
done

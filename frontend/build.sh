#!/usr/bin/env bash
# build.sh — injected at Vercel build time.
# Replaces URL/config placeholders in every HTML file with values from
# environment variables.
#
# Required:
#   RAILWAY_API_URL       — Railway API for the current environment
#                           (staging on main branch, production on production branch)
#
# Optional — used by status.html; the dashboard degrades gracefully if unset:
#   RAILWAY_PROD_URL      — Railway production API URL
#   RAILWAY_STAGING_URL   — Railway staging API URL
#   VERCEL_PROD_URL       — Vercel production frontend URL
#   VERCEL_STAGING_URL    — Vercel staging (preview) frontend URL
#   VERCEL_DASHBOARD_URL  — Vercel project dashboard URL
#   RAILWAY_DASHBOARD_URL — Railway project dashboard URL
set -euo pipefail

: "${RAILWAY_API_URL:?RAILWAY_API_URL env var is required}"

RAILWAY_PROD_URL="${RAILWAY_PROD_URL:-}"
RAILWAY_STAGING_URL="${RAILWAY_STAGING_URL:-}"
VERCEL_PROD_URL="${VERCEL_PROD_URL:-}"
VERCEL_STAGING_URL="${VERCEL_STAGING_URL:-}"
VERCEL_DASHBOARD_URL="${VERCEL_DASHBOARD_URL:-}"
RAILWAY_DASHBOARD_URL="${RAILWAY_DASHBOARD_URL:-}"

for f in index.html test_image.html tests.html fixtures.html status.html; do
  [ -f "$f" ] || continue
  sed -i "s|__RAILWAY_API_URL__|${RAILWAY_API_URL}|g"           "$f"
  sed -i "s|__RAILWAY_PROD_URL__|${RAILWAY_PROD_URL}|g"         "$f"
  sed -i "s|__RAILWAY_STAGING_URL__|${RAILWAY_STAGING_URL}|g"   "$f"
  sed -i "s|__VERCEL_PROD_URL__|${VERCEL_PROD_URL}|g"           "$f"
  sed -i "s|__VERCEL_STAGING_URL__|${VERCEL_STAGING_URL}|g"     "$f"
  sed -i "s|__VERCEL_DASHBOARD_URL__|${VERCEL_DASHBOARD_URL}|g" "$f"
  sed -i "s|__RAILWAY_DASHBOARD_URL__|${RAILWAY_DASHBOARD_URL}|g" "$f"
  echo "✓ $f → processed"
done

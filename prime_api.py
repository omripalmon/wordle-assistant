"""REST API server that returns prime factorizations.

Usage:
    python prime_api.py 8080
"""

import argparse
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


def prime_factorization(n: int) -> list[dict]:
    """Return prime factorization as a list of {prime, power} dicts, ordered by prime."""
    if n < 2:
        return []
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return [{"prime": p, "power": factors[p]} for p in sorted(factors)]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/factorize":
            self.send_error(404, "Use GET /factorize?n=<number>")
            return

        params = parse_qs(parsed.query)
        raw = params.get("n", [None])[0]
        if raw is None:
            self.send_error(400, "Missing query parameter 'n'")
            return

        try:
            n = int(raw)
        except ValueError:
            self.send_error(400, f"'{raw}' is not a valid integer")
            return

        if n < 1:
            self.send_error(400, "n must be a positive integer")
            return

        result = {"number": n, "factors": prime_factorization(n)}
        body = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="Prime factorization REST API")
    parser.add_argument("port", type=int, help="Port to listen on")
    args = parser.parse_args()

    server = HTTPServer(("", args.port), Handler)
    print(f"Listening on port {args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()


if __name__ == "__main__":
    main()

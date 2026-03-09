"""Tests for prime_api.py."""
import json
import threading
import urllib.request
import urllib.error

import pytest
from prime_api import prime_factorization, Handler, HTTPServer


# -- prime_factorization unit tests --

class TestPrimeFactorization:
    def test_zero(self):
        assert prime_factorization(0) == []

    def test_one(self):
        assert prime_factorization(1) == []

    def test_prime(self):
        assert prime_factorization(7) == [{"prime": 7, "power": 1}]

    def test_small_composite(self):
        assert prime_factorization(12) == [
            {"prime": 2, "power": 2},
            {"prime": 3, "power": 1},
        ]

    def test_perfect_power(self):
        assert prime_factorization(8) == [{"prime": 2, "power": 3}]

    def test_large_composite(self):
        # 360 = 2^3 * 3^2 * 5
        assert prime_factorization(360) == [
            {"prime": 2, "power": 3},
            {"prime": 3, "power": 2},
            {"prime": 5, "power": 1},
        ]

    def test_large_prime(self):
        assert prime_factorization(7919) == [{"prime": 7919, "power": 1}]

    def test_two(self):
        assert prime_factorization(2) == [{"prime": 2, "power": 1}]

    def test_product_of_two_primes(self):
        # 15 = 3 * 5
        assert prime_factorization(15) == [
            {"prime": 3, "power": 1},
            {"prime": 5, "power": 1},
        ]


# -- HTTP handler integration tests --

@pytest.fixture(scope="module")
def server():
    """Start a test server on a random port and tear it down after tests."""
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


def _get_json(url):
    with urllib.request.urlopen(url) as resp:
        return resp.status, json.loads(resp.read())


class TestHTTPHandler:
    def test_factorize_valid(self, server):
        status, data = _get_json(f"{server}/factorize?n=12")
        assert status == 200
        assert data["number"] == 12
        assert data["factors"] == [
            {"prime": 2, "power": 2},
            {"prime": 3, "power": 1},
        ]

    def test_factorize_prime(self, server):
        status, data = _get_json(f"{server}/factorize?n=7")
        assert status == 200
        assert data["factors"] == [{"prime": 7, "power": 1}]

    def test_factorize_one(self, server):
        status, data = _get_json(f"{server}/factorize?n=1")
        assert status == 200
        assert data["factors"] == []

    def test_missing_n(self, server):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{server}/factorize")
        assert exc_info.value.code == 400

    def test_invalid_n(self, server):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{server}/factorize?n=abc")
        assert exc_info.value.code == 400

    def test_negative_n(self, server):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{server}/factorize?n=-1")
        assert exc_info.value.code == 400

    def test_unknown_path(self, server):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{server}/unknown")
        assert exc_info.value.code == 404

"""Tests for main.py CLI argument parsers."""
from main import parse_green, parse_yellow, parse_max


class TestParseGreen:
    def test_single(self):
        assert parse_green(["1=r"]) == {1: "r"}

    def test_multiple(self):
        assert parse_green(["1=r", "3=a"]) == {1: "r", 3: "a"}

    def test_uppercase_lowered(self):
        assert parse_green(["0=R"]) == {0: "r"}

    def test_empty(self):
        assert parse_green([]) == {}


class TestParseYellow:
    def test_single_position(self):
        assert parse_yellow(["e=4"]) == {"e": {4}}

    def test_multiple_positions(self):
        assert parse_yellow(["s=0,2"]) == {"s": {0, 2}}

    def test_multiple_letters(self):
        result = parse_yellow(["s=0,2", "e=4"])
        assert result == {"s": {0, 2}, "e": {4}}

    def test_uppercase_lowered(self):
        assert parse_yellow(["S=0"]) == {"s": {0}}

    def test_empty(self):
        assert parse_yellow([]) == {}


class TestParseMax:
    def test_single(self):
        assert parse_max(["e=1"]) == {"e": 1}

    def test_multiple(self):
        assert parse_max(["e=1", "s=1"]) == {"e": 1, "s": 1}

    def test_zero(self):
        assert parse_max(["x=0"]) == {"x": 0}

    def test_uppercase_lowered(self):
        assert parse_max(["E=2"]) == {"e": 2}

    def test_empty(self):
        assert parse_max([]) == {}

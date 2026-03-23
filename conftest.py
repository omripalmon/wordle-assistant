"""pytest root configuration.

- Ensures the project root is on sys.path.
- Reads known-bugs.json and automatically marks matching tests as xfail so that
  CI treats them as expected failures rather than blocking promotions.

  To mark a test as a known bug, add an entry to known-bugs.json:
      {
        "test_id": "test_wordle_image.py::test_parse_image[cruet_light.png]",
        "reason":  "Light-mode colour detection off on high-DPI screens",
        "issue":   "https://github.com/omripalmon/wordle-assistant/issues/42"
      }
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Known-bug xfail injection
# ---------------------------------------------------------------------------
_KNOWN_BUGS_FILE = ROOT / "known-bugs.json"


def _load_known_bugs() -> dict[str, dict]:
    """Return {test_id: entry} from known-bugs.json, or {} on any error."""
    try:
        data = json.loads(_KNOWN_BUGS_FILE.read_text())
        return {
            entry["test_id"]: entry
            for entry in data.get("known_bugs", [])
            if "test_id" in entry
        }
    except Exception:
        return {}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Attach xfail markers to any test listed in known-bugs.json."""
    known = _load_known_bugs()
    if not known:
        return

    for item in items:
        # item.nodeid is e.g. "test_wordle_image.py::test_parse_image[cruet_light.png]"
        entry = known.get(item.nodeid)
        if entry is None:
            continue
        reason = entry.get("reason", "known bug")
        issue  = entry.get("issue", "")
        full_reason = f"{reason} — {issue}" if issue else reason
        item.add_marker(pytest.mark.xfail(reason=full_reason, strict=False))

"""The two ladders must agree, so this checks them against each other.

The classification exists twice: once in `agent/ladder.py`, which is
authoritative, and once in JavaScript inside `frontend/index.html`, so the
sliders respond without a round trip. Two copies of the rule that allocates
human oversight is a real hazard — they can drift without anything failing, and
the failure mode is that the page shows one control tier while the server acts
on another.

So this test does not compare Python against a transcription of the JavaScript.
It extracts `classify` **from the shipped page**, runs it under `node`, and
compares every possible input. A copy would prove nothing; the page is what
users see.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.ladder import classify  # noqa: E402

PAGE = Path(__file__).resolve().parents[1] / "frontend" / "index.html"

#: Every combination the controls can produce. Harm and reversibility are 1-10
#: sliders, rights is a two-way select, and confidence only matters either side
#: of 70 -- the boundary plus both extremes covers it.
HARMS = range(1, 11)
REVERSIBILITIES = range(1, 11)
RIGHTS = (False, True)
CONFIDENCES = (0, 40, 69, 70, 100)


def _extract_js_classify() -> str:
    """Pull the `classify` function out of the page, verbatim."""
    source = PAGE.read_text(encoding="utf-8")
    match = re.search(
        r"function classify\(.*?\n\}", source, flags=re.S
    )
    if match is None:
        pytest.fail(
            f"no `function classify` found in {PAGE.name}. If the page stopped "
            f"computing the ladder client-side, delete this test; if it was "
            f"renamed, fix the pattern. Do not let it silently pass."
        )
    return match.group(0)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_page_and_the_server_classify_identically() -> None:
    js = _extract_js_classify()
    cases = [
        [h, r, int(rights), c]
        for h in HARMS
        for r in REVERSIBILITIES
        for rights in RIGHTS
        for c in CONFIDENCES
    ]

    # The page's `classify` returns a TIER object; only its `key` is compared,
    # so the wording of the labels can differ without failing this test.
    script = f"""
    const TIER = {{
      only:{{key:'human_only'}}, app:{{key:'human_approval'}},
      act:{{key:'ai_with_oversight'}}, rec:{{key:'ai_recommends'}}
    }};
    {js}
    const cases = {json.dumps(cases)};
    console.log(JSON.stringify(
      cases.map(([h, r, rt, c]) => classify(h, r, !!rt, c).key)
    ));
    """
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, f"node failed:\n{result.stderr}"
    from_page = json.loads(result.stdout)
    from_server = [classify(h, r, bool(rt), c) for h, r, rt, c in cases]

    assert len(from_page) == len(cases) == 1000
    disagreements = [
        (case, a, b)
        for case, a, b in zip(cases, from_page, from_server)
        if a != b
    ]
    assert not disagreements, (
        f"{len(disagreements)} of {len(cases)} inputs classify differently in "
        f"the page and on the server. First five: {disagreements[:5]}"
    )


def test_the_page_still_classifies_client_side() -> None:
    """If this fails, the parity test above has quietly stopped testing anything."""
    assert "function classify" in PAGE.read_text(encoding="utf-8")

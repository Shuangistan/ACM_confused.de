"""Does this machine have what it needs? Run before blaming the code.

    python check_setup.py

Every check prints what it looked for, so a failure names the fix rather than
the symptom. The last check spends about 20 tokens on a real call, because
everything up to that point can pass on a key that has no credit.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OK, BAD, WARN = "  [ ok ]", "  [FAIL]", "  [warn]"
failed = 0


def report(good: bool, label: str, detail: str = "", fatal: bool = True,
           only_if_bad: bool = False) -> bool:
    """`only_if_bad` suppresses the detail on success.

    Without it a passing check prints the explanation written for its failure,
    which reads as a contradiction: "[ok] model reachable — not in this
    account's catalogue".
    """
    global failed
    mark = OK if good else (BAD if fatal else WARN)
    show = detail and not (good and only_if_bad)
    print(f"{mark} {label}" + (f" — {detail}" if show else ""))
    if not good and fatal:
        failed += 1
    return good


def main() -> int:
    print("\nconfused.de — setup check\n" + "-" * 58)

    v = sys.version_info
    report(v >= (3, 10), "Python 3.10 or newer",
           f"found {v.major}.{v.minor}.{v.micro}")

    for mod in ("fastapi", "uvicorn", "pydantic", "langgraph"):
        try:
            __import__(mod)
            report(True, f"package: {mod}")
        except ImportError:
            report(False, f"package: {mod}", "pip install -r requirements.txt")

    provider = (os.environ.get("LLM_PROVIDER") or "anthropic").strip().lower()
    print(f"\n  provider: {provider}\n" + "-" * 58)

    sdk = "openai" if provider in ("openai", "oai") else "anthropic"
    try:
        __import__(sdk)
        report(True, f"package: {sdk}")
    except ImportError:
        report(False, f"package: {sdk}", f"pip install {sdk}")

    try:
        from agent import llm
    except Exception as exc:  # noqa: BLE001
        report(False, "agent.llm imports", str(exc))
        return 1

    try:
        key = llm.load_api_key()
        shown = key[:7] + "…" + key[-4:] if len(key) > 14 else "(short)"
        report(True, "API key found", shown)
    except Exception as exc:  # noqa: BLE001
        report(False, "API key found", str(exc))
        key = ""

    for entry in llm.catalogue():
        print(f"{OK} tier {entry['value']:<7} -> {entry['model']}"
              + ("  (default)" if entry["default"] else ""))

    if key:
        try:
            names = {m.id for m in llm.client().models.list()}
            for entry in llm.catalogue():
                report(entry["model"] in names,
                       f"model reachable: {entry['model']}",
                       "not in this account's catalogue; set the override "
                       "environment variable to one that is",
                       fatal=False, only_if_bad=True)
        except Exception as exc:  # noqa: BLE001
            report(False, "model catalogue listed", str(exc)[:90], fatal=False)

        try:
            r = llm.ask(llm.DEFAULT_MODEL, "Reply with the single word: ready.",
                        [{"role": "user", "content": "ready?"}], max_tokens=16)
            report(bool(r.data), "a real call succeeded",
                   f"{r.input_tokens} in / {r.output_tokens} out")
        except Exception as exc:  # noqa: BLE001
            report(False, "a real call succeeded", str(exc)[:160])

    print("-" * 58)
    with socket.socket() as s:
        s.settimeout(0.4)
        busy = s.connect_ex(("127.0.0.1", 8000)) == 0
    report(True, "port 8000",
           "already in use — stop that server first" if busy else "free",
           fatal=False)

    data = HERE / "data"
    report(os.access(data.parent, os.W_OK), "data directory writable", str(data))

    print("-" * 58)
    print("\nAll checks passed. Start the server with:  .\\start.ps1\n"
          if not failed else
          f"\n{failed} check(s) failed. Fix those before starting.\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

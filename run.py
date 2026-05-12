#!/usr/bin/env python3
"""
run.py — Run llmproxy without installing.

Usage:
    python run.py              # start the server
    python run.py --setup      # configure providers
    python run.py --version
    python run.py --list-providers
    python run.py --port 9000 --log-level DEBUG
"""
import os
import sys

# ---------------------------------------------------------------------------
# Locate the llmproxy package directory.
#
# We search three candidate locations in priority order:
#   1. The directory that contains this run.py file          (normal case)
#   2. The parent of that directory                          (if run.py sits
#                                                             one level deep)
#   3. The current working directory                         (fallback)
#
# The first candidate that contains a 'llmproxy/' subdirectory with an
# '__init__.py' wins and is prepended to sys.path.
# ---------------------------------------------------------------------------

def _find_package_root() -> str | None:
    """Return the directory that contains the llmproxy package, or None."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        script_dir,
        os.path.dirname(script_dir),
        os.getcwd(),
    ]
    for candidate in candidates:
        init = os.path.join(candidate, "llmproxy", "__init__.py")
        if os.path.isfile(init):
            return candidate
    return None


root = _find_package_root()

if root is None:
    # Give a clear diagnostic rather than a confusing ImportError.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("ERROR: Cannot find the llmproxy package.")
    print()
    print("Expected to find a 'llmproxy/' subdirectory containing '__init__.py'")
    print(f"alongside this run.py file ({script_dir}).")
    print()
    print("Your directory layout should look like this:")
    print()
    print("  llmproxy/                ← you should be here, or run.py should be here")
    print("  ├── run.py")
    print("  ├── llmproxy_test_client.py")
    print("  └── llmproxy/            ← the package (contains __init__.py)")
    print("      ├── __init__.py")
    print("      ├── __main__.py")
    print("      ├── config.py")
    print("      ├── server.py")
    print("      └── setup_wizard.py")
    print()
    print("Current directory:    ", os.getcwd())
    print("run.py lives in:      ", script_dir)
    print("Contents of run.py's directory:")
    try:
        for entry in sorted(os.listdir(script_dir)):
            kind = "/" if os.path.isdir(os.path.join(script_dir, entry)) else ""
            print(f"    {entry}{kind}")
    except Exception:
        print("    (could not list directory)")
    sys.exit(1)

if root not in sys.path:
    sys.path.insert(0, root)

from llmproxy.__main__ import main  # noqa: E402  (import after path fix)

if __name__ == "__main__":
    main()

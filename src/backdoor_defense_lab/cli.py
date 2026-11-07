"""Command line interface (filled in by later modules)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from backdoor_defense_lab import __version__


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="backdoor_defense_lab")
    parser.add_argument("--version", action="version", version=__version__)
    parser.parse_args(argv)
    return 0

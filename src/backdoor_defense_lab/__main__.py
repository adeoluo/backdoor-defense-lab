"""Entrypoint: `python -m backdoor_defense_lab`."""

from __future__ import annotations

from backdoor_defense_lab import __version__
from backdoor_defense_lab.configuration import settings


def main() -> None:
    header = f"Backdoor Defense Lab v{__version__}"
    status = f"environment={settings.environment}, seed={settings.seed}"
    print(f"{header} - {status}")


if __name__ == "__main__":
    main()

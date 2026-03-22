from __future__ import annotations

import argparse

from .runner import run_from_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified DreamID-V inference entry point.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()
    run_from_yaml(args.config)


if __name__ == "__main__":
    main()

"""`ars` console entrypoint — thin dispatcher over subsystem CLIs.

Phase 0 ships `config` and `taxonomy` inspection; later phases register their own.
Usage: `python -m ars <command> [--config configs/default.yaml]`
"""

from __future__ import annotations

import argparse
import json
import sys

from ars.config import DEFAULT_CONFIG, Settings
from ars.noise_lab.taxonomy import load_taxonomy


def _cmd_config(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    print(json.dumps(settings.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


def _cmd_taxonomy(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    tax = load_taxonomy(settings.paths.noise_taxonomy)
    print(f"families: {len(tax.families)}  subtypes: {len(tax.subtypes)}")
    print(f"canonical cells: {len(tax.canonical_cells())}")
    for code in tax.subtype_codes():
        sub = tax.subtypes[code]
        print(f"  {code}  {sub.name:22s} (family {sub.family}: {tax.families[sub.family].name})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ars", description="ARS control CLI")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("config", help="print resolved settings")
    sub.add_parser("taxonomy", help="print the noise taxonomy")

    args = parser.parse_args(argv)
    dispatch = {"config": _cmd_config, "taxonomy": _cmd_taxonomy}
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

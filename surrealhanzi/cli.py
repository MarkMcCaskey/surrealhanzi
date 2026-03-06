"""CLI entry point for SurrealHanzi."""

import argparse
import sys

from .glyph_data import GlyphData
from .ids_parser import IDS_OPERATORS, parse_ids
from .renderer import Renderer


def _has_ids_operator(s: str) -> bool:
    return any(ch in IDS_OPERATORS for ch in s)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="surrealhanzi",
        description="Render Chinese characters from IDS sequences to SVG",
    )
    subparsers = parser.add_subparsers(dest="command")

    render_p = subparsers.add_parser("render", help="Render a character or IDS sequence")
    render_p.add_argument("input", help="A character or IDS string (e.g. 好 or ⿰女子)")
    render_p.add_argument("-o", "--output", help="Output SVG file (default: stdout)")
    render_p.add_argument("-s", "--size", type=int, default=256, help="SVG size in pixels")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "render":
        glyph_data = GlyphData()
        glyph_data.load()
        renderer = Renderer(glyph_data)

        inp = args.input

        if _has_ids_operator(inp):
            svg = renderer.render_ids(inp, size=args.size)
        elif len(inp) == 1:
            svg = renderer.render_char(inp, size=args.size)
        else:
            print(f"Error: input must be a single character or IDS string, got: {inp!r}", file=sys.stderr)
            sys.exit(1)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(svg)
            print(f"Written to {args.output}")
        else:
            print(svg)


if __name__ == "__main__":
    main()

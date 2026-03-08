"""CLI entry point for SurrealHanzi."""

import argparse
import os
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

    # Render command
    render_p = subparsers.add_parser("render", help="Render a character or IDS sequence")
    render_p.add_argument("input", help="A character or IDS string (e.g. 好 or ⿰女子)")
    render_p.add_argument("-o", "--output", help="Output SVG file (default: stdout)")
    render_p.add_argument("-s", "--size", type=int, default=256, help="SVG size in pixels")

    # Train command
    train_p = subparsers.add_parser("train", help="Train the IDS transformer")
    train_p.add_argument("--epochs", type=int, default=50)
    train_p.add_argument("--d-model", type=int, default=192)
    train_p.add_argument("--n-layers", type=int, default=4)
    train_p.add_argument("--n-heads", type=int, default=4)
    train_p.add_argument("--batch-size", type=int, default=256)
    train_p.add_argument("--lr", type=float, default=3e-4)

    # Generate command
    gen_p = subparsers.add_parser("generate", help="Generate and render novel characters")
    gen_p.add_argument("-n", type=int, default=10, help="Number of characters to generate")
    gen_p.add_argument("--temperature", type=float, default=1.0)
    gen_p.add_argument("--top-k", type=int, default=50)
    gen_p.add_argument("-o", "--output-dir", help="Output directory for SVGs")
    gen_p.add_argument("-s", "--size", type=int, default=256, help="SVG size in pixels")

    # Site command
    site_p = subparsers.add_parser("site", help="Build the static dictionary site")
    site_p.add_argument("-o", "--output-dir", help="Output directory (default: docs/)")

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

    elif args.command == "train":
        from .train import train
        train(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

    elif args.command == "generate":
        from .train import generate as gen_ids
        from .ids_parser import IDSParseError

        glyph_data = GlyphData()
        glyph_data.load()

        sequences = gen_ids(
            n=args.n,
            temperature=args.temperature,
            top_k=args.top_k,
            glyph_data=glyph_data,
        )

        if args.output_dir and sequences:
            os.makedirs(args.output_dir, exist_ok=True)
            renderer = Renderer(glyph_data)

            for i, ids_str in enumerate(sequences):
                try:
                    svg = renderer.render_ids(ids_str, size=args.size)
                    path = os.path.join(args.output_dir, f"surreal_{i:03d}.svg")
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(svg)
                    print(f"  Rendered {ids_str} -> {path}")
                except Exception as e:
                    print(f"  Failed to render {ids_str}: {e}", file=sys.stderr)


    elif args.command == "site":
        from .site import build
        build(output_dir=args.output_dir)


if __name__ == "__main__":
    main()

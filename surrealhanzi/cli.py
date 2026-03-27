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

    # Dict generation command
    dg_p = subparsers.add_parser("dictgen", help="Generate dictionary entries via Claude")
    dg_p.add_argument("ids", nargs="*", help="IDS strings to generate entries for")
    dg_p.add_argument("-f", "--file", help="File with one IDS per line")
    dg_p.add_argument("--from-generate", type=int, metavar="N",
                       help="Generate N novel IDS sequences first, then create entries")
    dg_p.add_argument("-o", "--output-dir", help="Output dir for YAML (default: site/characters/)")
    dg_p.add_argument("--examples-dir", help="Output dir for SVGs (default: examples/)")
    dg_p.add_argument("--model", default="claude-sonnet-4-20250514", help="Anthropic model")
    dg_p.add_argument("--dry-run", action="store_true", help="Show prompt without calling API")
    dg_p.add_argument("--preview", action="store_true", help="Call API but don't write files")
    dg_p.add_argument("--skip-existing", action="store_true", help="Skip IDS with existing entries")
    dg_p.add_argument("-s", "--size", type=int, default=512, help="SVG render size")

    # Generate diversity options
    gen_p.add_argument("--with-radical", help="Only keep sequences containing this radical")
    gen_p.add_argument("--style", choices=["natural", "weird"],
                       help="Preset: natural (low temp) or weird (high temp)")

    # Grade command
    grade_p = subparsers.add_parser("grade", help="Grade rendering quality of all characters")
    grade_p.add_argument("-d", "--characters-dir", default="site/characters",
                         help="Directory with YAML entries (default: site/characters/)")
    grade_p.add_argument("-o", "--output", default="examples/grade_report.html",
                         help="Output HTML report path")
    grade_p.add_argument("--ids", nargs="*",
                         help="Grade specific IDS strings instead of dictionary entries")
    grade_p.add_argument("--no-report", action="store_true",
                         help="Skip HTML report, terminal output only")

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
        from .ids_parser import IDSParseError, collect_leaves

        glyph_data = GlyphData()
        glyph_data.load()

        # Apply style presets
        temperature = args.temperature
        top_k = args.top_k
        if args.style == "natural":
            temperature = 0.7
            top_k = 30
        elif args.style == "weird":
            temperature = 1.4
            top_k = 100

        # Build filter function for --with-radical
        filter_fn = None
        if args.with_radical:
            target = args.with_radical
            def filter_fn(ids_str):
                try:
                    tree = parse_ids(ids_str)
                    return target in collect_leaves(tree)
                except IDSParseError:
                    return False

        sequences = gen_ids(
            n=args.n,
            temperature=temperature,
            top_k=top_k,
            glyph_data=glyph_data,
            filter_fn=filter_fn,
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

    elif args.command == "dictgen":
        from .dictgen import generate_entry, _existing_ids_strings

        glyph_data = GlyphData()
        glyph_data.load()
        renderer = Renderer(glyph_data)

        # Collect IDS strings from various sources
        ids_list = list(args.ids) if args.ids else []

        if args.file:
            with open(args.file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        ids_list.append(line)

        if args.from_generate:
            from .train import generate as gen_ids
            print(f"Generating {args.from_generate} novel IDS sequences...")
            sequences = gen_ids(
                n=args.from_generate,
                glyph_data=glyph_data,
            )
            ids_list.extend(sequences)

        if not ids_list:
            print("Error: no IDS strings provided. Use positional args, -f, or --from-generate.",
                  file=sys.stderr)
            sys.exit(1)

        # Skip existing if requested
        characters_dir = args.output_dir or None
        if args.skip_existing:
            existing_ids = _existing_ids_strings(characters_dir or "site/characters")
            ids_list = [s for s in ids_list if s not in existing_ids]
            if not ids_list:
                print("All IDS strings already have entries. Nothing to do.")
                sys.exit(0)

        # Create Anthropic client (lazy import)
        client = None
        if not args.dry_run:
            try:
                import anthropic
            except ImportError:
                print("Error: 'anthropic' package required. Install with: pip install anthropic",
                      file=sys.stderr)
                sys.exit(1)
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print("Error: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
                sys.exit(1)
            client = anthropic.Anthropic(api_key=api_key)

        print(f"Processing {len(ids_list)} IDS sequence(s)...\n")
        for ids_str in ids_list:
            print(f"  {ids_str}")
            try:
                generate_entry(
                    ids_str=ids_str,
                    renderer=renderer,
                    glyph_data=glyph_data,
                    client=client,
                    model=args.model,
                    size=args.size,
                    characters_dir=characters_dir,
                    examples_dir=args.examples_dir,
                    dry_run=args.dry_run,
                    preview=args.preview,
                )
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)

        # Rebuild site if we wrote any files
        if not args.dry_run and not args.preview:
            print("\nRebuilding site...")
            from .site import build
            build(output_dir=None)

    elif args.command == "grade":
        from .grade import (
            grade_all, grade_ids_list, generate_report, print_summary,
        )

        glyph_data = GlyphData()
        glyph_data.load()
        renderer = Renderer(glyph_data)

        if args.ids:
            results = grade_ids_list(args.ids, renderer, glyph_data)
        else:
            results = grade_all(args.characters_dir, renderer, glyph_data)

        print_summary(results)

        if not args.no_report:
            generate_report(results, args.output)
            print(f"\n  Report: {args.output}")

    elif args.command == "site":
        from .site import build
        build(output_dir=args.output_dir)


if __name__ == "__main__":
    main()

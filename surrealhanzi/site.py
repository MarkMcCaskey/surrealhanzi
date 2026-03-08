"""Static site generator for SurrealHanzi dictionary.

Generates:
  - manifest.json: lightweight list of character stubs (for index grid)
  - characters/{id}.json: full character data with inlined SVG (loaded on demand)
  - index.html: single-page app with hash routing
  - static/style.css: styles
"""

import json
import os
import re
import shutil

import yaml


# Paths relative to project root
_ROOT = os.path.dirname(os.path.dirname(__file__))
SITE_DIR = os.path.join(_ROOT, "site")
CHARACTERS_DIR = os.path.join(SITE_DIR, "characters")
STATIC_DIR = os.path.join(SITE_DIR, "static")
EXAMPLES_DIR = os.path.join(_ROOT, "examples")
DEFAULT_OUTPUT = os.path.join(_ROOT, "docs")
INDEX_HTML = os.path.join(SITE_DIR, "index.html")


def _load_svg(svg_filename: str) -> str:
    """Load an SVG file, strip XML declaration and inline color styles."""
    path = os.path.join(EXAMPLES_DIR, svg_filename)
    if not os.path.exists(path):
        return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256"><text x="128" y="140" text-anchor="middle" font-size="20" fill="currentColor">?</text></svg>'
    with open(path, encoding="utf-8") as f:
        svg = f.read()
    svg = re.sub(r'<\?xml[^?]*\?>\s*', '', svg)
    svg = svg.replace(' style="color: #000"', '')
    return svg.strip()


def _load_characters() -> list[dict]:
    """Load all character YAML files."""
    characters = []
    if not os.path.isdir(CHARACTERS_DIR):
        return characters

    for filename in sorted(os.listdir(CHARACTERS_DIR)):
        if not filename.endswith(('.yaml', '.yml')):
            continue
        with open(os.path.join(CHARACTERS_DIR, filename), encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if data is None:
            continue
        if 'id' not in data:
            data['id'] = filename.rsplit('.', 1)[0]
        characters.append(data)

    return characters


def build(output_dir: str | None = None) -> None:
    """Build the static site."""
    output_dir = output_dir or DEFAULT_OUTPUT
    chars_out = os.path.join(output_dir, "characters")
    static_out = os.path.join(output_dir, "static")

    os.makedirs(chars_out, exist_ok=True)
    os.makedirs(static_out, exist_ok=True)

    characters = _load_characters()
    print(f"Loaded {len(characters)} character(s)")

    # Build manifest (lightweight stubs) and per-character JSON files
    manifest = []
    for char in characters:
        # Full character JSON with inlined SVG
        full = dict(char)
        svg_file = full.pop('svg', '')
        full['svg'] = _load_svg(svg_file)

        char_path = os.path.join(chars_out, f"{char['id']}.json")
        with open(char_path, 'w', encoding='utf-8') as f:
            json.dump(full, f, ensure_ascii=False)
        print(f"  characters/{char['id']}.json")

        # Manifest stub — no SVG, just metadata for the grid
        manifest.append({
            'id': char['id'],
            'ids': char.get('ids', ''),
            'title': char.get('title', char['id']),
            'meaning': char.get('meaning', ''),
        })

    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False)
    print(f"  manifest.json ({len(manifest)} entries)")

    # Copy index.html
    shutil.copy2(INDEX_HTML, os.path.join(output_dir, 'index.html'))
    print(f"  index.html")

    # Copy static files
    if os.path.isdir(STATIC_DIR):
        for filename in os.listdir(STATIC_DIR):
            src = os.path.join(STATIC_DIR, filename)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(static_out, filename))
    print(f"  static/")

    # .nojekyll for GitHub Pages
    with open(os.path.join(output_dir, '.nojekyll'), 'w') as f:
        pass

    print(f"\nSite built to {output_dir}/")

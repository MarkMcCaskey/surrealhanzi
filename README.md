# SurrealHanzi

A surrealist dictionary of Chinese characters that don't exist — but feel like they should.

Characters are composed from real radicals using [IDS](https://en.wikipedia.org/wiki/Ideographic_Description_Characters_(Unicode_block)) (Ideographic Description Sequences), rendered as SVG, and given fictional dictionary entries: pronunciation, etymology, classical citations, and example sentences in literary Chinese.

## Setup

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dictgen]"
```

This installs all dependencies including the Anthropic SDK and CairoSVG. For rendering and training only (no dictionary generation), `pip install -e ".[site]"` is sufficient.

## Generation Pipeline

The full pipeline from novel character to dictionary entry:

### 1. Train the IDS transformer

```sh
surrealhanzi train --epochs 50
```

Trains a small (4M param) character-level transformer on ~107K real IDS decompositions. The model learns the grammar of how radicals compose. Saves to `models/ids_transformer.pt`.

### 2. Generate novel IDS sequences

```sh
surrealhanzi generate -n 10 -o examples/generated/
```

Samples the transformer for new IDS strings, filters for valid parses and renderable components, and writes SVGs.

Options for controlling diversity:
- `--style natural` — lower temperature (0.7), favors common radical combinations
- `--style weird` — higher temperature (1.4), favors unusual combinations
- `--with-radical "水"` — only keep sequences containing a specific radical
- `--temperature` / `--top-k` — manual control

### 3. Generate dictionary entries

```sh
export ANTHROPIC_API_KEY=sk-ant-...
surrealhanzi dictgen "⿰心言"
```

Takes an IDS string, renders it to SVG, converts to PNG, and sends the image along with the structural decomposition to Claude. The model returns a full dictionary entry: pronunciation (pinyin + bopomofo), definitions in Traditional Chinese and English, etymology, a classical source citation with fanqie reading, literary example sentences, and related real characters.

Output is written to `site/characters/{slug}.yaml` and the SVG to `examples/{slug}.svg`.

Options:
- `--dry-run` — show the prompt without calling the API
- `--preview` — call the API and print the result, but don't write files
- `--skip-existing` — skip IDS strings that already have entries
- `--from-generate 5` — generate 5 novel IDS sequences first, then create entries for each
- `-f ids.txt` — read IDS strings from a file (one per line)
- `--model claude-sonnet-4-20250514` — choose the model

### 4. Build and preview the site

```sh
surrealhanzi site
```

Reads YAML entries from `site/characters/`, inlines SVGs, and builds the static site to `docs/`.

To preview locally:

```sh
cd docs && python -m http.server 8000
```

Then open [http://localhost:8000](http://localhost:8000).

### End-to-end example

Generate 3 novel characters and create dictionary entries for all of them:

```sh
surrealhanzi dictgen --from-generate 3
surrealhanzi site
cd docs && python -m http.server 8000
```

## Rendering only

Render any IDS string or real character to SVG:

```sh
surrealhanzi render "⿰火水" -o fire-water.svg
surrealhanzi render 好 -o hao.svg
```

## Project structure

```
surrealhanzi/
  cli.py          — CLI entry point
  renderer.py     — SVG composition engine
  dictgen.py      — dictionary entry generation via Claude
  train.py        — transformer training and IDS generation
  transformer.py  — model architecture
  ids_parser.py   — IDS string parser
  ids_dataset.py  — dataset loading and vocab
  glyph_data.py   — stroke data from Make Me a Hanzi
  site.py         — static site builder (YAML → JSON)
site/
  index.html      — single-page app (vanilla JS, hash routing)
  static/style.css
  characters/*.yaml — dictionary entries
data/
  graphics.txt    — stroke data (~9.5K characters)
  dictionary.txt  — decompositions
  ids_babelstone.txt — 97K+ IDS sequences
docs/             — built site (GitHub Pages)
models/           — trained transformer checkpoint
```

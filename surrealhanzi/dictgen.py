"""Generate dictionary entries for fictional characters using Claude."""

import base64
import json
import os
import re
import sys

import yaml

from .ids_parser import IDSNode, IDS_OPERATORS, collect_leaves, parse_ids

# Paths
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARACTERS_DIR = os.path.join(_ROOT, "site", "characters")
EXAMPLES_DIR = os.path.join(_ROOT, "examples")

# IDS operator names for structural descriptions
OPERATOR_NAMES: dict[str, str] = {
    "\u2FF0": "left-right",
    "\u2FF1": "top-bottom",
    "\u2FF2": "left-middle-right",
    "\u2FF3": "top-middle-bottom",
    "\u2FF4": "full surround",
    "\u2FF5": "surround from above",
    "\u2FF6": "surround from below",
    "\u2FF7": "surround from left",
    "\u2FF8": "surround from upper-left",
    "\u2FF9": "surround from upper-right",
    "\u2FFA": "surround from lower-left",
    "\u2FFB": "overlaid",
}

# Common radical/component meanings (Traditional Chinese focus)
RADICAL_MEANINGS: dict[str, str] = {
    "人": "person", "亻": "person", "口": "mouth", "心": "heart", "忄": "heart",
    "手": "hand", "扌": "hand", "水": "water", "氵": "water", "火": "fire",
    "灬": "fire", "木": "wood/tree", "金": "metal/gold", "釒": "metal",
    "土": "earth", "日": "sun/day", "月": "moon/month", "山": "mountain",
    "石": "stone", "雨": "rain", "風": "wind", "雲": "cloud",
    "田": "field", "禾": "grain", "米": "rice", "竹": "bamboo",
    "言": "speech/words", "訁": "speech", "目": "eye", "耳": "ear",
    "足": "foot", "馬": "horse", "魚": "fish", "鳥": "bird",
    "虫": "insect", "犬": "dog", "犭": "dog/beast", "牛": "ox/cow",
    "羊": "sheep", "豕": "pig", "鬼": "ghost", "龍": "dragon",
    "女": "woman", "子": "child/son", "王": "king/jade", "玉": "jade",
    "刀": "knife/blade", "刂": "blade", "弓": "bow", "矢": "arrow",
    "門": "gate/door", "戶": "door", "車": "cart/vehicle",
    "食": "food/eat", "飠": "food", "酉": "wine/alcohol",
    "貝": "shell/money", "衣": "clothing", "衤": "clothing",
    "糸": "silk/thread", "纟": "thread", "力": "strength/power",
    "大": "big/great", "小": "small", "白": "white", "黑": "black",
    "赤": "red", "青": "blue-green",
    "天": "heaven/sky", "地": "earth/ground", "氣": "qi/air",
    "生": "life/birth", "死": "death", "老": "old", "少": "young/few",
    "長": "long/grow", "高": "tall/high", "見": "see",
    "走": "walk/run", "辶": "walk/advance", "立": "stand",
    "示": "spirit/show", "礻": "spirit", "宀": "roof/house",
    "广": "shelter", "囗": "enclosure", "冖": "cover",
    "艸": "grass/plant", "艹": "grass", "花": "flower",
    "鉄": "iron", "玄": "dark/mysterious", "夕": "evening",
    "文": "writing/culture", "武": "martial", "頁": "page/head",
    "音": "sound", "革": "leather", "骨": "bone", "血": "blood",
    "肉": "flesh/meat", "月": "moon/flesh", "身": "body",
    "里": "village/mile", "方": "square/direction", "北": "north",
    "南": "south", "東": "east", "西": "west",
    "上": "above", "下": "below", "中": "middle/center",
    "非": "not/wrong", "乞": "beg", "賣": "sell",
    "雪": "snow", "電": "lightning/electricity", "冰": "ice",
    "霧": "fog", "露": "dew", "霜": "frost",
}

# System prompt
SYSTEM_PROMPT = """\
You are a lexicographer for SurrealHanzi, a surrealist dictionary of Chinese \
characters that don't exist but feel like they should. You write in Traditional \
Chinese (zh-TW). You create dictionary entries that feel authentic — as if \
these were rare classical characters that simply fell out of use.

Your entries should be linguistically plausible: pronunciation follows the \
phonetic component, meanings arise naturally from the combination of radicals, \
and example sentences use literary Chinese (文言文 or semi-literary style)."""

# User prompt template
USER_PROMPT_TEMPLATE = """\
Here is a rendered image of a fictional Chinese character and its structural \
decomposition. Create the dictionary entry.

## Structure

IDS: {ids}
Composition: {structure}
Components:
{components}

## Output

Return a single JSON object with these exact fields:

```json
{{
  "title": "English name for this character (2-4 words, evocative)",
  "pronunciation": "pinyin with tone marks (based on phonetic component)",
  "bopomofo": "ㄅㄆㄇㄈ equivalent",
  "meaning": "One-sentence English summary of the primary meaning",
  "formation": "phono-semantic | ideographic | associative",
  "definitions": [
    {{
      "zh": "Traditional Chinese definition",
      "en": "English definition"
    }}
  ],
  "etymology": "2-3 sentences explaining why this character should exist. \
Written in English, literary but accessible. This is a surrealist dictionary — \
the tone should be slightly wistful, as if mourning a character that was never written.",
  "source": {{
    "work": "Plausible classical source (《說文》《玉篇》《廣韻》《集韻》etc.)",
    "gloss": "Classical Chinese gloss with fanqie reading (反切). \
The fanqie must be phonologically consistent with the pronunciation."
  }},
  "examples": [
    {{
      "text": "Literary Chinese example sentence using {ids} for the character",
      "translation": "English translation",
      "style": "classical"
    }},
    {{
      "text": "Contemporary example using {ids}",
      "translation": "English translation",
      "style": "contemporary"
    }}
  ],
  "related": [
    {{
      "char": "A real character with semantic proximity",
      "pinyin": "its pinyin",
      "gloss": "brief English gloss"
    }}
  ],
  "tags": ["2-4 thematic tags"]
}}
```

Rules:
- 2-3 definitions, ordered concrete → abstract
- Use "{ids}" in example sentences where the character would appear
- All Chinese text must be Traditional Chinese (zh-TW)
- Pronunciation must follow the phonetic component's standard reading
- The fanqie (反切) must produce the correct initial and final
- Related characters must be real characters
- Do not include any text outside the JSON"""


def get_radical_meaning(char: str) -> str:
    """Look up English meaning for a radical/component character."""
    return RADICAL_MEANINGS.get(char, "")


def describe_structure(node: IDSNode, depth: int = 0) -> str:
    """Build a human-readable structural description from an IDS tree."""
    if node.is_leaf:
        char = node.character or "?"
        meaning = get_radical_meaning(char)
        if meaning:
            return f"{char} ({meaning})"
        return char

    op = node.operator or "?"
    op_name = OPERATOR_NAMES.get(op, "unknown")
    children_desc = [describe_structure(c, depth + 1) for c in node.children]

    if len(children_desc) == 2:
        return f"{op_name} ({op}): {children_desc[0]} and {children_desc[1]}"
    elif len(children_desc) == 3:
        return f"{op_name} ({op}): {children_desc[0]}, {children_desc[1]}, and {children_desc[2]}"
    return f"{op_name} ({op}): {', '.join(children_desc)}"


def describe_components(node: IDSNode) -> str:
    """List all leaf components with their meanings."""
    leaves = collect_leaves(node)
    lines = []
    seen = set()
    for char in leaves:
        if char in seen:
            continue
        seen.add(char)
        meaning = get_radical_meaning(char)
        if meaning:
            lines.append(f"- {char}: {meaning}")
        else:
            lines.append(f"- {char}")
    return "\n".join(lines)


def build_prompt(ids_str: str, tree: IDSNode) -> str:
    """Build the user prompt with structural analysis filled in."""
    return USER_PROMPT_TEMPLATE.format(
        ids=ids_str,
        structure=describe_structure(tree),
        components=describe_components(tree),
    )


def svg_to_png(svg_string: str, size: int = 512) -> bytes:
    """Convert SVG string to PNG bytes for the vision API."""
    import cairosvg
    return cairosvg.svg2png(
        bytestring=svg_string.encode("utf-8"),
        output_width=size,
        output_height=size,
    )


def parse_response(text: str) -> dict:
    """Extract and parse JSON from a model response."""
    # Try to find JSON in code fences first
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        # Try to find raw JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

    return json.loads(text)


def ids_to_slug(title: str, existing: set[str]) -> str:
    """Derive a URL-safe slug from a title, avoiding collisions."""
    slug = title.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s]+", "-", slug)
    slug = slug.strip("-")
    if not slug:
        slug = "character"

    if slug not in existing:
        return slug

    for i in range(2, 100):
        candidate = f"{slug}-{i}"
        if candidate not in existing:
            return candidate
    return f"{slug}-{len(existing)}"


def _existing_slugs(characters_dir: str) -> set[str]:
    """Scan existing YAML files for their IDs."""
    slugs = set()
    if not os.path.isdir(characters_dir):
        return slugs
    for fn in os.listdir(characters_dir):
        if fn.endswith((".yaml", ".yml")):
            slugs.add(fn.rsplit(".", 1)[0])
    return slugs


def _existing_ids_strings(characters_dir: str) -> set[str]:
    """Scan existing YAML files for their IDS strings."""
    ids_set = set()
    if not os.path.isdir(characters_dir):
        return ids_set
    for fn in os.listdir(characters_dir):
        if not fn.endswith((".yaml", ".yml")):
            continue
        path = os.path.join(characters_dir, fn)
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data and "ids" in data:
                ids_set.add(data["ids"])
        except Exception:
            pass
    return ids_set


def generate_entry(
    ids_str: str,
    renderer,
    glyph_data,
    client,
    model: str = "claude-sonnet-4-20250514",
    size: int = 512,
    characters_dir: str | None = None,
    examples_dir: str | None = None,
    dry_run: bool = False,
    preview: bool = False,
) -> dict | None:
    """Generate a dictionary entry for an IDS string.

    Returns the entry dict, or None on failure.
    """
    characters_dir = characters_dir or CHARACTERS_DIR
    examples_dir = examples_dir or EXAMPLES_DIR

    # Parse IDS
    tree = parse_ids(ids_str)

    # Build prompt
    prompt_text = build_prompt(ids_str, tree)

    if dry_run:
        print(f"\n{'='*60}")
        print(f"IDS: {ids_str}")
        print(f"Structure: {describe_structure(tree)}")
        print(f"\n--- SYSTEM PROMPT ---")
        print(SYSTEM_PROMPT)
        print(f"\n--- USER PROMPT ---")
        print(prompt_text)
        print(f"{'='*60}")
        return None

    # Render SVG
    svg_str = renderer.render_ids(ids_str, size=size)

    # Convert to PNG for vision API
    try:
        png_bytes = svg_to_png(svg_str, size=size)
        image_content = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(png_bytes).decode(),
            },
        }
    except Exception as e:
        print(f"  Warning: SVG→PNG conversion failed ({e}), sending without image", file=sys.stderr)
        image_content = None

    # Build message content
    content = []
    if image_content:
        content.append(image_content)
    content.append({"type": "text", "text": prompt_text})

    # Call API
    print(f"  Calling {model}...")
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )

    # Parse response
    response_text = response.content[0].text
    try:
        entry = parse_response(response_text)
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"  Error: Failed to parse response: {e}", file=sys.stderr)
        print(f"  Raw response:\n{response_text[:500]}", file=sys.stderr)
        return None

    # Fill in known fields
    existing = _existing_slugs(characters_dir)
    title = entry.get("title", "untitled")
    slug = ids_to_slug(title, existing)

    entry["id"] = slug
    entry["ids"] = ids_str
    entry["svg"] = f"{slug}.svg"

    # Ensure components from tree are included
    if "components" not in entry:
        entry["components"] = []
        leaves = collect_leaves(tree)
        for char in leaves:
            comp = {"char": char, "meaning": get_radical_meaning(char) or char}
            entry["components"].append(comp)

    if preview:
        print(f"\n--- Preview: {ids_str} → {slug} ---")
        print(yaml.dump(entry, allow_unicode=True, default_flow_style=False, sort_keys=False))
        return entry

    # Write SVG
    os.makedirs(examples_dir, exist_ok=True)
    svg_path = os.path.join(examples_dir, f"{slug}.svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_str)
    print(f"  SVG: {svg_path}")

    # Write YAML
    os.makedirs(characters_dir, exist_ok=True)
    yaml_path = os.path.join(characters_dir, f"{slug}.yaml")

    # Order fields nicely for the YAML output
    ordered = _order_entry(entry)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(ordered, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"  YAML: {yaml_path}")

    return entry


def _order_entry(entry: dict) -> dict:
    """Reorder entry fields for readable YAML output."""
    key_order = [
        "id", "ids", "svg", "title", "pronunciation", "bopomofo",
        "meaning", "formation", "definitions", "etymology", "source",
        "components", "examples", "related", "tags",
    ]
    ordered = {}
    for key in key_order:
        if key in entry:
            ordered[key] = entry[key]
    # Any remaining keys
    for key, val in entry.items():
        if key not in ordered:
            ordered[key] = val
    return ordered

"""Parse BabelStone IDS data for comprehensive CJK character decompositions.

BabelStone IDS (https://babelstone.co.uk/CJK/IDS.TXT) provides IDS
decompositions for all 97,680 CJK unified ideographs in Unicode 16.0.
This is public domain data maintained by Andrew West.

Format: U+XXXX<TAB>char<TAB>^IDS$(sources)[<TAB>^IDS$(sources)...]
"""

import os
import re
from typing import Optional

from .ids_parser import IDS_OPERATORS, parse_ids, IDSParseError

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Curly-brace unencoded component expansions from BabelStone header.
# Maps {N} -> IDS expansion string. Only includes entries with known expansions.
_CURLY_EXPANSIONS: dict[int, str] = {
    1: "⿹⿺㇉一灬",
    2: "⿱⺈𫩏",
    3: "⿱⺈⿵冂人",
    4: "⿱⿻⿰丨丨丷冖",
    8: "⿻⿱㇒一乚",
    9: "⿱⿰㇒㇖丨",
    10: "⿻⿻㇈丿丶",
    11: "⿰⿱丶㇀⿱㇒丶",
    13: "⿸⿰丿⿱⺊⺂七",
    15: "⿻𱍸㇒",
    16: "⿲⿺𠄌⺀⿺𠄌⺀㇂",
    20: "丸",
    22: "⿺㇉一",
    28: "⿻弋一",
    29: "⿻弋𢆶",
    31: "⿴囗⿻𰀪丶",
    32: "⿱⿻𠮛⿰丨丨冖",
    34: "⿹勹丿",
    35: "⿻几𠄠",
    37: "⿱⺊冖",
    38: "⿱𰀉冖",
    41: "⿻𦉫𠄠",
    42: "彐",
    43: "⺕",
    44: "⿱十冖",
    46: "⿻日乚",
    48: "㐄",
    49: "⿰丨丿",
    67: "⿴⿰丨丨𠄠",
    73: "⿴𠀃三",
    78: "⿱⿻口⿰丨丨一",
    80: "⿱冖八",
    82: "⿱𦘒一",
    83: "⿳亠丷冖",
    84: "⿰丿乛",
    85: "⿰丿⿱丶乛",
    86: "⿲丶丶丶",
    87: "⿳亠口冖",
    88: "⿵𠆢一",
    93: "⿻昌乚",
    95: "⿻尸一",
    98: "⿻一曲",
    99: "⿻𦉫𠄠",
    109: "⿻⿱㇒一丿",
    110: "冂",
    113: "⿵冂八",
    114: "⿱土八",
    115: "⿻𰀪⺀",
    117: "⿺𠃊⺊",
    119: "⿻己⿱工工",
    121: "⿱⺈罒",
}

_CURLY_RE = re.compile(r"\{(\d+)\}")


def _expand_curlies(ids_str: str) -> str:
    """Replace {N} references with their IDS expansions where known."""
    def _replace(m: re.Match) -> str:
        num = int(m.group(1))
        expansion = _CURLY_EXPANSIONS.get(num)
        if expansion is not None:
            # Recursively expand in case expansions reference other curlies
            return _expand_curlies(expansion)
        return m.group(0)  # keep as-is if no expansion
    return _CURLY_RE.sub(_replace, ids_str)


def _extract_ids(field: str) -> Optional[str]:
    """Extract the IDS string from a ^IDS$(sources) field."""
    m = re.match(r"\^(.+?)\$", field)
    return m.group(1) if m else None


def load_babelstone_ids(
    path: Optional[str] = None,
    expand_curlies: bool = True,
    include_variants: bool = True,
) -> list[str]:
    """Load IDS sequences from BabelStone IDS.TXT.

    Args:
        path: Path to IDS.TXT file.
        expand_curlies: Whether to expand {N} component references.
        include_variants: Whether to include regional variant decompositions
            (multiple IDS per character) as additional training data.

    Returns:
        List of valid IDS decomposition strings.
    """
    path = path or os.path.join(DATA_DIR, "ids_babelstone.txt")
    sequences: list[str] = []
    seen: set[str] = set()

    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3 or not parts[0].startswith("U"):
                continue

            char = parts[1]
            ids_fields = parts[2:]

            # Determine which fields to process
            fields_to_use = ids_fields if include_variants else ids_fields[:1]

            for field in fields_to_use:
                raw = _extract_ids(field)
                if raw is None:
                    continue

                # Strip variation indicator (cosmetic, doesn't affect structure)
                ids_str = raw.replace("\u303E", "")  # 〾

                # Skip atomic/self-decomposed
                if ids_str == char:
                    continue

                # Expand curly-brace references
                if expand_curlies:
                    ids_str = _expand_curlies(ids_str)

                # Skip if still has unresolved curlies or unknowns
                if "{" in ids_str or "？" in ids_str:
                    continue

                # Must be at least operator + 1 operand
                if len(ids_str) < 2:
                    continue

                # Validate it parses as valid IDS
                try:
                    parse_ids(ids_str)
                except IDSParseError:
                    continue

                # Deduplicate
                if ids_str not in seen:
                    seen.add(ids_str)
                    sequences.append(ids_str)

    return sequences


def load_babelstone_character_map(
    path: Optional[str] = None,
) -> dict[str, str]:
    """Load a mapping of character -> primary IDS decomposition.

    Returns only the first (primary) IDS for each character.
    Useful for looking up decompositions of specific characters.
    """
    path = path or os.path.join(DATA_DIR, "ids_babelstone.txt")
    char_to_ids: dict[str, str] = {}

    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3 or not parts[0].startswith("U"):
                continue

            char = parts[1]
            raw = _extract_ids(parts[2])
            if raw is None:
                continue

            ids_str = raw.replace("\u303E", "")
            if ids_str == char:
                continue

            ids_str = _expand_curlies(ids_str)
            if "{" in ids_str or "？" in ids_str:
                continue

            try:
                parse_ids(ids_str)
            except IDSParseError:
                continue

            char_to_ids[char] = ids_str

    return char_to_ids

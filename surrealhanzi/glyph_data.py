"""Load and index Make Me a Hanzi stroke data."""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@dataclass
class ComponentStrokes:
    """Strokes for a component extracted from a composed character.

    These strokes are pre-positioned — they're drawn in the context of
    the composed character, not standalone.
    """
    strokes: list[str]
    bbox: tuple[float, float, float, float]  # min_x, min_y, max_x, max_y


def _parse_svg_numbers(path_d: str) -> list[float]:
    return [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', path_d)]


def _stroke_bbox(strokes: list[str]) -> Optional[tuple[float, float, float, float]]:
    all_x: list[float] = []
    all_y: list[float] = []
    for s in strokes:
        nums = _parse_svg_numbers(s)
        for i in range(0, len(nums) - 1, 2):
            all_x.append(nums[i])
            all_y.append(nums[i + 1])
    if not all_x:
        return None
    return min(all_x), min(all_y), max(all_x), max(all_y)


class GlyphData:
    """Index of character stroke data from Make Me a Hanzi."""

    def __init__(self) -> None:
        self._strokes: dict[str, list[str]] = {}
        self._decompositions: dict[str, str] = {}
        self._matches: dict[str, list] = {}
        self._loaded_graphics = False
        self._loaded_dictionary = False

    def load_graphics(self, path: Optional[str] = None) -> None:
        path = path or os.path.join(DATA_DIR, "graphics.txt")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self._strokes[entry["character"]] = entry["strokes"]
        self._loaded_graphics = True

    def load_dictionary(self, path: Optional[str] = None) -> None:
        path = path or os.path.join(DATA_DIR, "dictionary.txt")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                char = entry["character"]
                decomp = entry.get("decomposition", "")
                if decomp and not decomp.startswith("？"):
                    self._decompositions[char] = decomp
                matches = entry.get("matches")
                if matches:
                    self._matches[char] = matches
        self._loaded_dictionary = True

    def load(self) -> None:
        self.load_graphics()
        self.load_dictionary()

    def get_strokes(self, char: str) -> Optional[list[str]]:
        return self._strokes.get(char)

    def get_decomposition(self, char: str) -> Optional[str]:
        return self._decompositions.get(char)

    def get_matches(self, char: str) -> Optional[list]:
        return self._matches.get(char)

    def get_component_strokes(self, char: str, component_index: int) -> Optional[ComponentStrokes]:
        """Extract strokes for a specific component from a composed character.

        Uses the matches field to identify which strokes belong to which
        component. component_index is the top-level index (0 for first child,
        1 for second child of the root IDS operator).

        Returns the pre-positioned strokes (drawn in composition context).
        """
        all_strokes = self._strokes.get(char)
        matches = self._matches.get(char)
        if all_strokes is None or matches is None:
            return None

        component_stroke_list = []
        for i, match in enumerate(matches):
            if match is not None and len(match) > 0 and match[0] == component_index:
                if i < len(all_strokes):
                    component_stroke_list.append(all_strokes[i])

        if not component_stroke_list:
            return None

        bbox = _stroke_bbox(component_stroke_list)
        if bbox is None:
            return None

        return ComponentStrokes(strokes=component_stroke_list, bbox=bbox)

    def has_char(self, char: str) -> bool:
        return char in self._strokes

    @property
    def char_count(self) -> int:
        return len(self._strokes)

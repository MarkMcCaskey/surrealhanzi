"""Load and index Make Me a Hanzi stroke data."""

import json
import os
from typing import Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


class GlyphData:
    """Index of character stroke data from Make Me a Hanzi graphics.txt."""

    def __init__(self) -> None:
        self._strokes: dict[str, list[str]] = {}
        self._decompositions: dict[str, str] = {}
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
                decomp = entry.get("decomposition", "")
                if decomp and not decomp.startswith("？"):
                    self._decompositions[entry["character"]] = decomp
        self._loaded_dictionary = True

    def load(self) -> None:
        self.load_graphics()
        self.load_dictionary()

    def get_strokes(self, char: str) -> Optional[list[str]]:
        return self._strokes.get(char)

    def get_decomposition(self, char: str) -> Optional[str]:
        return self._decompositions.get(char)

    def has_char(self, char: str) -> bool:
        return char in self._strokes

    @property
    def char_count(self) -> int:
        return len(self._strokes)

"""Font-based glyph data using fonttools.

Extracts glyph outlines from an OTF/TTF font file so the Renderer
can compose characters that visually match the source font.
"""

from __future__ import annotations

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont


class FontGlyphData:
    """Glyph data extracted from an OpenType font.

    Provides the same interface as GlyphData so the Renderer can use
    either data source transparently.
    """

    def __init__(self, font_path: str) -> None:
        self._font = TTFont(font_path)
        self._glyph_set = self._font.getGlyphSet()
        self._cmap = self._font.getBestCmap()

        head = self._font["head"]
        self.units_per_em: int = head.unitsPerEm

        os2 = self._font["OS/2"]
        self.ascender: int = os2.sTypoAscender
        self.descender: int = os2.sTypoDescender

        # Cache: char -> (path_list, bbox) or None
        self._cache: dict[str, tuple[list[str], tuple[float, float, float, float]] | None] = {}

    @property
    def source_w(self) -> float:
        """Width of the em-square in font units."""
        return float(self.units_per_em)

    @property
    def source_h(self) -> float:
        """Height of the em-square (ascender - descender)."""
        return float(self.ascender - self.descender)

    @property
    def source_y_offset(self) -> float:
        """Y coordinate of the top of the em-square (ascender)."""
        return float(self.ascender)

    def has_char(self, char: str) -> bool:
        return ord(char) in self._cmap

    def _load(self, char: str) -> tuple[list[str], tuple[float, float, float, float]] | None:
        """Extract glyph path and bounding box for a character."""
        if char in self._cache:
            return self._cache[char]

        glyph_name = self._cmap.get(ord(char))
        if not glyph_name:
            self._cache[char] = None
            return None

        try:
            glyph = self._glyph_set[glyph_name]

            # SVG path data
            pen = SVGPathPen(self._glyph_set)
            glyph.draw(pen)
            path = pen.getCommands()
            if not path:
                self._cache[char] = None
                return None

            # Correct bounding box via BoundsPen (handles curves properly)
            bp = BoundsPen(self._glyph_set)
            glyph.draw(bp)
            bounds = bp.bounds  # (xMin, yMin, xMax, yMax)
            if bounds is None:
                self._cache[char] = None
                return None

            result = ([path], bounds)
            self._cache[char] = result
            return result
        except Exception:
            self._cache[char] = None
            return None

    def get_strokes(self, char: str) -> list[str] | None:
        """Return SVG path data for a character's glyph outline."""
        entry = self._load(char)
        return entry[0] if entry else None

    def get_bbox(self, char: str) -> tuple[float, float, float, float] | None:
        """Return correct bounding box (xMin, yMin, xMax, yMax) from fonttools.

        This is computed by BoundsPen which properly handles cubic/quadratic
        curves, unlike naive SVG path number extraction.
        """
        entry = self._load(char)
        return entry[1] if entry else None

    def get_decomposition(self, _char: str) -> str | None:
        return None

    def get_matches(self, _char: str) -> list | None:
        return None

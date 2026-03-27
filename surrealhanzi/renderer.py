"""SVG composition engine for IDS-based character rendering.

Uses data-driven proportions, radical variant mapping, and tight-fit
transforms to produce well-integrated character compositions.
"""

import re
from dataclasses import dataclass
from typing import Optional
from xml.sax.saxutils import escape

from .ids_parser import IDSNode, parse_ids


@dataclass
class BBox:
    """A bounding box for layout."""
    x: float
    y: float
    w: float
    h: float


# Make Me a Hanzi coordinate system:
# - 1024x1024 canvas
# - Origin at (0, 900), y decreases downward
# - Visible range: x [0, 1024], y [-124, 900]
SOURCE_W = 1024.0
SOURCE_H = 1024.0
SOURCE_Y_OFFSET = 900.0

# Padding fraction within each component cell (prevents components touching edges)
PADDING = 0.02

# Minimum effective source dimension (fraction of SOURCE_W) when computing
# scales for composed components.  Prevents small/simple radicals from being
# enlarged so much that their strokes become visually heavier than neighbours.
MIN_SOURCE_FRAC = 0.30

# How much non-uniform scaling (squishing) is allowed for composed components.
# 0.0 = purely uniform, 1.0 = fully fill the box.  Higher values let
# components adapt to narrow/short boxes (e.g., filling height in ⿰ layouts).
SQUISH_BLEND = 0.60

# Stroke weight compensation for composed sub-components.
# Adds an outline stroke to each component's paths, scaled inversely with
# the transform scale, so that components scaled down more get thicker
# outlines to compensate for visual thinning.  stroke_factor also accounts
# for stroke count: simpler radicals (fewer, thicker paths) get less
# compensation than dense components (many thin paths).
STROKE_COMPENSATE = 1.0  # base visual output units of added stroke

# Character body fraction of the em-square for composed characters.
# Standalone chars naturally fill ~85% of the em-square (from font data).
# Composed characters should match, so they look the same size as real glyphs.
EM_BODY = 0.93  # content area = 93% of viewBox, matching real font characters

# Font to use in SVGs (for text fallbacks; matches site CSS)
SVG_FONT = '"Noto Serif TC", "Source Han Serif TC", "Songti TC", serif'


# --- Radical variant mapping (Traditional Chinese / zh-TW) ---
# When a radical appears in certain positions, use its positional variant.

# Map from standalone form to positional variant
LEFT_VARIANTS: dict[str, str] = {
    "人": "亻", "水": "氵", "手": "扌", "心": "忄",
    "犬": "犭", "示": "礻", "衣": "衤", "玉": "王",
    "食": "飠", "金": "釒", "肉": "月",
    "足": "⻊", "糸": "糹", "骨": "⻣", "邑": "⻏",
}

BOTTOM_VARIANTS: dict[str, str] = {
    "火": "灬",
    "心": "心",  # stays same at bottom (as in 思)
}

# Known surround radicals per operator.  When the outer component is NOT in
# this set, the renderer uses a non-overlapping fallback layout to prevent
# the inner component from being hidden under a full-coverage glyph.
SURROUND_OUTERS: dict[str, set[str]] = {
    "\u2FF4": {"囗", "口", "回"},
    "\u2FF5": {"門", "冂", "冈", "鬥", "冃"},
    "\u2FF6": {"凵", "山"},
    "\u2FF7": {"匚", "匸", "臣", "巨"},
    "\u2FF8": {"广", "厂", "尸", "疒", "戶", "户", "虍"},
    "\u2FF9": {"戈", "弋", "气"},
    "\u2FFA": {"辶", "辵", "廴", "走", "之", "毛"},
}

RIGHT_VARIANTS: dict[str, str] = {
    "刀": "刂",
    "邑": "⻏",
}


def _get_variant(char: str, operator: str, child_index: int) -> str:
    """Return the positional variant of a radical if one exists."""
    if operator == "\u2FF0":  # ⿰ left-right
        if child_index == 0 and char in LEFT_VARIANTS:
            return LEFT_VARIANTS[char]
        if child_index == 1 and char in RIGHT_VARIANTS:
            return RIGHT_VARIANTS[char]
    elif operator == "\u2FF1":  # ⿱ top-bottom
        if child_index == 1 and char in BOTTOM_VARIANTS:
            return BOTTOM_VARIANTS[char]
    elif operator == "\u2FF2":  # ⿲ left-mid-right
        if child_index == 0 and char in LEFT_VARIANTS:
            return LEFT_VARIANTS[char]
        if child_index == 2 and char in RIGHT_VARIANTS:
            return RIGHT_VARIANTS[char]
    elif operator == "\u2FF3":  # ⿳ top-mid-bottom
        if child_index == 2 and char in BOTTOM_VARIANTS:
            return BOTTOM_VARIANTS[char]
    return char


# --- Stroke bounding box computation ---

def _parse_svg_numbers(path_d: str) -> list[float]:
    """Extract all numeric values from an SVG path d attribute."""
    return [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', path_d)]


def _stroke_bbox(strokes: list[str]) -> Optional[tuple[float, float, float, float]]:
    """Compute bounding box (min_x, min_y, max_x, max_y) from SVG path data."""
    all_x: list[float] = []
    all_y: list[float] = []
    for s in strokes:
        nums = _parse_svg_numbers(s)
        # SVG paths alternate x,y coordinates (roughly — this is an approximation
        # that works well for the Move/Line/Quad/Cubic commands in this dataset)
        for i in range(0, len(nums) - 1, 2):
            all_x.append(nums[i])
            all_y.append(nums[i + 1])
    if not all_x:
        return None
    return min(all_x), min(all_y), max(all_x), max(all_y)


# --- Data-driven subdivision ---

# Proportions derived from statistical analysis of ~6000 ⿰ and ~2000 ⿱ characters
# in Make Me a Hanzi. These are median values.
LEFT_RIGHT_RATIO = 0.41     # left component gets 41% of width
TOP_BOTTOM_RATIO = 0.44     # top component gets 44% of height
THREE_WAY_OUTER = 0.30      # outer columns in ⿲/⿳

# Surround operators: inner component inset
SURROUND_INSET = 0.20       # margin from the outer edge


def _subdivide(bbox: BBox, operator: str, child_index: int, num_children: int) -> BBox:
    """Compute the bounding box for a child given an IDS operator.

    Uses data-driven proportions from real character analysis.
    """
    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h

    if operator == "\u2FF0":  # ⿰ Left-Right
        lw = w * LEFT_RIGHT_RATIO
        rw = w - lw
        if child_index == 0:
            return BBox(x, y, lw, h)
        else:
            return BBox(x + lw, y, rw, h)

    elif operator == "\u2FF1":  # ⿱ Top-Bottom
        th = h * TOP_BOTTOM_RATIO
        bh = h - th
        if child_index == 0:
            return BBox(x, y, w, th)
        else:
            return BBox(x, y + th, w, bh)

    elif operator == "\u2FF2":  # ⿲ Left-Mid-Right
        lw = w * THREE_WAY_OUTER
        rw = w * THREE_WAY_OUTER
        mw = w - lw - rw
        if child_index == 0:
            return BBox(x, y, lw, h)
        elif child_index == 1:
            return BBox(x + lw, y, mw, h)
        else:
            return BBox(x + lw + mw, y, rw, h)

    elif operator == "\u2FF3":  # ⿳ Top-Mid-Bottom
        th = h * THREE_WAY_OUTER
        bh = h * THREE_WAY_OUTER
        mh = h - th - bh
        if child_index == 0:
            return BBox(x, y, w, th)
        elif child_index == 1:
            return BBox(x, y + th, w, mh)
        else:
            return BBox(x, y + th + mh, w, bh)

    elif operator == "\u2FF4":  # ⿴ Full Surround
        if child_index == 0:
            return bbox
        inset = SURROUND_INSET
        return BBox(x + w * inset, y + h * inset,
                    w * (1 - 2 * inset), h * (1 - 2 * inset))

    elif operator == "\u2FF5":  # ⿵ Surround from Above
        if child_index == 0:
            return bbox
        return BBox(x + w * 0.15, y + h * 0.35,
                    w * 0.70, h * 0.58)

    elif operator == "\u2FF6":  # ⿶ Surround from Below
        if child_index == 0:
            return bbox
        return BBox(x + w * 0.15, y + h * 0.05,
                    w * 0.70, h * 0.58)

    elif operator == "\u2FF7":  # ⿷ Surround from Left
        if child_index == 0:
            return bbox
        return BBox(x + w * 0.35, y + h * 0.15,
                    w * 0.58, h * 0.70)

    elif operator == "\u2FF8":  # ⿸ Surround from Upper-Left
        if child_index == 0:
            return bbox
        return BBox(x + w * 0.33, y + h * 0.33,
                    w * 0.62, h * 0.62)

    elif operator == "\u2FF9":  # ⿹ Surround from Upper-Right
        if child_index == 0:
            return bbox
        return BBox(x + w * 0.05, y + h * 0.33,
                    w * 0.62, h * 0.62)

    elif operator == "\u2FFA":  # ⿺ Surround from Lower-Left
        if child_index == 0:
            return bbox
        return BBox(x + w * 0.33, y + h * 0.05,
                    w * 0.62, h * 0.62)

    elif operator == "\u2FFB":  # ⿻ Overlaid
        return bbox

    elif operator == "\u2FFC":  # ⿼ Surround from Right
        if child_index == 0:
            return bbox
        return BBox(x + w * 0.05, y + h * 0.15,
                    w * 0.62, h * 0.70)

    elif operator == "\u2FFD":  # ⿽ Surround from Below-Right
        if child_index == 0:
            return bbox
        return BBox(x + w * 0.05, y + h * 0.05,
                    w * 0.62, h * 0.62)

    elif operator == "\u2FFE":  # ⿾ Mirror (unary)
        return bbox

    elif operator == "\u2FFF":  # ⿿ Rotation (unary)
        return bbox

    elif operator == "\u31EF":  # ㇯ Subtraction
        # Render only the first operand (the base); second is subtracted
        if child_index == 0:
            return bbox
        return BBox(0, 0, 0, 0)  # hide the subtracted part

    # Fallback
    cw = w / num_children
    return BBox(x + child_index * cw, y, cw, h)


def _surround_fallback(bbox: BBox, operator: str, child_index: int) -> BBox:
    """Non-overlapping layout for surround operators with non-standard outers.

    When the outer component isn't a typical surround radical (e.g., 風 in ⿺風日),
    it fills its entire space and the inner component would be hidden underneath.
    This function provides a split layout inspired by the surround direction but
    without overlap.
    """
    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h

    if operator == "\u2FFA":  # ⿺ lower-left → outer bottom-left, inner top-right
        if child_index == 0:
            return BBox(x, y + h * 0.30, w * 0.55, h * 0.70)
        return BBox(x + w * 0.45, y, w * 0.55, h * 0.55)

    elif operator == "\u2FF8":  # ⿸ upper-left → outer top-left, inner bottom-right
        if child_index == 0:
            return BBox(x, y, w * 0.55, h * 0.60)
        return BBox(x + w * 0.40, y + h * 0.35, w * 0.60, h * 0.65)

    elif operator == "\u2FF9":  # ⿹ upper-right → outer top-right, inner bottom-left
        if child_index == 0:
            return BBox(x + w * 0.45, y, w * 0.55, h * 0.60)
        return BBox(x, y + h * 0.35, w * 0.60, h * 0.65)

    elif operator == "\u2FF4":  # ⿴ full surround → treat as ⿰
        if child_index == 0:
            return BBox(x, y, w * 0.50, h)
        return BBox(x + w * 0.50, y, w * 0.50, h)

    elif operator == "\u2FF5":  # ⿵ surround above → treat as ⿱
        if child_index == 0:
            return BBox(x, y, w, h * 0.50)
        return BBox(x, y + h * 0.50, w, h * 0.50)

    elif operator == "\u2FF6":  # ⿶ surround below → treat as ⿱
        if child_index == 0:
            return BBox(x, y + h * 0.50, w, h * 0.50)
        return BBox(x, y, w, h * 0.50)

    elif operator == "\u2FF7":  # ⿷ surround left → treat as ⿰
        if child_index == 0:
            return BBox(x, y, w * 0.50, h)
        return BBox(x + w * 0.50, y, w * 0.50, h)

    # Default: standard subdivide
    return _subdivide(bbox, operator, child_index, 2)


def _render_strokes_tightfit(strokes: list[str], target: BBox,
                             clamp_weight: bool = False,
                             squish: bool = False,
                             align_x: float = 0.5,
                             align_y: float = 0.5,
                             src_bbox: tuple[float, float, float, float] | None = None,
                             squish_blend: float = SQUISH_BLEND,
                             ) -> str:
    """Render strokes with a tight-fit transform.

    Instead of mapping from the full 1024x1024 source space, compute the
    actual bounding box of the strokes and map from that.

    *clamp_weight*: clamp effective source size to prevent over-enlargement.
    *squish*: allow mild non-uniform scaling so components adapt to their box
              shape (e.g., horizontally compressed in narrow ⿰ boxes).
    *align_x*: horizontal alignment (0=left, 0.5=center, 1=right).
    *align_y*: vertical alignment (0=top, 0.5=center, 1=bottom).
    *src_bbox*: pre-computed source bounding box (xMin, yMin, xMax, yMax).
                If None, computed from path data (fine for simple paths, but
                incorrect for complex SVG commands like H/V/C).
    """
    if src_bbox is None:
        src_bbox = _stroke_bbox(strokes)
    if src_bbox is None:
        return ""

    src_x, src_y, src_x2, src_y2 = src_bbox
    src_w = src_x2 - src_x
    src_h = src_y2 - src_y

    if src_w < 1 or src_h < 1:
        return _render_strokes_fullcanvas(strokes, target)

    # Add padding to target
    pad_x = target.w * PADDING
    pad_y = target.h * PADDING
    t = BBox(target.x + pad_x, target.y + pad_y,
             target.w - 2 * pad_x, target.h - 2 * pad_y)

    if clamp_weight:
        # Adaptive clamping: simple/narrow components get clamped more
        # to prevent over-enlargement that distorts proportions.
        # Use bounding box width as a complexity proxy (works for both
        # multi-stroke Make Me a Hanzi data and single-path font glyphs).
        width_ratio = src_w / SOURCE_W  # how much of the em-square the glyph spans
        if width_ratio < 0.40:
            frac = MIN_SOURCE_FRAC + 0.20  # narrow radicals like 氵, 亻
        elif width_ratio < 0.65:
            frac = MIN_SOURCE_FRAC + 0.10  # moderate width
        else:
            frac = MIN_SOURCE_FRAC          # full-width glyphs: minimal clamping
        min_src = SOURCE_W * frac
        eff_w = max(src_w, min_src)
        eff_h = max(src_h, min_src)
    else:
        eff_w = src_w
        eff_h = src_h

    # Base uniform scale
    uniform_scale = min(t.w / eff_w, t.h / eff_h)

    if squish and squish_blend > 0:
        # Compute per-axis scales that would fill the box
        fill_sx = t.w / src_w
        fill_sy = t.h / src_h
        # Blend between uniform and fill, capped by weight clamp
        sx = uniform_scale * (1 - squish_blend) + fill_sx * squish_blend
        sy = uniform_scale * (1 - squish_blend) + fill_sy * squish_blend
        # Don't exceed the weight-clamped uniform scale by too much
        if clamp_weight:
            max_scale = uniform_scale * 2.0
            sx = min(sx, max_scale)
            sy = min(sy, max_scale)
    else:
        sx = sy = uniform_scale

    # Position content within the padded target using alignment
    rendered_w = src_w * sx
    rendered_h = src_h * sy
    offset_x = t.x + (t.w - rendered_w) * align_x
    offset_y = t.y + (t.h - rendered_h) * align_y

    transform = (
        f"translate({offset_x:.2f},{offset_y:.2f}) "
        f"scale({sx:.6f},{sy:.6f}) "
        f"translate({-src_x:.2f},{src_y2:.2f}) "
        f"scale(1,-1)"
    )

    # Compute compensating stroke for weight normalization in compositions.
    stroke_attrs = ""
    if clamp_weight and STROKE_COMPENSATE > 0:
        # MMA data: components scaled down more get thicker outlines.
        # Uses width_ratio as a complexity proxy.
        avg_scale = (abs(sx) + abs(sy)) / 2
        if avg_scale > 0.001:
            stroke_factor = min(width_ratio * 2.0, 1.5)
            sw = STROKE_COMPENSATE * stroke_factor / avg_scale
            sw = min(sw, 8)
            if sw > 1.0:
                stroke_attrs = (
                    f' stroke="currentColor" stroke-width="{sw:.1f}"'
                    f' stroke-linejoin="round" paint-order="stroke fill"'
                )
    elif not clamp_weight and squish:
        # Font data: non-uniform scaling thins strokes on the compressed
        # axis.  Add a stroke in font units proportional to scale loss
        # vs fullcanvas.  This scales with display size so the relative
        # thickening is consistent at all sizes.
        avg_scale = (abs(sx) + abs(sy)) / 2
        if avg_scale > 0.001:
            # fullcanvas_scale ≈ viewbox_size / em_units; approximate from
            # the target box (which is close to the viewbox size).
            fullcanvas_scale = max(target.w, target.h) / max(src_w, src_h)
            if avg_scale < fullcanvas_scale:
                # How much thinner are we vs fullcanvas? Add proportional stroke.
                deficit = fullcanvas_scale / avg_scale - 1.0
                sw = 12.0 * deficit  # ~12 font units per 1x deficit
                sw = min(sw, 30)
                if sw > 2.0:
                    stroke_attrs = (
                        f' stroke="currentColor" stroke-width="{sw:.1f}"'
                        f' stroke-linejoin="round" paint-order="stroke fill"'
                    )

    parts = [f'<g transform="{transform}"{stroke_attrs}>']
    for stroke in strokes:
        parts.append(f'  <path d="{stroke}" fill="currentColor" />')
    parts.append("</g>")
    return "\n".join(parts)


def _render_strokes_fullcanvas(strokes: list[str], bbox: BBox,
                               source_w: float = SOURCE_W,
                               source_h: float = SOURCE_H,
                               source_y_offset: float = SOURCE_Y_OFFSET) -> str:
    """Render strokes mapping from the full source em-square.

    Used for complete characters where strokes are already positioned
    correctly within the full canvas.
    """
    sx = bbox.w / source_w
    sy = bbox.h / source_h
    transform = (
        f"translate({bbox.x:.2f},{bbox.y:.2f}) "
        f"scale({sx:.6f},{sy:.6f}) "
        f"translate(0,{source_y_offset:.0f}) "
        f"scale(1,-1)"
    )
    parts = [f'<g transform="{transform}">']
    for stroke in strokes:
        parts.append(f'  <path d="{stroke}" fill="currentColor" />')
    parts.append("</g>")
    return "\n".join(parts)


def _render_placeholder(char: str, bbox: BBox) -> str:
    """Render a subtle placeholder for a missing glyph (just the character)."""
    cx = bbox.x + bbox.w / 2
    cy = bbox.y + bbox.h / 2
    font_size = min(bbox.w, bbox.h) * 0.6
    return (
        f'<text x="{cx:.1f}" y="{cy:.1f}" '
        f'text-anchor="middle" dominant-baseline="central" '
        f'font-size="{font_size:.1f}" font-family={SVG_FONT!r} '
        f'fill="currentColor">'
        f'{escape(char)}</text>'
    )


_FONT_CSS = (
    '@import url("https://fonts.googleapis.com/css2?'
    'family=Noto+Serif+TC:wght@400;700&display=swap");'
)


def _svg_document(content: str, size: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {size} {size}" '
        f'width="{size}" height="{size}" '
        f'style="color: #000">\n'
        f'<style>{_FONT_CSS}</style>\n'
        f'{content}\n'
        f'</svg>'
    )


class Renderer:
    """Renders IDS trees to SVG with proper radical integration.

    Two rendering strategies:
    1. **Donor-based**: For IDS patterns that match a known character,
       extract pre-drawn component strokes using the matches field.
       These strokes are already optimized for their compositional context.
    2. **Algorithmic**: For novel/surreal patterns with no donor character,
       compose standalone component glyphs using data-driven proportions.

    Accepts either GlyphData (Make Me a Hanzi) or FontGlyphData (OTF font).
    The coordinate system is auto-detected from the glyph data source.
    """

    def __init__(self, glyph_data) -> None:
        self.glyph_data = glyph_data
        self._decomp_index: Optional[dict[str, list[str]]] = None

        # Detect coordinate system from glyph data
        if hasattr(glyph_data, 'source_w'):
            # FontGlyphData — font-based coordinate system.
            # Higher squish: font strokes are designed to handle deformation,
            # and real CJK components DO fill their allocated space.
            self._source_w = glyph_data.source_w
            self._source_h = glyph_data.source_h
            self._source_y_offset = glyph_data.source_y_offset
            self._is_font_data = True
            self._squish_blend = 0.90
        else:
            # GlyphData — Make Me a Hanzi coordinate system
            self._source_w = SOURCE_W
            self._source_h = SOURCE_H
            self._source_y_offset = SOURCE_Y_OFFSET
            self._is_font_data = False
            self._squish_blend = SQUISH_BLEND

    def _get_decomp_index(self) -> dict[str, list[str]]:
        """Build reverse index: IDS decomposition string -> list of characters."""
        if self._decomp_index is None:
            self._decomp_index = {}
            strokes_dict = getattr(self.glyph_data, '_strokes', None)
            if strokes_dict:
                for char in strokes_dict:
                    decomp = self.glyph_data.get_decomposition(char)
                    if decomp:
                        self._decomp_index.setdefault(decomp, []).append(char)
        return self._decomp_index

    def _find_donor(self, ids_string: str) -> Optional[str]:
        """Find a known character that matches an IDS decomposition.

        Returns a character with stroke data and matches, or None.
        """
        index = self._get_decomp_index()
        candidates = index.get(ids_string, [])
        for char in candidates:
            if (self.glyph_data.get_strokes(char) is not None
                    and self.glyph_data.get_matches(char) is not None):
                return char
        return None

    def _render_via_donor(self, donor_char: str, tree: IDSNode, bbox: BBox,
                          in_composition: bool = False) -> Optional[str]:
        """Render using pre-drawn component strokes from a donor character.

        Extracts strokes for each top-level component using the matches field.
        When used at the root level, maps from the full 1024x1024 canvas.
        When used inside a composition, uses tight-fit to fill the sub-bbox.
        """
        all_strokes = self.glyph_data.get_strokes(donor_char)
        matches = self.glyph_data.get_matches(donor_char)
        if all_strokes is None or matches is None:
            return None

        # Verify the tree has the expected structure
        if tree.is_leaf:
            return None

        # Collect all matched strokes (preserving relative positioning)
        collected: list[str] = []
        for i, match in enumerate(matches):
            if match is not None and len(match) > 0 and i < len(all_strokes):
                collected.append(all_strokes[i])

        if not collected:
            return None

        if in_composition:
            return _render_strokes_tightfit(collected, bbox)
        return _render_strokes_fullcanvas(
            collected, bbox,
            self._source_w, self._source_h, self._source_y_offset,
        )

    def _resolve_variant(self, node: IDSNode, operator: str, child_index: int) -> IDSNode:
        """Apply radical variant mapping to a leaf node."""
        if not node.is_leaf or node.character is None:
            return node
        variant = _get_variant(node.character, operator, child_index)
        if variant != node.character and self.glyph_data.has_char(variant):
            return IDSNode(character=variant)
        return node

    @staticmethod
    def _alignment_for(parent_op: Optional[str], child_idx: int) -> tuple[float, float]:
        """Compute (align_x, align_y) so siblings lean toward each other."""
        if parent_op == "\u2FF0":  # ⿰ left-right
            return (1.0, 0.5) if child_idx == 0 else (0.0, 0.5)
        if parent_op == "\u2FF1":  # ⿱ top-bottom
            return (0.5, 1.0) if child_idx == 0 else (0.5, 0.0)
        if parent_op == "\u2FF2":  # ⿲ left-mid-right
            if child_idx == 0:
                return (1.0, 0.5)
            if child_idx == 2:
                return (0.0, 0.5)
            return (0.5, 0.5)
        if parent_op == "\u2FF3":  # ⿳ top-mid-bottom
            if child_idx == 0:
                return (0.5, 1.0)
            if child_idx == 2:
                return (0.5, 0.0)
            return (0.5, 0.5)
        return (0.5, 0.5)

    def _leaf_size(self, node: IDSNode) -> tuple[float, float] | None:
        """Get the natural (width, height) of a leaf node's glyph."""
        if not node.is_leaf or node.character is None:
            return None
        bb = self._get_bbox(node.character)
        if bb:
            return (bb[2] - bb[0], bb[3] - bb[1])
        # Fallback: try naive bbox from stroke data
        strokes = self.glyph_data.get_strokes(node.character)
        if strokes:
            sb = _stroke_bbox(strokes)
            if sb:
                return (sb[2] - sb[0], sb[3] - sb[1])
        return None

    def _subtree_size(self, node: IDSNode) -> tuple[float, float] | None:
        """Recursively estimate natural (width, height) of any IDS subtree.

        For leaves, returns the glyph's natural size.
        For composites, combines children based on the operator type.
        """
        if node.is_leaf:
            # Resolve variant before measuring
            if node.character:
                return self._leaf_size(node)
            return None

        child_sizes = [self._subtree_size(c) for c in node.children]
        if any(s is None for s in child_sizes):
            return None

        op = node.operator
        if op in ("\u2FF0", "\u2FF2"):  # ⿰⿲ left-right
            w = sum(s[0] for s in child_sizes)
            h = max(s[1] for s in child_sizes)
            return (w, h)
        elif op in ("\u2FF1", "\u2FF3"):  # ⿱⿳ top-bottom
            w = max(s[0] for s in child_sizes)
            h = sum(s[1] for s in child_sizes)
            return (w, h)
        elif op == "\u2FFB":  # ⿻ overlaid
            w = max(s[0] for s in child_sizes)
            h = max(s[1] for s in child_sizes)
            return (w, h)
        else:  # surround operators — use outer child's size
            return child_sizes[0]

    def _compute_split_ratios(self, operator: str,
                              children: list[IDSNode]) -> list[float] | None:
        """Compute proportional split ratios from component natural sizes.

        Returns a list of ratios (summing to 1.0) for dividing the parent box,
        or None to fall back to fixed ratios.
        """
        if operator == "\u2FF0":  # ⿰ left-right: split by width
            sizes = [self._subtree_size(c) for c in children]
            if all(s is not None for s in sizes):
                widths = [s[0] for s in sizes]
                # In CJK typography, the left radical is narrower than its
                # standalone form. Scale left component width down to match
                # the convention (real fonts compress left radicals ~85-90%).
                widths[0] *= 0.87
                total = sum(widths)
                if total > 0:
                    return [w / total for w in widths]
        elif operator == "\u2FF1":  # ⿱ top-bottom: split by height
            sizes = [self._subtree_size(c) for c in children]
            if all(s is not None for s in sizes):
                heights = [s[1] for s in sizes]
                total = sum(heights)
                if total > 0:
                    return [h / total for h in heights]
        elif operator == "\u2FF2":  # ⿲ left-mid-right: split by width
            sizes = [self._subtree_size(c) for c in children]
            if all(s is not None for s in sizes):
                widths = [s[0] for s in sizes]
                total = sum(widths)
                if total > 0:
                    return [w / total for w in widths]
        elif operator == "\u2FF3":  # ⿳ top-mid-bottom: split by height
            sizes = [self._subtree_size(c) for c in children]
            if all(s is not None for s in sizes):
                heights = [s[1] for s in sizes]
                total = sum(heights)
                if total > 0:
                    return [h / total for h in heights]
        return None

    @staticmethod
    def _dynamic_subdivide(bbox: BBox, operator: str, child_index: int,
                           ratios: list[float]) -> BBox:
        """Subdivide a bbox using dynamic ratios instead of fixed ones."""
        x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h

        if operator in ("\u2FF0", "\u2FF2"):  # left-right splits
            offset = sum(ratios[:child_index]) * w
            child_w = ratios[child_index] * w
            return BBox(x + offset, y, child_w, h)
        elif operator in ("\u2FF1", "\u2FF3"):  # top-bottom splits
            offset = sum(ratios[:child_index]) * h
            child_h = ratios[child_index] * h
            return BBox(x, y + offset, w, child_h)
        # Fallback
        return _subdivide(bbox, operator, child_index, len(ratios))

    def _get_bbox(self, char: str) -> tuple[float, float, float, float] | None:
        """Get correct bounding box for a character.

        Uses FontGlyphData.get_bbox() when available (correct for complex
        SVG paths), otherwise falls back to _stroke_bbox (fine for MMA data).
        """
        get_bbox = getattr(self.glyph_data, 'get_bbox', None)
        if get_bbox:
            return get_bbox(char)
        return None  # let _render_strokes_tightfit compute via _stroke_bbox

    def _render_node(self, node: IDSNode, bbox: BBox,
                     parent_op: Optional[str] = None,
                     child_idx: int = 0) -> str:
        """Recursively render an IDS node into SVG elements."""
        if node.is_leaf:
            char = node.character or "?"
            strokes = self.glyph_data.get_strokes(char)
            if strokes:
                if parent_op == "\u2FFB":
                    # ⿻ Overlaid: use full-canvas so both children preserve
                    # their natural em-square positions. Without this, small
                    # components like 丶 get enlarged to fill the full box.
                    return _render_strokes_fullcanvas(
                        strokes, bbox,
                        self._source_w, self._source_h,
                        self._source_y_offset,
                    )
                if parent_op is not None:
                    ax, ay = self._alignment_for(parent_op, child_idx)
                    # Font glyphs have balanced weights from the designer —
                    # skip weight clamping and stroke compensation.
                    # MMA data needs both to normalize varying stroke weights.
                    cw = not self._is_font_data
                    return _render_strokes_tightfit(
                        strokes, bbox,
                        clamp_weight=cw, squish=True,
                        align_x=ax, align_y=ay,
                        src_bbox=self._get_bbox(char),
                        squish_blend=self._squish_blend,
                    )
                return _render_strokes_fullcanvas(
                    strokes, bbox,
                    self._source_w, self._source_h, self._source_y_offset,
                )

            # No stroke data — try decomposition before giving up
            decomp = self.glyph_data.get_decomposition(char)
            if decomp:
                try:
                    subtree = parse_ids(decomp)
                    return self._render_node(subtree, bbox, parent_op, child_idx)
                except Exception:
                    pass

            return _render_placeholder(char, bbox)

        # Try donor-based rendering: find a known character with this exact
        # IDS decomposition and use its pre-drawn component strokes
        ids_str = node.to_ids()
        donor = self._find_donor(ids_str)
        if donor:
            result = self._render_via_donor(
                donor, node, bbox, in_composition=parent_op is not None
            )
            if result is not None:
                return result

        # Fall back to algorithmic composition
        # Resolve variants first so we can measure their natural sizes
        resolved_children = [
            self._resolve_variant(child, node.operator, i)
            for i, child in enumerate(node.children)
        ]

        # For surround operators with non-standard outers, use a
        # non-overlapping layout instead of the normal overlapping one.
        use_fallback_layout = False
        if (node.operator in SURROUND_OUTERS
                and len(resolved_children) >= 2):
            outer = resolved_children[0]
            if outer.is_leaf and outer.character:
                known = SURROUND_OUTERS.get(node.operator, set())
                if outer.character not in known:
                    use_fallback_layout = True

        # Compute dynamic split ratio based on component natural widths/heights
        split_ratios = self._compute_split_ratios(node.operator, resolved_children)

        parts = []
        for i, resolved in enumerate(resolved_children):
            if use_fallback_layout:
                child_bbox = _surround_fallback(
                    bbox, node.operator, i)
            elif split_ratios:
                child_bbox = self._dynamic_subdivide(
                    bbox, node.operator, i, split_ratios)
            else:
                child_bbox = _subdivide(
                    bbox, node.operator, i, len(node.children))
            parts.append(self._render_node(resolved, child_bbox, node.operator, i))
        return "\n".join(parts)

    def render_ids(self, ids_string: str, size: int = 256) -> str:
        """Render an IDS string to a complete SVG document."""
        tree = parse_ids(ids_string)
        if tree.is_leaf:
            # Standalone character — natural margins already in source data
            bbox = BBox(0, 0, float(size), float(size))
        else:
            # Composed character — add margins to match standalone glyph proportions.
            # Without this, tight-fit fills edge-to-edge and the character looks
            # oversized compared to real font glyphs at the same em size.
            margin = size * (1 - EM_BODY) / 2
            bbox = BBox(margin, margin,
                        float(size) - 2 * margin, float(size) - 2 * margin)
        content = self._render_node(tree, bbox)
        return _svg_document(content, size)

    def render_char(self, char: str, size: int = 256) -> str:
        """Render a character.

        If stroke data exists, renders the whole character directly.
        If only decomposition exists, renders via IDS composition.
        """
        strokes = self.glyph_data.get_strokes(char)
        if strokes:
            bbox = BBox(0, 0, float(size), float(size))
            content = _render_strokes_fullcanvas(
                strokes, bbox,
                self._source_w, self._source_h, self._source_y_offset,
            )
            return _svg_document(content, size)

        decomp = self.glyph_data.get_decomposition(char)
        if decomp:
            return self.render_ids(decomp, size)

        bbox = BBox(0, 0, float(size), float(size))
        content = _render_placeholder(char, bbox)
        return _svg_document(content, size)

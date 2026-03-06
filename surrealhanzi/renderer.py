"""SVG composition engine for IDS-based character rendering.

Uses data-driven proportions, radical variant mapping, and tight-fit
transforms to produce well-integrated character compositions.
"""

import re
from dataclasses import dataclass
from typing import Optional
from xml.sax.saxutils import escape

from .glyph_data import GlyphData
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
PADDING = 0.04


# --- Radical variant mapping ---
# When a radical appears in certain positions, use its positional variant.
# Key: (standalone_radical, position) -> variant_character
# Positions: "left", "right", "top", "bottom", "inner", "outer"

# Map from standalone form to positional variant
LEFT_VARIANTS: dict[str, str] = {
    "人": "亻", "水": "氵", "手": "扌", "心": "忄",
    "犬": "犭", "示": "礻", "衣": "衤", "玉": "王",
    "食": "飠", "金": "釒",
    # Simplified forms
    "言": "讠", "貝": "贝",
}

BOTTOM_VARIANTS: dict[str, str] = {
    "火": "灬",
    "心": "心",  # stays same at bottom (as in 思)
}

RIGHT_VARIANTS: dict[str, str] = {
    "刀": "刂",
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

    # Fallback
    cw = w / num_children
    return BBox(x + child_index * cw, y, cw, h)


def _render_strokes_tightfit(strokes: list[str], target: BBox) -> str:
    """Render strokes with a tight-fit transform.

    Instead of mapping from the full 1024x1024 source space, compute the
    actual bounding box of the strokes and map from that, so strokes fill
    the target area properly with small padding.
    """
    src_bbox = _stroke_bbox(strokes)
    if src_bbox is None:
        return ""

    src_x, src_y, src_x2, src_y2 = src_bbox
    src_w = src_x2 - src_x
    src_h = src_y2 - src_y

    if src_w < 1 or src_h < 1:
        # Degenerate — fall back to full-canvas mapping
        return _render_strokes_fullcanvas(strokes, target)

    # Add padding to target
    pad_x = target.w * PADDING
    pad_y = target.h * PADDING
    t = BBox(target.x + pad_x, target.y + pad_y,
             target.w - 2 * pad_x, target.h - 2 * pad_y)

    # Uniform scale to preserve aspect ratio
    scale = min(t.w / src_w, t.h / src_h)

    # Center within the padded target
    rendered_w = src_w * scale
    rendered_h = src_h * scale
    offset_x = t.x + (t.w - rendered_w) / 2
    offset_y = t.y + (t.h - rendered_h) / 2

    # Transform chain (applied right-to-left):
    # 1. Flip y: scale(1, -1)
    # 2. Shift source bbox origin to (0,0): translate(-src_x, src_y2)
    #    (after y-flip, src_y2 becomes the top)
    # 3. Scale to target: scale(scale, scale)
    # 4. Translate to target position: translate(offset_x, offset_y)
    transform = (
        f"translate({offset_x:.2f},{offset_y:.2f}) "
        f"scale({scale:.6f},{scale:.6f}) "
        f"translate({-src_x:.2f},{src_y2:.2f}) "
        f"scale(1,-1)"
    )

    parts = [f'<g transform="{transform}">']
    for stroke in strokes:
        parts.append(f'  <path d="{stroke}" fill="currentColor" />')
    parts.append("</g>")
    return "\n".join(parts)


def _render_strokes_fullcanvas(strokes: list[str], bbox: BBox) -> str:
    """Render strokes mapping from full 1024x1024 source canvas.

    Used for complete characters where strokes are already positioned
    correctly within the full canvas.
    """
    sx = bbox.w / SOURCE_W
    sy = bbox.h / SOURCE_H
    transform = (
        f"translate({bbox.x:.2f},{bbox.y:.2f}) "
        f"scale({sx:.6f},{sy:.6f}) "
        f"translate(0,{SOURCE_Y_OFFSET:.0f}) "
        f"scale(1,-1)"
    )
    parts = [f'<g transform="{transform}">']
    for stroke in strokes:
        parts.append(f'  <path d="{stroke}" fill="currentColor" />')
    parts.append("</g>")
    return "\n".join(parts)


def _render_placeholder(char: str, bbox: BBox) -> str:
    """Render a placeholder for a missing glyph."""
    cx = bbox.x + bbox.w / 2
    cy = bbox.y + bbox.h / 2
    font_size = min(bbox.w, bbox.h) * 0.6
    return (
        f'<rect x="{bbox.x:.1f}" y="{bbox.y:.1f}" '
        f'width="{bbox.w:.1f}" height="{bbox.h:.1f}" '
        f'fill="none" stroke="#ccc" stroke-width="1" stroke-dasharray="4,4" />\n'
        f'<text x="{cx:.1f}" y="{cy:.1f}" '
        f'text-anchor="middle" dominant-baseline="central" '
        f'font-size="{font_size:.1f}" fill="#999">'
        f'{escape(char)}</text>'
    )


def _svg_document(content: str, size: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {size} {size}" '
        f'width="{size}" height="{size}" '
        f'style="color: #000">\n'
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
    """

    def __init__(self, glyph_data: GlyphData) -> None:
        self.glyph_data = glyph_data
        self._decomp_index: Optional[dict[str, list[str]]] = None

    def _get_decomp_index(self) -> dict[str, list[str]]:
        """Build reverse index: IDS decomposition string -> list of characters."""
        if self._decomp_index is None:
            self._decomp_index = {}
            for char in self.glyph_data._strokes:
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

    def _render_via_donor(self, donor_char: str, tree: IDSNode, bbox: BBox) -> Optional[str]:
        """Render using pre-drawn component strokes from a donor character.

        Extracts strokes for each top-level component using the matches field,
        then renders them in the target bbox with full-canvas mapping.
        """
        all_strokes = self.glyph_data.get_strokes(donor_char)
        matches = self.glyph_data.get_matches(donor_char)
        if all_strokes is None or matches is None:
            return None

        # Verify the tree has the expected structure
        if tree.is_leaf:
            return None

        # Group strokes by top-level component index
        component_groups: dict[int, list[str]] = {}
        for i, match in enumerate(matches):
            if match is not None and len(match) > 0 and i < len(all_strokes):
                comp_idx = match[0]
                component_groups.setdefault(comp_idx, []).append(all_strokes[i])

        # Each component gets rendered with full-canvas mapping
        # (strokes are already at correct positions within 1024x1024)
        parts = []
        for comp_idx in sorted(component_groups.keys()):
            strokes = component_groups[comp_idx]
            parts.append(_render_strokes_fullcanvas(strokes, bbox))

        if not parts:
            return None
        return "\n".join(parts)

    def _resolve_variant(self, node: IDSNode, operator: str, child_index: int) -> IDSNode:
        """Apply radical variant mapping to a leaf node."""
        if not node.is_leaf or node.character is None:
            return node
        variant = _get_variant(node.character, operator, child_index)
        if variant != node.character and self.glyph_data.has_char(variant):
            return IDSNode(character=variant)
        return node

    def _render_node(self, node: IDSNode, bbox: BBox,
                     parent_op: Optional[str] = None,
                     child_idx: int = 0) -> str:
        """Recursively render an IDS node into SVG elements."""
        if node.is_leaf:
            char = node.character or "?"
            strokes = self.glyph_data.get_strokes(char)
            if strokes:
                return _render_strokes_fullcanvas(strokes, bbox)
            else:
                return _render_placeholder(char, bbox)

        # Try donor-based rendering: find a known character with this exact
        # IDS decomposition and use its pre-drawn component strokes
        ids_str = node.to_ids()
        donor = self._find_donor(ids_str)
        if donor:
            result = self._render_via_donor(donor, node, bbox)
            if result is not None:
                return result

        # Fall back to algorithmic composition
        parts = []
        for i, child in enumerate(node.children):
            resolved = self._resolve_variant(child, node.operator, i)
            child_bbox = _subdivide(bbox, node.operator, i, len(node.children))
            parts.append(self._render_node(resolved, child_bbox, node.operator, i))
        return "\n".join(parts)

    def render_ids(self, ids_string: str, size: int = 256) -> str:
        """Render an IDS string to a complete SVG document."""
        tree = parse_ids(ids_string)
        bbox = BBox(0, 0, float(size), float(size))
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
            content = _render_strokes_fullcanvas(strokes, bbox)
            return _svg_document(content, size)

        decomp = self.glyph_data.get_decomposition(char)
        if decomp:
            return self.render_ids(decomp, size)

        bbox = BBox(0, 0, float(size), float(size))
        content = _render_placeholder(char, bbox)
        return _svg_document(content, size)

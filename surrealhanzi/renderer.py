"""SVG composition engine for IDS-based character rendering."""

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
# - So visible range is x: [0, 1024], y: [-124, 900]
# - To render in normal SVG (y increases downward), we need to flip y
SOURCE_W = 1024.0
SOURCE_H = 1024.0
SOURCE_Y_OFFSET = 900.0  # The "top" in source coordinates


def _subdivide(bbox: BBox, operator: str, child_index: int, num_children: int) -> BBox:
    """Compute the bounding box for a child given an IDS operator."""
    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h

    if operator == "\u2FF0":  # ⿰ Left-Right
        cw = w / num_children
        return BBox(x + child_index * cw, y, cw, h)

    elif operator == "\u2FF1":  # ⿱ Top-Bottom
        ch = h / num_children
        return BBox(x, y + child_index * ch, w, ch)

    elif operator == "\u2FF2":  # ⿲ Left-Mid-Right
        cw = w / 3
        return BBox(x + child_index * cw, y, cw, h)

    elif operator == "\u2FF3":  # ⿳ Top-Mid-Bottom
        ch = h / 3
        return BBox(x, y + child_index * ch, w, ch)

    elif operator == "\u2FF4":  # ⿴ Full Surround
        if child_index == 0:  # outer
            return bbox
        else:  # inner, centered at 60%
            iw, ih = w * 0.55, h * 0.55
            return BBox(x + (w - iw) / 2, y + (h - ih) / 2, iw, ih)

    elif operator == "\u2FF5":  # ⿵ Surround from Above
        if child_index == 0:
            return bbox
        else:  # inner, in bottom-center area
            iw, ih = w * 0.6, h * 0.55
            return BBox(x + (w - iw) / 2, y + h * 0.4, iw, ih)

    elif operator == "\u2FF6":  # ⿶ Surround from Below
        if child_index == 0:
            return bbox
        else:  # inner, in top-center area
            iw, ih = w * 0.6, h * 0.55
            return BBox(x + (w - iw) / 2, y + h * 0.05, iw, ih)

    elif operator == "\u2FF7":  # ⿷ Surround from Left
        if child_index == 0:
            return bbox
        else:  # inner, in right-center area
            iw, ih = w * 0.55, h * 0.6
            return BBox(x + w * 0.4, y + (h - ih) / 2, iw, ih)

    elif operator == "\u2FF8":  # ⿸ Surround from Upper-Left
        if child_index == 0:
            return bbox
        else:  # inner, in lower-right
            iw, ih = w * 0.6, h * 0.6
            return BBox(x + w * 0.35, y + h * 0.35, iw, ih)

    elif operator == "\u2FF9":  # ⿹ Surround from Upper-Right
        if child_index == 0:
            return bbox
        else:  # inner, in lower-left
            iw, ih = w * 0.6, h * 0.6
            return BBox(x + w * 0.05, y + h * 0.35, iw, ih)

    elif operator == "\u2FFA":  # ⿺ Surround from Lower-Left
        if child_index == 0:
            return bbox
        else:  # inner, in upper-right
            iw, ih = w * 0.6, h * 0.6
            return BBox(x + w * 0.35, y + h * 0.05, iw, ih)

    elif operator == "\u2FFB":  # ⿻ Overlaid
        return bbox  # both children fill the same box

    # Fallback: equal split horizontally
    cw = w / num_children
    return BBox(x + child_index * cw, y, cw, h)


def _render_leaf_strokes(strokes: list[str], bbox: BBox) -> str:
    """Render stroke paths into an SVG <g> element, transformed to fit bbox.

    Source coordinates: 1024x1024, origin at (0, 900), y-axis goes up.
    Target: bbox in normal SVG coordinates (y-axis goes down).
    """
    sx = bbox.w / SOURCE_W
    sy = bbox.h / SOURCE_H

    # Transform chain (applied right-to-left):
    # 1. In source space, flip y: y' = -y
    # 2. Translate so top-left of visible area is at origin: translate(0, SOURCE_Y_OFFSET)
    #    After flip, the visible range [-124, 900] becomes [-900, 124]
    #    translate(0, 900) maps this to [0, 1024]
    # 3. Scale to target size: scale(sx, sy)
    # 4. Translate to target position: translate(bbox.x, bbox.y)
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


class Renderer:
    """Renders IDS trees to SVG."""

    def __init__(self, glyph_data: GlyphData) -> None:
        self.glyph_data = glyph_data

    def _render_node(self, node: IDSNode, bbox: BBox) -> str:
        """Recursively render an IDS node into SVG elements."""
        if node.is_leaf:
            char = node.character or "?"
            strokes = self.glyph_data.get_strokes(char)
            if strokes:
                return _render_leaf_strokes(strokes, bbox)
            else:
                return _render_placeholder(char, bbox)

        parts = []
        for i, child in enumerate(node.children):
            child_bbox = _subdivide(bbox, node.operator, i, len(node.children))
            parts.append(self._render_node(child, child_bbox))
        return "\n".join(parts)

    def render_ids(self, ids_string: str, size: int = 256) -> str:
        """Render an IDS string to a complete SVG document."""
        tree = parse_ids(ids_string)
        bbox = BBox(0, 0, float(size), float(size))
        content = self._render_node(tree, bbox)

        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {size} {size}" '
            f'width="{size}" height="{size}" '
            f'style="color: #000">\n'
            f'{content}\n'
            f'</svg>'
        )

    def render_char(self, char: str, size: int = 256) -> str:
        """Render a character, using its IDS decomposition if available.

        If the character has stroke data, renders it directly.
        If it has an IDS decomposition, renders via composition.
        """
        # If we have direct stroke data, render the whole character
        strokes = self.glyph_data.get_strokes(char)
        if strokes:
            bbox = BBox(0, 0, float(size), float(size))
            content = _render_leaf_strokes(strokes, bbox)
            return (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'viewBox="0 0 {size} {size}" '
                f'width="{size}" height="{size}" '
                f'style="color: #000">\n'
                f'{content}\n'
                f'</svg>'
            )

        # Try IDS decomposition
        decomp = self.glyph_data.get_decomposition(char)
        if decomp:
            return self.render_ids(decomp, size)

        # Fallback: placeholder
        bbox = BBox(0, 0, float(size), float(size))
        content = _render_placeholder(char, bbox)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {size} {size}" '
            f'width="{size}" height="{size}" '
            f'style="color: #000">\n'
            f'{content}\n'
            f'</svg>'
        )

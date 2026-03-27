"""Grading system for rendered character quality.

Renders all dictionary entries, runs automated quality checks,
and generates an HTML report for visual review.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional

import yaml

from .ids_parser import IDSParseError, collect_leaves, parse_ids
from .renderer import Renderer


@dataclass
class Issue:
    """A single quality issue found during grading."""
    severity: str  # "error", "warning", "info"
    message: str


@dataclass
class GradeResult:
    """Grading result for a single character."""
    entry_id: str
    ids: str
    title: str
    svg: str
    grade: str  # A, B, C, F
    issues: list[Issue] = field(default_factory=list)
    leaves: list[str] = field(default_factory=list)
    depth: int = 0
    num_components: int = 0


def _ids_depth(ids_string: str) -> int:
    """Compute nesting depth of an IDS expression."""
    try:
        tree = parse_ids(ids_string)
    except IDSParseError:
        return 0

    def _depth(node) -> int:
        if node.is_leaf:
            return 0
        return 1 + max((_depth(c) for c in node.children), default=0)

    return _depth(tree)


def _analyze_transforms(svg: str) -> list[tuple[float, float]]:
    """Extract (sx, sy) scale pairs from SVG transform attributes.

    Only extracts the composition-level scales (the first scale in each
    transform chain), skipping the final scale(1,-1) y-flip.
    """
    scales = []
    for match in re.finditer(r'scale\(([^)]+)\)', svg):
        parts = match.group(1).split(',')
        if len(parts) == 2:
            sx, sy = float(parts[0]), float(parts[1])
            # Skip the y-flip scale(1,-1)
            if sx == 1.0 and sy == -1.0:
                continue
            scales.append((abs(sx), abs(sy)))
    return scales


def _analyze_stroke_weights(svg: str) -> tuple[list[float], list[float]]:
    """Extract stroke-width values and visual stroke weights from SVG.

    Returns (raw_widths, visual_widths) where visual_widths = stroke_width * avg_scale.
    Visual weights are what the viewer actually sees — comparing these is more
    meaningful than comparing raw SVG attribute values.
    """
    raw_widths: list[float] = []
    visual_widths: list[float] = []
    # Match <g transform="..." stroke-width="..."> groups
    pattern = re.compile(
        r'<g\s+transform="[^"]*scale\(([^)]+)\)[^"]*"'
        r'[^>]*stroke-width="([^"]+)"')
    for match in pattern.finditer(svg):
        s_parts = match.group(1).split(',')
        sw = float(match.group(2))
        raw_widths.append(sw)
        if len(s_parts) == 2:
            sx, sy = abs(float(s_parts[0])), abs(float(s_parts[1]))
            avg_scale = (sx + sy) / 2
            visual_widths.append(sw * avg_scale)
        else:
            visual_widths.append(sw)
    return raw_widths, visual_widths


def _analyze_component_areas(svg: str, size: int = 256) -> list[float]:
    """Estimate the fraction of the viewbox each component group occupies.

    Parses the translate+scale transforms to estimate rendered area.
    """
    areas = []
    pattern = re.compile(
        r'translate\(([^)]+)\)\s+scale\(([^)]+)\)\s+translate\(([^)]+)\)')
    for match in pattern.finditer(svg):
        try:
            tx_parts = match.group(1).split(',')
            s_parts = match.group(2).split(',')
            if len(s_parts) == 2:
                sx, sy = abs(float(s_parts[0])), abs(float(s_parts[1]))
                # Rough rendered area in viewbox units
                # Source data is ~1024x1024, so rendered size ≈ 1024*scale
                area = (1024 * sx * 1024 * sy) / (size * size)
                areas.append(area)
        except (ValueError, IndexError):
            continue
    return areas


# Surround operators and their inner box insets (from renderer._subdivide)
_SURROUND_OPS: dict[str, tuple[float, float, float, float]] = {
    "\u2FF4": (0.20, 0.20, 0.60, 0.60),  # ⿴ Full Surround
    "\u2FF5": (0.15, 0.35, 0.70, 0.58),  # ⿵ Surround from Above
    "\u2FF6": (0.15, 0.05, 0.70, 0.58),  # ⿶ Surround from Below
    "\u2FF7": (0.35, 0.15, 0.58, 0.70),  # ⿷ Surround from Left
    "\u2FF8": (0.33, 0.33, 0.62, 0.62),  # ⿸ Surround from Upper-Left
    "\u2FF9": (0.05, 0.33, 0.62, 0.62),  # ⿹ Surround from Upper-Right
    "\u2FFA": (0.33, 0.05, 0.62, 0.62),  # ⿺ Surround from Lower-Left
    "\u2FFB": (0.00, 0.00, 1.00, 1.00),  # ⿻ Overlaid (100% overlap)
    "\u2FFC": (0.05, 0.15, 0.62, 0.70),  # ⿼ Surround from Right
    "\u2FFD": (0.05, 0.05, 0.62, 0.62),  # ⿽ Surround from Below-Right
}

# Outer radicals that are known to have natural openings for their operator.
# Characters not in this set will be flagged for potential overlap.
_KNOWN_SURROUND_OUTERS: dict[str, set[str]] = {
    "\u2FF4": {"囗", "口", "回"},
    "\u2FF5": {"門", "冂", "冈", "鬥", "冃", "同", "网", "罔"},
    "\u2FF6": {"凵", "山", "凶"},
    "\u2FF7": {"匚", "匸", "臣", "巨"},
    "\u2FF8": {"广", "厂", "尸", "疒", "戶", "户", "麻", "虍", "庇",
               "厄", "痒"},
    "\u2FF9": {"戈", "弋", "气", "戊", "成", "戌", "武", "我", "伐"},
    "\u2FFA": {"辶", "辵", "廴", "走", "之", "毛", "鬼", "是"},
    "\u2FFB": set(),  # overlaid — everything is expected to overlap
    "\u2FFC": set(),
    "\u2FFD": set(),
}


def _estimate_overlap(outer_char: str, operator: str,
                      glyph_data) -> float:
    """Estimate what fraction of the inner box is covered by the outer component's strokes.

    Returns 0.0 (no overlap) to 1.0 (fully covered).
    Uses the outer component's stroke bounding box vs. the inner inset region.
    """
    inset = _SURROUND_OPS.get(operator)
    if not inset:
        return 0.0

    ix, iy, iw, ih = inset  # inner box as fraction of outer box

    # Get outer component's strokes
    strokes = glyph_data.get_strokes(outer_char)
    if not strokes:
        return 0.0

    # Get bounding box of outer strokes (in source coordinates ~0-1024)
    from .renderer import _stroke_bbox, SOURCE_W, SOURCE_H
    get_bbox = getattr(glyph_data, 'get_bbox', None)
    if get_bbox:
        bbox = get_bbox(outer_char)
    else:
        bbox = _stroke_bbox(strokes)
    if not bbox:
        return 0.0

    # Normalize stroke bbox to 0-1 range
    bx1, by1, bx2, by2 = bbox
    source_w = getattr(glyph_data, 'source_w', SOURCE_W)
    source_h = getattr(glyph_data, 'source_h', SOURCE_H)
    norm_x1 = bx1 / source_w
    norm_y1 = by1 / source_h
    norm_x2 = bx2 / source_w
    norm_y2 = by2 / source_h

    # Inner box boundaries
    inner_x1, inner_y1 = ix, iy
    inner_x2, inner_y2 = ix + iw, iy + ih

    # Compute overlap of outer stroke bbox with inner box
    overlap_x1 = max(norm_x1, inner_x1)
    overlap_y1 = max(norm_y1, inner_y1)
    overlap_x2 = min(norm_x2, inner_x2)
    overlap_y2 = min(norm_y2, inner_y2)

    if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
        return 0.0

    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
    inner_area = iw * ih

    if inner_area <= 0:
        return 0.0

    return min(overlap_area / inner_area, 1.0)


def _check_overlap(tree, glyph_data) -> list[Issue]:
    """Check for overlap issues in surround and overlay operators."""
    issues = []

    def _walk(node):
        if node.is_leaf:
            return
        op = node.operator
        if op in _SURROUND_OPS and len(node.children) >= 2:
            outer = node.children[0]
            if outer.is_leaf and outer.character:
                char = outer.character
                known = _KNOWN_SURROUND_OUTERS.get(op, set())
                op_name = {
                    "\u2FF4": "⿴ full-surround", "\u2FF5": "⿵ surround-above",
                    "\u2FF6": "⿶ surround-below", "\u2FF7": "⿷ surround-left",
                    "\u2FF8": "⿸ upper-left", "\u2FF9": "⿹ upper-right",
                    "\u2FFA": "⿺ lower-left", "\u2FFB": "⿻ overlaid",
                    "\u2FFC": "⿼ surround-right", "\u2FFD": "⿽ below-right",
                }.get(op, op)

                if op == "\u2FFB":
                    # Overlaid is always full overlap by design
                    issues.append(Issue("info",
                        f"Overlay ({op_name}): {char} + "
                        f"{node.children[1].to_ids()} fully overlapping"))
                elif char not in known:
                    # Renderer uses a non-overlapping fallback layout for
                    # non-standard outers, so flag as info rather than
                    # computing overlap against the standard insets.
                    from .renderer import SURROUND_OUTERS
                    if op in SURROUND_OUTERS:
                        issues.append(Issue("info",
                            f"Non-standard {op_name} outer: {char} — "
                            f"using split layout"))
                    else:
                        # Operators not in SURROUND_OUTERS (⿻, ⿼, ⿽)
                        overlap = _estimate_overlap(char, op, glyph_data)
                        if overlap > 0.7:
                            issues.append(Issue("warning",
                                f"Heavy overlap ({overlap:.0%}): {char} "
                                f"covers inner component in {op_name}"))
                        elif overlap > 0.4:
                            issues.append(Issue("info",
                                f"Partial overlap ({overlap:.0%}): {char} "
                                f"in {op_name}"))

        for child in node.children:
            _walk(child)

    _walk(tree)
    return issues


def grade_character(ids_string: str, renderer: Renderer, glyph_data,
                    entry_id: str = "", title: str = "") -> GradeResult:
    """Grade a single character rendering."""
    issues: list[Issue] = []

    # 1. Parse IDS
    try:
        tree = parse_ids(ids_string)
    except IDSParseError as e:
        return GradeResult(
            entry_id=entry_id, ids=ids_string, title=title,
            svg="", grade="F",
            issues=[Issue("error", f"IDS parse error: {e}")],
        )

    leaves = collect_leaves(tree)
    depth = _ids_depth(ids_string)

    # 2. Check component availability
    missing = []
    for char in leaves:
        if not glyph_data.get_strokes(char):
            # Check if decomposition exists as fallback
            decomp = glyph_data.get_decomposition(char)
            if not decomp:
                missing.append(char)
    if missing:
        issues.append(Issue("error", f"Missing glyphs: {', '.join(missing)}"))

    # 3. Check variant availability
    from .renderer import LEFT_VARIANTS, BOTTOM_VARIANTS, RIGHT_VARIANTS
    if not tree.is_leaf and tree.operator:
        for i, child in enumerate(tree.children):
            if child.is_leaf and child.character:
                char = child.character
                variant = None
                if tree.operator == "\u2FF0" and i == 0:
                    variant = LEFT_VARIANTS.get(char)
                elif tree.operator == "\u2FF0" and i == 1:
                    variant = RIGHT_VARIANTS.get(char)
                elif tree.operator == "\u2FF1" and i == 1:
                    variant = BOTTOM_VARIANTS.get(char)
                if variant and not glyph_data.has_char(variant):
                    issues.append(Issue("warning",
                        f"Variant {variant} for {char} not in glyph data"))

    # 4. Check for overlap in surround operators
    issues.extend(_check_overlap(tree, glyph_data))

    # 5. Render
    try:
        svg = renderer.render_ids(ids_string, size=256)
    except Exception as e:
        return GradeResult(
            entry_id=entry_id, ids=ids_string, title=title,
            svg="", grade="F",
            issues=issues + [Issue("error", f"Render error: {e}")],
            leaves=leaves, depth=depth, num_components=len(leaves),
        )

    # 6. Check for placeholder text
    if '<text' in svg:
        placeholder_chars = re.findall(r'<text[^>]*>([^<]+)</text>', svg)
        issues.append(Issue("error",
            f"Placeholder fallback for: {', '.join(placeholder_chars)}"))

    # 7. Analyze transforms for quality issues
    scales = _analyze_transforms(svg)
    for sx, sy in scales:
        if sx > 0 and sy > 0:
            ratio = max(sx, sy) / min(sx, sy)
            if ratio > 2.5:
                issues.append(Issue("warning",
                    f"Extreme squish ratio {ratio:.1f}x "
                    f"(scale {sx:.3f} x {sy:.3f})"))
            elif ratio > 1.8:
                issues.append(Issue("info",
                    f"Notable squish ratio {ratio:.1f}x "
                    f"(scale {sx:.3f} x {sy:.3f})"))
            # Very small scales mean the component is tiny
            if min(sx, sy) < 0.05:
                issues.append(Issue("warning",
                    f"Very small scale {min(sx, sy):.3f} — "
                    f"component may be illegible"))
            elif min(sx, sy) < 0.08:
                issues.append(Issue("info",
                    f"Small scale {min(sx, sy):.3f} — "
                    f"component may be hard to read"))

    # 8. Analyze stroke weight compensation
    raw_widths, visual_widths = _analyze_stroke_weights(svg)
    if raw_widths:
        max_sw = max(raw_widths)
        if max_sw > 20:
            issues.append(Issue("warning",
                f"Heavy stroke compensation ({max_sw:.0f}px) — "
                f"strokes may look bloated"))
        elif max_sw > 10:
            issues.append(Issue("info",
                f"Moderate stroke compensation ({max_sw:.0f}px)"))
        # Compare visual stroke weights (stroke-width * scale), which is
        # what the viewer actually sees. Raw SVG values can differ 3x while
        # the visual thickness is nearly identical.
        if len(visual_widths) > 1:
            max_vw = max(visual_widths)
            min_vw = min(visual_widths)
            if max_vw > 0 and min_vw > 0:
                vw_ratio = max_vw / min_vw
                if vw_ratio > 2.5:
                    issues.append(Issue("warning",
                        f"Visual stroke weight disparity {vw_ratio:.1f}x "
                        f"between components"))
                elif vw_ratio > 1.8:
                    issues.append(Issue("info",
                        f"Visual stroke weight difference {vw_ratio:.1f}x"))

    # 9. Analyze component area balance
    # Use higher thresholds when surround operators are present, since
    # outer-fills-all / inner-inset layouts inherently have large ratios.
    def _has_surround_op(node) -> bool:
        if node.is_leaf:
            return False
        if node.operator in _SURROUND_OPS:
            return True
        return any(_has_surround_op(c) for c in node.children)
    has_surround = _has_surround_op(tree)
    area_warn = 10.0 if has_surround else 6.0
    area_info = 6.0 if has_surround else 3.5

    areas = _analyze_component_areas(svg)
    if len(areas) >= 2:
        max_area = max(areas)
        min_area = min(areas)
        if min_area > 0:
            area_ratio = max_area / min_area
            if area_ratio > area_warn:
                issues.append(Issue("warning",
                    f"Component size imbalance {area_ratio:.1f}x"))
            elif area_ratio > area_info:
                issues.append(Issue("info",
                    f"Component size difference {area_ratio:.1f}x"))

    # 10. Check nesting depth
    if depth > 3:
        issues.append(Issue("warning",
            f"Deep nesting (depth {depth}) — inner components may be small"))
    elif depth > 2:
        issues.append(Issue("info",
            f"Nesting depth {depth}"))

    # 11. Assign grade
    errors = sum(1 for i in issues if i.severity == "error")
    warnings = sum(1 for i in issues if i.severity == "warning")
    infos = sum(1 for i in issues if i.severity == "info")

    if errors > 0:
        grade = "F"
    elif warnings > 1:
        grade = "C"
    elif warnings == 1:
        grade = "B"
    elif infos > 3:
        grade = "B"
    else:
        grade = "A"

    return GradeResult(
        entry_id=entry_id, ids=ids_string, title=title,
        svg=svg, grade=grade, issues=issues,
        leaves=leaves, depth=depth, num_components=len(leaves),
    )


def grade_all(characters_dir: str, renderer: Renderer,
              glyph_data) -> list[GradeResult]:
    """Grade all dictionary entries."""
    results = []
    yaml_files = sorted(
        f for f in os.listdir(characters_dir) if f.endswith('.yaml')
    )

    for fname in yaml_files:
        path = os.path.join(characters_dir, fname)
        with open(path, encoding="utf-8") as f:
            entry = yaml.safe_load(f)
        ids = entry.get("ids", "")
        entry_id = entry.get("id", fname.replace(".yaml", ""))
        title = entry.get("title", entry_id)

        result = grade_character(ids, renderer, glyph_data,
                                 entry_id=entry_id, title=title)
        results.append(result)

    return results


def grade_ids_list(ids_list: list[str], renderer: Renderer,
                   glyph_data) -> list[GradeResult]:
    """Grade a list of IDS strings (not from dictionary)."""
    results = []
    for i, ids in enumerate(ids_list):
        result = grade_character(ids, renderer, glyph_data,
                                 entry_id=f"ids_{i:03d}", title=ids)
        results.append(result)
    return results


def _grade_color(grade: str) -> str:
    return {"A": "#2d8a4e", "B": "#b58900", "C": "#cb4b16", "F": "#dc322f"
            }.get(grade, "#666")


def _severity_color(severity: str) -> str:
    return {"error": "#dc322f", "warning": "#cb4b16", "info": "#268bd2"
            }.get(severity, "#666")


def generate_report(results: list[GradeResult], output_path: str) -> None:
    """Generate an HTML report with inline SVGs and grades."""
    # Summary counts
    counts = {"A": 0, "B": 0, "C": 0, "F": 0}
    for r in results:
        counts[r.grade] = counts.get(r.grade, 0) + 1

    summary_parts = []
    for g in ("A", "B", "C", "F"):
        if counts[g] > 0:
            summary_parts.append(
                f'<span style="color:{_grade_color(g)};font-weight:bold">'
                f'{counts[g]} {g}</span>')

    cards_html = []
    for r in results:
        # Strip outer <svg> wrapper to inline it, or use the full SVG
        svg_display = r.svg if r.svg else (
            '<svg viewBox="0 0 256 256" width="256" height="256">'
            '<text x="128" y="128" text-anchor="middle" fill="#999" '
            'font-size="24">ERROR</text></svg>')

        issues_html = ""
        if r.issues:
            issue_items = []
            for issue in r.issues:
                color = _severity_color(issue.severity)
                issue_items.append(
                    f'<div style="color:{color};font-size:13px;margin:2px 0">'
                    f'<b>{issue.severity}:</b> {issue.message}</div>')
            issues_html = "\n".join(issue_items)
        else:
            issues_html = '<div style="color:#2d8a4e;font-size:13px">No issues</div>'

        leaves_str = " ".join(r.leaves) if r.leaves else ""

        card = f'''<div class="card" style="border-color:{_grade_color(r.grade)}">
  <div class="grade" style="background:{_grade_color(r.grade)}">{r.grade}</div>
  <div class="svg-container">{svg_display}</div>
  <div class="info">
    <div class="title">{r.title or r.entry_id}</div>
    <div class="ids" title="{r.ids}">{r.ids}</div>
    <div class="meta">components: {r.num_components} &middot; depth: {r.depth}</div>
    <div class="leaves">{leaves_str}</div>
    <div class="issues">{issues_html}</div>
  </div>
</div>'''
        cards_html.append(card)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SurrealHanzi — Rendering Grade Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #fafafa; color: #333; padding: 24px;
  }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .summary {{ font-size: 15px; margin-bottom: 20px; color: #666; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 16px;
  }}
  .card {{
    background: #fff; border-radius: 8px; border: 2px solid #ddd;
    overflow: hidden; position: relative;
  }}
  .grade {{
    position: absolute; top: 8px; right: 8px; z-index: 1;
    color: #fff; font-weight: bold; font-size: 18px;
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
  }}
  .svg-container {{
    display: flex; align-items: center; justify-content: center;
    padding: 16px; background: #fff; min-height: 200px;
  }}
  .svg-container svg {{ max-width: 180px; max-height: 180px; }}
  .info {{ padding: 12px 16px 16px; border-top: 1px solid #eee; }}
  .title {{ font-weight: 600; font-size: 15px; }}
  .ids {{ font-family: "Noto Serif TC", serif; font-size: 18px; margin: 2px 0; color: #555; }}
  .meta {{ font-size: 12px; color: #999; margin: 4px 0; }}
  .leaves {{ font-size: 16px; margin: 4px 0; letter-spacing: 4px; }}
  .issues {{ margin-top: 8px; }}

  @media (prefers-color-scheme: dark) {{
    body {{ background: #1a1a1a; color: #ccc; }}
    .card {{ background: #252525; border-color: #444; }}
    .svg-container {{ background: #252525; }}
    .svg-container svg {{ color: #e0e0e0; }}
    .info {{ border-top-color: #444; }}
    .ids {{ color: #aaa; }}
    .meta {{ color: #777; }}
  }}
</style>
</head>
<body>
<h1>SurrealHanzi Rendering Grade Report</h1>
<div class="summary">{len(results)} characters: {" &middot; ".join(summary_parts)}</div>
<div class="grid">
{"".join(cards_html)}
</div>
</body>
</html>'''

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def print_summary(results: list[GradeResult]) -> None:
    """Print a terminal summary of grading results."""
    grade_symbols = {"A": "\033[32m", "B": "\033[33m",
                     "C": "\033[31m", "F": "\033[91m"}
    reset = "\033[0m"

    for r in results:
        color = grade_symbols.get(r.grade, "")
        prefix = f"  {color}{r.grade}{reset}"
        label = r.title or r.entry_id
        print(f"{prefix}  {r.ids:20s}  {label}")
        for issue in r.issues:
            sev_color = {"error": "\033[91m", "warning": "\033[33m",
                         "info": "\033[36m"}.get(issue.severity, "")
            print(f"       {sev_color}{issue.severity}: {issue.message}{reset}")

    # Summary line
    counts = {}
    for r in results:
        counts[r.grade] = counts.get(r.grade, 0) + 1
    parts = []
    for g in ("A", "B", "C", "F"):
        if counts.get(g, 0) > 0:
            color = grade_symbols.get(g, "")
            parts.append(f"{color}{counts[g]} {g}{reset}")
    print(f"\n  {len(results)} characters: {', '.join(parts)}")

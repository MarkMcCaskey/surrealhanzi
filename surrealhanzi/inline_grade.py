"""Inline rendering grade system.

Uses Playwright to render characters inline with Chinese text in a browser,
then measures size, alignment, and visual density against reference characters.
"""

import base64
import io
import os
import re
from dataclasses import dataclass, field

import yaml

from .grade import GradeResult, Issue, _grade_color, _severity_color
from .renderer import Renderer


# --- HTML template ---
# Replicates the site's exact inline rendering context.

_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+TC:wght@400;700&display=swap');
body {{
  font-family: "Noto Serif TC", "Source Han Serif TC", "Songti TC", serif;
  font-size: 1.05rem;
  line-height: 1.8;
  color: #000;
  background: #fff;
  padding: 2rem;
  max-width: 640px;
}}
.inline-char {{
  display: inline-block;
  width: 1em;
  height: 1em;
  vertical-align: -0.12em;
  color: currentColor;
}}
.test-line {{
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid #eee;
}}
.label {{
  font-size: 0.7rem;
  color: #999;
  font-family: sans-serif;
}}
</style>
</head>
<body>
{body}
</body>
</html>"""

# Standardized test sentences.  {char} is replaced with the inline SVG.
# Reference characters on each side are wrapped with data-ref for measurement.
_TEST_SENTENCES = [
    (
        '<span data-ref>試</span><span data-ref>文</span>'
        '{char}'
        '<span data-ref>測</span><span data-ref>試</span>文字。',
        "mid-sentence",
    ),
    (
        '<span data-ref>國</span><span data-ref>際</span>'
        '{char}'
        '<span data-ref>體</span><span data-ref>驗</span>當中。',
        "dense context",
    ),
    (
        '<span data-ref>人</span><span data-ref>大</span>'
        '{char}'
        '<span data-ref>上</span><span data-ref>下</span>方。',
        "sparse context",
    ),
]

# JS to extract bounding boxes of the inline SVG and reference characters
# within a single .test-line element.
_MEASURE_JS = """(lineIndex) => {
    const lines = document.querySelectorAll('.test-line');
    const line = lines[lineIndex];
    if (!line) return null;
    const svg = line.querySelector('.inline-char');
    const refs = line.querySelectorAll('[data-ref]');
    if (!svg || refs.length === 0) return null;
    const lineR = line.getBoundingClientRect();
    const svgR = svg.getBoundingClientRect();
    // Return coordinates relative to the test-line element (for screenshot cropping)
    function rel(r) {
        return {top: r.top - lineR.top, bottom: r.bottom - lineR.top,
                left: r.left - lineR.left, right: r.right - lineR.left,
                width: r.width, height: r.height};
    }
    const refRects = Array.from(refs).map(r => rel(r.getBoundingClientRect()));
    return {
        svg: rel(svgR),
        refs: refRects,
    };
}"""


# --- Data structures ---

@dataclass
class InlineMetrics:
    """Measured metrics for inline rendering (averaged across test sentences)."""
    height_ratio: float       # SVG height / ref char height
    width_ratio: float        # SVG width / ref char width
    baseline_offset_em: float # vertical offset (bottom edge) in em units
    density_ratio: float      # ink density vs ref chars
    stroke_weight_ratio: float  # approx stroke width vs ref chars


@dataclass
class InlineCapture:
    """Screenshot data and metrics for one character."""
    entry_id: str
    ids: str
    title: str
    svg: str
    metrics: InlineMetrics
    test_screenshots: list[bytes] = field(default_factory=list)
    example_screenshots: list[bytes] = field(default_factory=list)


# --- SVG preparation ---

def _make_inline_svg(svg: str) -> str:
    """Prepare an SVG for inline use (mirrors makeInlineSvg() from site JS)."""
    svg = re.sub(r'\s+width="[^"]*"', '', svg)
    svg = re.sub(r'\s+height="[^"]*"', '', svg)
    svg = re.sub(r'\s+style="[^"]*"', '', svg)
    svg = svg.replace('<svg ', '<svg class="inline-char" ')
    return svg


# --- Pixel analysis ---

def _crop_bbox(img, bbox: dict):
    """Crop a PIL Image to a bounding box dict."""
    left = max(0, int(bbox['left']))
    top = max(0, int(bbox['top']))
    right = min(img.width, int(bbox['right']))
    bottom = min(img.height, int(bbox['bottom']))
    if right <= left or bottom <= top:
        return None
    return img.crop((left, top, right, bottom))


def _ink_density(img_bytes: bytes, bbox: dict) -> float:
    """Compute ink density (fraction of dark pixels) in a bounding box region."""
    from PIL import Image
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    crop = _crop_bbox(img, bbox)
    if crop is None:
        return 0.0
    pixels = list(crop.getdata())
    if not pixels:
        return 0.0
    dark = sum(1 for p in pixels if p < 128)
    return dark / len(pixels)


def _ink_bbox(img_bytes: bytes, bbox: dict) -> tuple[int, int] | None:
    """Compute the tight bounding box of ink (dark pixels) within a region.

    Returns (ink_width, ink_height) of the actual visible content, excluding
    whitespace padding. This is more accurate than CSS bounding boxes for
    comparing SVG characters to font glyphs, since font metrics include
    ascent/descent that may be empty.
    """
    from PIL import Image
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    crop = _crop_bbox(img, bbox)
    if crop is None:
        return None
    w, h = crop.size
    threshold = 200  # generous: anything darker than near-white counts
    # Find vertical extent of ink
    top_ink = h
    bottom_ink = 0
    left_ink = w
    right_ink = 0
    for y in range(h):
        for x in range(w):
            if crop.getpixel((x, y)) < threshold:
                top_ink = min(top_ink, y)
                bottom_ink = max(bottom_ink, y)
                left_ink = min(left_ink, x)
                right_ink = max(right_ink, x)
    if bottom_ink <= top_ink or right_ink <= left_ink:
        return None
    return (right_ink - left_ink + 1, bottom_ink - top_ink + 1)


def _avg_stroke_width(img_bytes: bytes, bbox: dict) -> float:
    """Estimate average stroke width via horizontal dark run-lengths."""
    from PIL import Image
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    left = max(0, int(bbox['left']))
    top = max(0, int(bbox['top']))
    right = min(img.width, int(bbox['right']))
    bottom = min(img.height, int(bbox['bottom']))
    if right <= left or bottom <= top:
        return 0.0
    crop = img.crop((left, top, right, bottom))
    w, h = crop.size
    if w == 0 or h == 0:
        return 0.0

    run_lengths = []
    for y in range(h):
        in_run = False
        run_len = 0
        for x in range(w):
            pixel = crop.getpixel((x, y))
            if pixel < 128:  # dark
                in_run = True
                run_len += 1
            else:
                if in_run and run_len > 0:
                    run_lengths.append(run_len)
                in_run = False
                run_len = 0
        if in_run and run_len > 0:
            run_lengths.append(run_len)
    return sum(run_lengths) / len(run_lengths) if run_lengths else 0.0


# --- Metric checks ---

def _check_size(m: InlineMetrics) -> list[Issue]:
    issues = []
    for name, ratio in [("height", m.height_ratio), ("width", m.width_ratio)]:
        if ratio < 0.80 or ratio > 1.20:
            word = "small" if ratio < 1 else "large"
            issues.append(Issue("warning",
                f"Character {name} {ratio:.2f}x vs reference — appears too {word}"))
        elif ratio < 0.90 or ratio > 1.10:
            word = "small" if ratio < 1 else "large"
            issues.append(Issue("info",
                f"Character {name} {ratio:.2f}x vs reference — slightly {word}"))
    return issues


def _check_alignment(m: InlineMetrics) -> list[Issue]:
    issues = []
    off = m.baseline_offset_em
    if abs(off) > 0.15:
        direction = "low" if off > 0 else "high"
        issues.append(Issue("warning",
            f"Baseline offset {off:+.2f}em — character sits too {direction}"))
    elif abs(off) > 0.08:
        direction = "low" if off > 0 else "high"
        issues.append(Issue("info",
            f"Baseline offset {off:+.2f}em — slightly {direction}"))
    return issues


def _check_density(m: InlineMetrics) -> list[Issue]:
    issues = []
    r = m.density_ratio
    if r > 0 and (r < 0.5 or r > 2.0):
        word = "sparse" if r < 1 else "dense"
        issues.append(Issue("warning",
            f"Ink density {r:.2f}x vs reference — too {word}"))
    elif r > 0 and (r < 0.7 or r > 1.5):
        word = "sparse" if r < 1 else "dense"
        issues.append(Issue("info",
            f"Ink density {r:.2f}x vs reference — slightly {word}"))
    return issues


def _check_stroke_weight(m: InlineMetrics) -> list[Issue]:
    issues = []
    r = m.stroke_weight_ratio
    if r > 0 and (r < 0.5 or r > 2.0):
        word = "thin" if r < 1 else "thick"
        issues.append(Issue("warning",
            f"Stroke weight {r:.2f}x vs reference — too {word}"))
    elif r > 0 and (r < 0.7 or r > 1.5):
        word = "thin" if r < 1 else "thick"
        issues.append(Issue("info",
            f"Stroke weight {r:.2f}x vs reference — slightly {word}"))
    return issues


# --- Capture ---

def _build_test_html(inline_svg: str, examples: list[dict] | None,
                     ids: str) -> str:
    """Build the full HTML page for one character."""
    blocks = []
    for sentence_tpl, label in _TEST_SENTENCES:
        line_html = sentence_tpl.replace('{char}', inline_svg)
        blocks.append(
            f'<div class="test-line">'
            f'<div class="label">{label}</div>'
            f'{line_html}</div>'
        )

    # Real examples (for visual review, no data-ref markers)
    if examples:
        for ex in examples[:3]:
            text = ex.get('text', '')
            if ids in text:
                text_html = text.replace('&', '&amp;').replace('<', '&lt;')
                text_html = text_html.replace(ids, inline_svg)
                blocks.append(
                    f'<div class="test-line">'
                    f'<div class="label">example</div>'
                    f'{text_html}</div>'
                )

    return _TEMPLATE.format(body='\n'.join(blocks))


def _measure_line(page, line_index: int) -> dict | None:
    """Extract bounding boxes for one test line via JS."""
    return page.evaluate(_MEASURE_JS, line_index)


def _compute_metrics(measurements: list[dict],
                     screenshots: list[bytes]) -> InlineMetrics:
    """Compute averaged metrics from multiple test line measurements."""
    height_ratios = []
    width_ratios = []
    baseline_offsets = []
    density_ratios = []
    stroke_weight_ratios = []

    for i, m in enumerate(measurements):
        if m is None:
            continue
        svg_r = m['svg']
        ref_rs = m['refs']
        if not ref_rs or i >= len(screenshots) or not screenshots[i]:
            continue

        ss = screenshots[i]

        # Size comparison: use ink bounding boxes (actual visible content)
        # instead of CSS rects, since font metrics include empty ascent/descent
        svg_ink = _ink_bbox(ss, svg_r)
        ref_inks = [_ink_bbox(ss, r) for r in ref_rs]
        ref_inks = [r for r in ref_inks if r is not None]

        if svg_ink and ref_inks:
            avg_ref_ink_h = sum(r[1] for r in ref_inks) / len(ref_inks)
            avg_ref_ink_w = sum(r[0] for r in ref_inks) / len(ref_inks)
            if avg_ref_ink_h > 0:
                height_ratios.append(svg_ink[1] / avg_ref_ink_h)
            if avg_ref_ink_w > 0:
                width_ratios.append(svg_ink[0] / avg_ref_ink_w)

        # Baseline: compare bottom edges of CSS rects (still useful)
        avg_ref_bottom = sum(r['bottom'] for r in ref_rs) / len(ref_rs)
        if svg_r['height'] > 0:
            offset = (svg_r['bottom'] - avg_ref_bottom) / svg_r['height']
            baseline_offsets.append(offset)

        # Density and stroke weight from pixel analysis
        svg_density = _ink_density(ss, svg_r)
        ref_densities = [_ink_density(ss, r) for r in ref_rs]
        avg_ref_density = (sum(ref_densities) / len(ref_densities)
                           if ref_densities else 0)
        if avg_ref_density > 0:
            density_ratios.append(svg_density / avg_ref_density)

        svg_sw = _avg_stroke_width(ss, svg_r)
        ref_sws = [_avg_stroke_width(ss, r) for r in ref_rs]
        avg_ref_sw = sum(ref_sws) / len(ref_sws) if ref_sws else 0
        if avg_ref_sw > 0:
            stroke_weight_ratios.append(svg_sw / avg_ref_sw)

    def _avg(lst): return sum(lst) / len(lst) if lst else 0.0

    return InlineMetrics(
        height_ratio=_avg(height_ratios),
        width_ratio=_avg(width_ratios),
        baseline_offset_em=_avg(baseline_offsets),
        density_ratio=_avg(density_ratios),
        stroke_weight_ratio=_avg(stroke_weight_ratios),
    )


def _capture_one(page, entry: dict, renderer: Renderer) -> InlineCapture:
    """Capture and measure one character."""
    ids = entry.get('ids', '')
    entry_id = entry.get('id', '')
    title = entry.get('title', entry_id)
    examples = entry.get('examples')

    # Render SVG and prepare for inline use
    svg = renderer.render_ids(ids, size=256)
    inline_svg = _make_inline_svg(svg)

    # Build HTML and load page
    html = _build_test_html(inline_svg, examples, ids)
    page.set_content(html)

    # Wait for fonts (5s timeout)
    try:
        page.wait_for_function("document.fonts.ready", timeout=5000)
    except Exception:
        pass  # continue with fallback font

    # Measure each standardized test line and screenshot it
    num_test = len(_TEST_SENTENCES)
    measurements = []
    test_screenshots = []
    for i in range(num_test):
        m = _measure_line(page, i)
        measurements.append(m)
        # Screenshot the test line element
        lines = page.query_selector_all('.test-line')
        if i < len(lines):
            test_screenshots.append(lines[i].screenshot())
        else:
            test_screenshots.append(b'')

    # Screenshot example lines
    example_screenshots = []
    lines = page.query_selector_all('.test-line')
    for i in range(num_test, len(lines)):
        example_screenshots.append(lines[i].screenshot())

    metrics = _compute_metrics(measurements, test_screenshots)

    return InlineCapture(
        entry_id=entry_id, ids=ids, title=title, svg=svg,
        metrics=metrics,
        test_screenshots=test_screenshots,
        example_screenshots=example_screenshots,
    )


def _find_chrome() -> str | None:
    """Find a system Chrome/Chromium executable."""
    import platform
    candidates = []
    if platform.system() == "Darwin":
        candidates = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
    else:
        candidates = ["google-chrome", "chromium", "chromium-browser"]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _capture_all(entries: list[dict], renderer: Renderer) -> list[InlineCapture]:
    """Capture all characters using a single browser session."""
    from playwright.sync_api import sync_playwright

    chrome_path = _find_chrome()

    captures = []
    with sync_playwright() as p:
        if chrome_path:
            browser = p.chromium.launch(executable_path=chrome_path,
                                        headless=True)
        else:
            browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 640, 'height': 800})
        for entry in entries:
            try:
                capture = _capture_one(page, entry, renderer)
                captures.append(capture)
                print(f"  {capture.ids:20s}  {capture.title}")
            except Exception as e:
                entry_id = entry.get('id', '?')
                print(f"  {entry_id}: error — {e}")
                captures.append(InlineCapture(
                    entry_id=entry_id,
                    ids=entry.get('ids', ''),
                    title=entry.get('title', entry_id),
                    svg='',
                    metrics=InlineMetrics(0, 0, 0, 0, 0),
                ))
        browser.close()
    return captures


# --- Grading ---

def grade_inline(capture: InlineCapture) -> GradeResult:
    """Convert inline capture metrics into a grade result."""
    issues: list[Issue] = []
    m = capture.metrics

    # Check for capture failure
    if m.height_ratio == 0 and m.width_ratio == 0:
        issues.append(Issue("error", "Inline capture failed"))

    issues.extend(_check_size(m))
    issues.extend(_check_alignment(m))
    issues.extend(_check_density(m))
    issues.extend(_check_stroke_weight(m))

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
        entry_id=capture.entry_id, ids=capture.ids, title=capture.title,
        svg=capture.svg, grade=grade, issues=issues,
    )


# --- Public API ---

def grade_all_inline(characters_dir: str, renderer: Renderer,
                     glyph_data) -> list[GradeResult]:
    """Grade all dictionary entries for inline rendering quality."""
    entries = []
    for fname in sorted(os.listdir(characters_dir)):
        if not fname.endswith('.yaml'):
            continue
        with open(os.path.join(characters_dir, fname), encoding='utf-8') as f:
            entries.append(yaml.safe_load(f))

    captures = _capture_all(entries, renderer)
    return [grade_inline(c) for c in captures], captures


def grade_ids_inline(ids_list: list[str], renderer: Renderer,
                     glyph_data) -> list[GradeResult]:
    """Grade specific IDS strings for inline rendering quality."""
    entries = [{'id': f'ids_{i}', 'ids': ids, 'title': ids}
               for i, ids in enumerate(ids_list)]
    captures = _capture_all(entries, renderer)
    return [grade_inline(c) for c in captures], captures


# --- Report ---

def _b64_png(data: bytes) -> str:
    """Encode PNG bytes as a data URL."""
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


def generate_inline_report(results: list[GradeResult],
                           captures: list[InlineCapture],
                           output_path: str) -> None:
    """Generate HTML report with inline rendering screenshots."""
    counts = {"A": 0, "B": 0, "C": 0, "F": 0}
    for r in results:
        counts[r.grade] = counts.get(r.grade, 0) + 1

    summary_parts = []
    for g in ("A", "B", "C", "F"):
        if counts[g] > 0:
            summary_parts.append(
                f'<span style="color:{_grade_color(g)};font-weight:bold">'
                f'{counts[g]} {g}</span>')

    cards = []
    for r, c in zip(results, captures):
        # Screenshots
        screenshots_html = ""
        for i, ss in enumerate(c.test_screenshots):
            if ss:
                label = _TEST_SENTENCES[i][1] if i < len(_TEST_SENTENCES) else "test"
                screenshots_html += (
                    f'<div style="margin:4px 0">'
                    f'<div style="font-size:11px;color:#999">{label}</div>'
                    f'<img src="{_b64_png(ss)}" style="max-width:100%;border:1px solid #eee;border-radius:4px">'
                    f'</div>')
        for ss in c.example_screenshots:
            if ss:
                screenshots_html += (
                    f'<div style="margin:4px 0">'
                    f'<div style="font-size:11px;color:#999">example</div>'
                    f'<img src="{_b64_png(ss)}" style="max-width:100%;border:1px solid #eee;border-radius:4px">'
                    f'</div>')

        # Metrics table
        m = c.metrics
        metrics_html = (
            f'<table style="font-size:12px;margin:8px 0;border-collapse:collapse">'
            f'<tr><td style="padding:2px 8px;color:#666">height</td>'
            f'<td style="padding:2px 8px">{m.height_ratio:.2f}x</td></tr>'
            f'<tr><td style="padding:2px 8px;color:#666">width</td>'
            f'<td style="padding:2px 8px">{m.width_ratio:.2f}x</td></tr>'
            f'<tr><td style="padding:2px 8px;color:#666">baseline</td>'
            f'<td style="padding:2px 8px">{m.baseline_offset_em:+.2f}em</td></tr>'
            f'<tr><td style="padding:2px 8px;color:#666">density</td>'
            f'<td style="padding:2px 8px">{m.density_ratio:.2f}x</td></tr>'
            f'<tr><td style="padding:2px 8px;color:#666">stroke wt</td>'
            f'<td style="padding:2px 8px">{m.stroke_weight_ratio:.2f}x</td></tr>'
            f'</table>'
        )

        # Issues
        if r.issues:
            issues_html = "\n".join(
                f'<div style="color:{_severity_color(i.severity)};font-size:13px;margin:2px 0">'
                f'<b>{i.severity}:</b> {i.message}</div>'
                for i in r.issues)
        else:
            issues_html = '<div style="color:#2d8a4e;font-size:13px">No issues</div>'

        # SVG preview
        svg_preview = c.svg.replace(' style="color: #000"', '') if c.svg else ''

        cards.append(f'''<div class="card" style="border-color:{_grade_color(r.grade)}">
  <div class="grade" style="background:{_grade_color(r.grade)}">{r.grade}</div>
  <div class="header">
    <div class="svg-small">{svg_preview}</div>
    <div>
      <div class="title">{r.title or r.entry_id}</div>
      <div class="ids">{r.ids}</div>
    </div>
  </div>
  <div class="body">
    {metrics_html}
    <div class="issues">{issues_html}</div>
    <div class="screenshots">{screenshots_html}</div>
  </div>
</div>''')

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SurrealHanzi — Inline Rendering Grade Report</title>
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
    grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
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
  .header {{
    display: flex; align-items: center; gap: 12px;
    padding: 12px 16px; border-bottom: 1px solid #eee;
  }}
  .svg-small {{ width: 48px; height: 48px; }}
  .svg-small svg {{ width: 48px; height: 48px; }}
  .title {{ font-weight: 600; font-size: 15px; }}
  .ids {{ font-family: "Noto Serif TC", serif; font-size: 16px; color: #555; }}
  .body {{ padding: 12px 16px; }}
  .screenshots {{ margin-top: 8px; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #1a1a1a; color: #ccc; }}
    .card {{ background: #252525; border-color: #444; }}
    .header {{ border-bottom-color: #444; }}
    .ids {{ color: #aaa; }}
    .svg-small svg {{ color: #e0e0e0; }}
  }}
</style>
</head>
<body>
<h1>SurrealHanzi — Inline Rendering Grade Report</h1>
<div class="summary">{len(results)} characters: {" &middot; ".join(summary_parts)}</div>
<div class="grid">
{"".join(cards)}
</div>
</body>
</html>'''

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

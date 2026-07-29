from __future__ import annotations

from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "figures" / "supplementary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PNG = OUT_DIR / "Supplementary_Figure_S1_cohort_flow.png"
PDF = OUT_DIR / "Supplementary_Figure_S1_cohort_flow.pdf"

W, H = 3000, 1760
SCALE = 1

RED = "#C25450"
BLUE = "#4A6FA5"
DARK = "#222222"
MID = "#555555"
LIGHT = "#D7DDE6"
FILL = "#FFFFFF"
PALE_RED = "#F7E8E6"
PALE_BLUE = "#EAF0F8"

FONT_DIR = Path("/System/Library/Fonts/Supplemental")
REG = str(FONT_DIR / "Times New Roman.ttf")
BOLD = str(FONT_DIR / "Times New Roman Bold.ttf")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD if bold else REG, size)


F_LABEL = font(68, True)
F_PANEL = font(46, True)
F_TITLE = font(40, True)
F_TEXT = font(28, False)
F_N = font(40, True)
F_SMALL = font(29, False)
F_TINY = font(20, False)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont):
    b = draw.textbbox((0, 0), text, font=fnt)
    return b[2] - b[0], b[3] - b[1]


def centered_text(draw, xy, text, fnt, fill=DARK, anchor="mm"):
    draw.text(xy, text, font=fnt, fill=fill, anchor=anchor)


def wrap_lines(text: str, chars: int) -> list[str]:
    return textwrap.wrap(text, width=chars, break_long_words=False, break_on_hyphens=False)


def draw_node(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    n: str,
    groups: str,
    accent: str,
    note: str | None = None,
    fill: str = FILL,
):
    x0, y0, x1, y1 = box
    r = 16
    draw.rounded_rectangle(box, radius=r, fill=fill, outline="#2E2E2E", width=3)

    cy = y0 + 52
    lines = wrap_lines(title, 20)
    for line in lines:
        centered_text(draw, ((x0 + x1) / 2, cy), line, F_TITLE, DARK)
        cy += 44
    cy += 4
    centered_text(draw, ((x0 + x1) / 2, cy), n, F_N, DARK)
    cy += 42
    centered_text(draw, ((x0 + x1) / 2, cy), groups, F_SMALL, MID)
    if note:
        cy += 30
        centered_text(draw, ((x0 + x1) / 2, cy), note, F_TINY, MID)


def arrow(draw, start, end, color="#444444", width=3):
    x0, y0 = start
    x1, y1 = end
    draw.line((x0, y0, x1, y1), fill=color, width=width)
    # arrow head
    import math

    ang = math.atan2(y1 - y0, x1 - x0)
    length = 18
    spread = 0.55
    p1 = (x1 - length * math.cos(ang - spread), y1 - length * math.sin(ang - spread))
    p2 = (x1 - length * math.cos(ang + spread), y1 - length * math.sin(ang + spread))
    draw.polygon((end, p1, p2), fill=color)


def elbow_arrow(draw, start, end, mid_y=None, color="#444444"):
    x0, y0 = start
    x1, y1 = end
    if mid_y is None:
        mid_y = (y0 + y1) // 2
    draw.line((x0, y0, x0, mid_y, x1, mid_y, x1, y1), fill=color, width=3)
    arrow(draw, (x1, y1 - 2), (x1, y1), color=color, width=3)


def draw_section_label(draw, x, y, label, color):
    draw.text((x, y), label, font=F_PANEL, fill=color, anchor="la")


img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)

# Panel labels and headings
draw.text((90, 70), "A", font=F_LABEL, fill=DARK)
draw_section_label(draw, 215, 88, "PRIMARY COHORT", RED)

draw.text((1725, 70), "B", font=F_LABEL, fill=DARK)
draw_section_label(draw, 1845, 88, "EXTERNAL HBN COHORT", BLUE)

# Subtle panel divider
draw.line((1610, 160, 1610, 1620), fill="#E8E8E8", width=3)

# Primary flow
top1 = (522, 210, 1277, 420)
mid1 = (522, 535, 1277, 745)
draw_node(draw, top1, "Resting EEG registration", "N = 168", "ASD = 80 | TD = 88", RED, fill=PALE_RED)
draw_node(draw, mid1, "Primary resting spectral cohort", "N = 138", "ASD = 61 | TD = 77", RED)
arrow(draw, ((top1[0] + top1[2]) // 2, top1[3] + 10), ((mid1[0] + mid1[2]) // 2, mid1[1] - 10))

node_w, node_h = 410, 210
gap_x, gap_y = 45, 64
xA = 240
xB = xA + node_w + gap_x
xC = xB + node_w + gap_x
yR1 = 930
yR2 = yR1 + node_h + gap_y
primary_nodes = [
    ((xA, yR1, xA + node_w, yR1 + node_h), "Paired rest-to-movie exponent", "N = 136", "ASD = 61 | TD = 75"),
    ((xB, yR1, xB + node_w, yR1 + node_h), "Movie Aperiodic-ISC cohort", "N = 136", "ASD = 58 | TD = 78"),
    ((xC, yR1, xC + node_w, yR1 + node_h), "Resting + movie matched", "N = 92", "ASD = 46 | TD = 46"),
    ((xA + 220, yR2, xA + 220 + node_w, yR2 + node_h), "IQ-balanced subset", "N = 76", "ASD = 38 | TD = 38"),
    ((xB + 220, yR2, xB + 220 + node_w, yR2 + node_h), "Strict specparam-QC", "N = 90", "ASD = 44 | TD = 46"),
]

bus_y = 850
center_mid = ((mid1[0] + mid1[2]) // 2, mid1[3] + 6)
draw.line((center_mid[0], center_mid[1], center_mid[0], bus_y), fill="#555555", width=3)
draw.line((xA + node_w / 2, bus_y, xC + node_w / 2, bus_y), fill="#555555", width=3)
for box, title, n, groups in primary_nodes:
    draw_node(draw, box, title, n, groups, RED)
    cx = (box[0] + box[2]) // 2
    arrow(draw, (cx, bus_y), (cx, box[1] - 12), color="#555555", width=3)

# HBN flow
hbn_top = (1875, 215, 2705, 425)
draw_node(draw, hbn_top, "HBN The Present matched cohort", "N = 238", "ASD = 119 | TD = 119", BLUE, fill=PALE_BLUE)

hnode_w, hnode_h = 610, 210
hy1 = 625
hy2 = hy1 + hnode_h + 75
hy3 = hy2 + hnode_h + 70
hbn_nodes = [
    ((1985, hy1, 1985 + hnode_w, hy1 + hnode_h), "The Present movie Aperiodic-ISC", "N = 238", "ASD = 119 | TD = 119"),
    ((1985, hy2, 1985 + hnode_w, hy2 + hnode_h), "Eyes-open resting subset", "N = 224", "ASD = 112 | TD = 112"),
    ((1985, hy3, 1985 + hnode_w, hy3 + hnode_h), "Eyes-closed resting subset", "N = 230", "ASD = 115 | TD = 115"),
]

h_bus_x = 1835
draw.line(((hbn_top[0] + hbn_top[2]) // 2, hbn_top[3] + 8, (hbn_top[0] + hbn_top[2]) // 2, 575), fill="#555555", width=3)
draw.line(((hbn_top[0] + hbn_top[2]) // 2, 575, h_bus_x, 575), fill="#555555", width=3)
for box, title, n, groups in hbn_nodes:
    cy = (box[1] + box[3]) // 2
    draw.line((h_bus_x, 575, h_bus_x, cy), fill="#555555", width=3)
    arrow(draw, (h_bus_x, cy), (box[0] - 12, cy), color="#555555", width=3)
    draw_node(draw, box, title, n, groups, BLUE)

# Crop away unused white space so the figure remains legible when inserted into Word.
img = img.crop((65, 50, 2775, 1505))
img.save(PNG, dpi=(300, 300))
img.save(PDF, "PDF", resolution=300.0)
print(PNG)
print(PDF)

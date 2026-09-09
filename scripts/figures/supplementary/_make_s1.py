"""Supplementary Figure S1 — cohort flow (current design)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from _si_style import ASD, OUT, TD

W, H = 1800, 1050
BG = "white"
TEXT = "#222222"
SUBTEXT = "#333333"
STROKE = "#1A1A1A"
ARROW = "#5A5A5A"
PALE_ASD = "#F3D4D4"
PALE_TD = "#D6D6EB"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_PANEL = font(48, True)
F_TITLE = font(34, True)
F_NODE = font(25, True)
F_N = font(24, True)
F_GROUP = font(21, True)


def center_multiline(draw, box, title_lines, n, groups):
    x1, y1, x2, y2 = box
    lines = [(line, F_NODE, TEXT) for line in title_lines]
    lines.append((n, F_N, TEXT))
    lines.append((groups, F_GROUP, SUBTEXT))
    heights = []
    for line, fnt, _ in lines:
        bbox = draw.textbbox((0, 0), line, font=fnt)
        heights.append(bbox[3] - bbox[1])
    spacing = 5
    total_h = sum(heights) + spacing * (len(lines) - 1)
    y = (y1 + y2 - total_h) / 2 - 1
    for (line, fnt, fill), h in zip(lines, heights):
        bbox = draw.textbbox((0, 0), line, font=fnt)
        draw.text(((x1 + x2 - (bbox[2] - bbox[0])) / 2, y - bbox[1]), line, font=fnt, fill=fill)
        y += h + spacing


def node(draw, box, title_lines, n, groups, fill="white"):
    draw.rounded_rectangle(box, radius=9, fill=fill, outline=STROKE, width=3)
    center_multiline(draw, box, title_lines, n, groups)


def arrow_head(draw, x, y, direction):
    if direction == "down":
        pts = [(x, y), (x - 8, y - 16), (x + 8, y - 16)]
    else:
        pts = [(x, y), (x - 14, y - 7), (x - 14, y + 7)]
    draw.polygon(pts, fill=ARROW)


def down_arrow(draw, x, y1, y2):
    draw.line((x, y1, x, y2), fill=ARROW, width=3)
    arrow_head(draw, x, y2, "down")


def branch_down(draw, start, targets):
    sx, sy = start
    joint_y = sy + 46
    min_x = min(x for x, _ in targets)
    max_x = max(x for x, _ in targets)
    draw.line((sx, sy, sx, joint_y), fill=ARROW, width=3)
    draw.line((min_x, joint_y, max_x, joint_y), fill=ARROW, width=3)
    for tx, ty in targets:
        draw.line((tx, joint_y, tx, ty), fill=ARROW, width=3)
        arrow_head(draw, tx, ty, "down")


def make_s1():
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    draw.text((40, 24), "a", font=F_PANEL, fill=TEXT)
    draw.text((118, 32), "PRIMARY COHORT", font=F_TITLE, fill=ASD)

    top = (540, 90, 1260, 195)
    primary = (540, 275, 1260, 380)
    paired = (80, 505, 520, 625)
    movie = (640, 505, 1160, 625)
    same = (1280, 505, 1720, 625)
    iq = (700, 700, 1100, 820)
    strict = (1320, 700, 1680, 820)

    node(draw, top, ["Resting EEG registration"], "N = 168", "(ASD=80, TD=88)", PALE_ASD)
    node(draw, primary, ["Primary resting spectral cohort"], "N = 138", "(ASD=61, TD=77)")
    down_arrow(draw, 900, 195, 268)
    branch_down(draw, (900, 380), [(300, 505), (900, 505), (1500, 505)])
    node(draw, paired, ["Paired rest-to-Partly Cloudy"], "N = 136", "(ASD=61, TD=75)")
    node(draw, movie, ["Partly Cloudy Aperiodic-ISC"], "N = 136", "(ASD=58, TD=78)")
    node(draw, same, ["Same-subject ISC ∩ state"], "N = 104", "(ASD=45, TD=59)")
    down_arrow(draw, 900, 625, 700)
    down_arrow(draw, 1500, 625, 700)
    node(draw, iq, ["IQ-balanced subset"], "N = 76", "(ASD=38, TD=38)")
    node(draw, strict, ["Strict specparam QC"], "N = 90", "(ASD=44, TD=46)")

    draw.line((105, 845, 1695, 845), fill="#E6E6E6", width=2)
    draw.text((40, 865), "b", font=F_PANEL, fill=TEXT)
    draw.text((118, 873), "EXTERNAL HBN COHORT", font=F_TITLE, fill=TD)

    hbn_top = (420, 940, 840, 1045)
    hbn_movie = (960, 940, 1380, 1045)
    node(draw, hbn_top, ["HBN The Present matched cohort"], "N = 238", "(ASD=119, TD=119)", PALE_TD)
    node(draw, hbn_movie, ["The Present movie Aperiodic-ISC"], "N = 238", "(ASD=119, TD=119)")
    y_mid = (hbn_top[1] + hbn_top[3]) // 2
    draw.line((hbn_top[2], y_mid, hbn_movie[0] - 10, y_mid), fill=ARROW, width=3)
    arrow_head(draw, hbn_movie[0] - 10, y_mid, "right")

    png = OUT / "SupplementaryFigureS1.png"
    img.save(png, dpi=(300, 300))
    img.save(OUT / "SupplementaryFigureS1.pdf", "PDF", resolution=300.0)
    print(png)


if __name__ == "__main__":
    make_s1()

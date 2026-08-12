#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "figures" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)
PNG = OUT / "Supplementary_Figure_S1_cohort_flow.png"
PDF = OUT / "Supplementary_Figure_S1_cohort_flow.pdf"

W, H = 1800, 1050
BG = "white"
TEXT = "#222222"
SUBTEXT = "#333333"
STROKE = "#555555"
ARROW = "#5A5A5A"
RED = "#C36B63"
BLUE = "#6A86B6"
PALE_RED = "#F6E8E6"
PALE_BLUE = "#E8EEF6"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_PANEL = font(44, True)
F_TITLE = font(31, True)
F_NODE = font(24, True)
F_N = font(23, True)
F_GROUP = font(20, True)


def center_line(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, fnt, fill=TEXT) -> None:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    draw.text((x - (bbox[2] - bbox[0]) / 2, y), text, font=fnt, fill=fill)


def center_multiline(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title_lines: list[str],
    n: str,
    groups: str,
) -> None:
    x1, y1, x2, y2 = box
    lines = [(line, F_NODE, TEXT) for line in title_lines]
    lines.append((n, F_N, TEXT))
    lines.append((groups, F_GROUP, SUBTEXT))
    line_metrics = []
    for line, fnt, _ in lines:
        bbox = draw.textbbox((0, 0), line, font=fnt)
        line_metrics.append((bbox, bbox[3] - bbox[1]))
    spacing = 6
    total_h = sum(h for _, h in line_metrics) + spacing * (len(lines) - 1)
    y = (y1 + y2 - total_h) / 2 - 1
    for (line, fnt, fill), (bbox, h) in zip(lines, line_metrics):
        bbox = draw.textbbox((0, 0), line, font=fnt)
        draw.text(((x1 + x2 - (bbox[2] - bbox[0])) / 2, y - bbox[1]), line, font=fnt, fill=fill)
        y += h + spacing


def node(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title_lines: list[str],
    n: str,
    groups: str,
    fill: str = "white",
) -> None:
    draw.rounded_rectangle(box, radius=10, fill=fill, outline=STROKE, width=2)
    center_multiline(draw, box, title_lines, n, groups)


def arrow_head(draw: ImageDraw.ImageDraw, x: int, y: int, direction: str) -> None:
    if direction == "down":
        pts = [(x, y), (x - 8, y - 16), (x + 8, y - 16)]
    elif direction == "right":
        pts = [(x, y), (x - 14, y - 7), (x - 14, y + 7)]
    else:
        raise ValueError(direction)
    draw.polygon(pts, fill=ARROW)


def down_arrow(draw: ImageDraw.ImageDraw, x: int, y1: int, y2: int) -> None:
    draw.line((x, y1, x, y2), fill=ARROW, width=3)
    arrow_head(draw, x, y2, "down")


def branch_down(draw: ImageDraw.ImageDraw, start: tuple[int, int], targets: list[tuple[int, int]]) -> None:
    sx, sy = start
    joint_y = sy + 46
    min_x = min(x for x, _ in targets)
    max_x = max(x for x, _ in targets)
    draw.line((sx, sy, sx, joint_y), fill=ARROW, width=3)
    draw.line((min_x, joint_y, max_x, joint_y), fill=ARROW, width=3)
    for tx, ty in targets:
        draw.line((tx, joint_y, tx, ty), fill=ARROW, width=3)
        arrow_head(draw, tx, ty, "down")


def elbow_right(draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int) -> None:
    draw.line((x1, y1, x1, y2, x2, y2), fill=ARROW, width=2)
    arrow_head(draw, x2, y2, "right")


def main() -> None:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    draw.text((40, 24), "A", font=F_PANEL, fill=TEXT)
    draw.text((118, 32), "PRIMARY COHORT", font=F_TITLE, fill=RED)

    top = (540, 90, 1260, 195)
    primary = (540, 275, 1260, 380)
    paired = (150, 505, 535, 625)
    movie = (708, 505, 1093, 625)
    matched = (1265, 505, 1650, 625)
    iq = (420, 700, 805, 820)
    strict = (995, 700, 1380, 820)

    node(draw, top, ["Resting EEG registration"], "N = 168", "(ASD=80, TD=88)", PALE_RED)
    node(draw, primary, ["Primary resting spectral cohort"], "N = 138", "(ASD=61, TD=77)")
    down_arrow(draw, 900, 195, 268)
    branch_down(draw, (900, 380), [(342, 505), (900, 505), (1458, 505)])
    node(draw, paired, ["Paired rest-to-movie exponent"], "N = 136", "(ASD=61, TD=75)")
    node(draw, movie, ["Movie Aperiodic-ISC cohort"], "N = 136", "(ASD=58, TD=78)")
    node(draw, matched, ["Resting + movie matched"], "N = 92", "(ASD=46, TD=46)")
    down_arrow(draw, 612, 430, 700)
    down_arrow(draw, 1188, 430, 700)
    node(draw, iq, ["IQ-balanced subset"], "N = 76", "(ASD=38, TD=38)")
    node(draw, strict, ["Strict specparam QC"], "N = 90", "(ASD=44, TD=46)")

    draw.line((105, 845, 1695, 845), fill="#E6E6E6", width=2)
    draw.text((40, 865), "B", font=F_PANEL, fill=TEXT)
    draw.text((118, 873), "EXTERNAL HBN COHORT", font=F_TITLE, fill=BLUE)

    hbn_top = (420, 940, 840, 1045)
    hbn_movie = (960, 940, 1380, 1045)
    node(draw, hbn_top, ["HBN The Present matched cohort"], "N = 238", "(ASD=119, TD=119)", PALE_BLUE)
    node(draw, hbn_movie, ["The Present movie Aperiodic-ISC"], "N = 238", "(ASD=119, TD=119)")
    y_mid = (hbn_top[1] + hbn_top[3]) // 2
    draw.line((hbn_top[2], y_mid, hbn_movie[0] - 10, y_mid), fill=ARROW, width=3)
    arrow_head(draw, hbn_movie[0] - 10, y_mid, "right")

    img.save(PNG, dpi=(300, 300))
    img.save(PDF, "PDF", resolution=300.0)
    print(PNG)
    print(PDF)


if __name__ == "__main__":
    main()

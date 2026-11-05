#!/usr/bin/env python3
"""Generate the animated GitHub profile header (banner + link arrows).

Run from the repository root:
    pip install -r scripts/banner/requirements.txt
    python scripts/banner/generate.py

Outputs (all self-contained SVG, SMIL + CSS, no JavaScript):
    assets/animated/banner.svg                 cinematic Salta banner
    assets/animated/link-<name>-<theme>.svg    arrow + icon tiles (one per link)

Technique credits: the particle choreography (optimal-transport matching of
particles to target points with scipy's linear_sum_assignment, and one SMIL
<animateTransform> per particle) is adapted from the idea used in
github.com/macu-dev/macu-dev (scripts/banner/generate.py). The code here is an
independent implementation for a different design.

Everything runs on one 14 s loop so the banner and the arrows feel like a
single composition:

    0.0 – 1.0  embers appear scattered over the landscape
    1.0 – 3.6  embers travel and organise into the letters (staggered)
    3.0 – 3.9  the crisp name comes into focus (blur 8 -> 0)
    4.0 – 5.2  subtitle is revealed by a wipe with a caret
    5.0 – 6.6  arrows draw towards Portfolio / LinkedIn / Gmail
    6.0 – 7.4  a light sweep crosses the name
    ~11.6      particles re-appear on the letters, name dissolves upward
    12.4 – 13  subtitle and arrows fade; landscape keeps moving
"""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path

import numpy as np
from fontTools.pens.basePen import BasePen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from PIL import Image, ImageChops, ImageDraw, ImageEnhance
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT_DIR = ROOT / "assets/animated"
BACKGROUND = ROOT / "assets/source/salta-clean.png"
FONT_TITLE = HERE / "fonts/Poppins-SemiBold.ttf"
FONT_SUB = HERE / "fonts/Poppins-Medium.ttf"
ICONS = HERE / "icons"

NAME = "Alejo Almada"
SUBTITLE = "Full Stack Developer  |  Explorando IA y Automatización"

W, H = 1200, 400
LOOP = 14.0
PARTICLES = 760
EMBERS = 34
SEED = 2026
EASE = "0.45 0 0.25 1"  # ease-in-out used for every moving segment

# Name / subtitle geometry (banner coordinates).
NAME_SIZE, NAME_BASELINE = 90, 186
SUB_SIZE, SUB_BASELINE = 23, 234

# Overscanned background so the slow pan never reveals an edge.
BG_X, BG_Y, BG_W, BG_H = -36, -12, 1272, 424
SUN = (BG_X + 0.228 * BG_W, BG_Y + 0.313 * BG_H)


# ───────────────────────────── helpers ──────────────────────────────
def f1(v: float) -> str:
    s = f"{v:.1f}"
    s = s.rstrip("0").rstrip(".") if "." in s else s
    return "0" if s in ("-0", "") else s


def kt(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".") or "0"


def timeline(points: list[tuple[float, str]]) -> tuple[str, str]:
    """(seconds, value) pairs on the LOOP -> (keyTimes, values) strings."""
    pts = sorted(points, key=lambda p: p[0])
    if pts[0][0] > 0:
        pts.insert(0, (0.0, pts[0][1]))
    if pts[-1][0] < LOOP:
        pts.append((LOOP, pts[-1][1]))
    # keyTimes must be strictly non-decreasing; nudge exact duplicates.
    times, last = [], -1.0
    for t, _ in pts:
        t = max(t, last + 1e-4) if last >= 0 else t
        times.append(min(t, LOOP))
        last = t
    times[-1] = LOOP
    return ";".join(kt(t / LOOP) for t in times), ";".join(v for _, v in pts)


def anim(attr: str, points: list[tuple[float, str]], spline: bool = False, tag: str = "animate",
         extra: str = "") -> str:
    keys, values = timeline(points)
    n = len(values.split(";"))
    mode = (f' calcMode="spline" keySplines="{";".join([EASE] * (n - 1))}"' if spline else "")
    return (f'<{tag} attributeName="{attr}" dur="{f1(LOOP)}s" repeatCount="indefinite" '
            f'keyTimes="{keys}" values="{values}"{mode}{extra}/>')


# ───────────────────────────── text → outlines ──────────────────────
class FlattenPen(BasePen):
    """Collect glyph contours as polylines (for rasterising particle targets)."""

    def __init__(self, glyphset):
        super().__init__(glyphset)
        self.contours: list[list[tuple[float, float]]] = []
        self.cur: list[tuple[float, float]] = []

    def _moveTo(self, p):
        self.cur = [p]

    def _lineTo(self, p):
        self.cur.append(p)

    def _curveToOne(self, p1, p2, p3):
        p0 = self.cur[-1]
        for t in np.linspace(0, 1, 9)[1:]:
            mt = 1 - t
            self.cur.append((mt**3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t**3 * p3[0],
                             mt**3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t**3 * p3[1]))

    def _qCurveToOne(self, p1, p2):
        p0 = self.cur[-1]
        for t in np.linspace(0, 1, 7)[1:]:
            mt = 1 - t
            self.cur.append((mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0],
                             mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1]))

    def _closePath(self):
        if len(self.cur) > 2:
            self.contours.append(self.cur)
        self.cur = []

    _endPath = _closePath


def kerning_pairs(font: TTFont) -> dict[tuple[str, str], int]:
    """Read simple PairPos (format 1/2) kerning from GPOS, enough for Latin text."""
    pairs: dict[tuple[str, str], int] = {}
    if "GPOS" not in font:
        return pairs
    gpos = font["GPOS"].table
    for lookup in gpos.LookupList.Lookup:
        for sub in lookup.SubTable:
            if lookup.LookupType == 9:
                sub = sub.ExtSubTable
            if getattr(sub, "LookupType", lookup.LookupType) != 2:
                continue
            cov = sub.Coverage.glyphs
            if sub.Format == 1:
                for i, first in enumerate(cov):
                    for rec in sub.PairSet[i].PairValueRecord:
                        v = getattr(rec.Value1, "XAdvance", 0) if rec.Value1 else 0
                        pairs.setdefault((first, rec.SecondGlyph), v)
            elif sub.Format == 2:
                c1 = sub.ClassDef1.classDefs
                c2 = sub.ClassDef2.classDefs
                for first in cov:
                    row = sub.Class1Record[c1.get(first, 0)]
                    for second, cls in list(c2.items()):
                        rec = row.Class2Record[cls]
                        v = getattr(rec.Value1, "XAdvance", 0) if rec.Value1 else 0
                        if v:
                            pairs.setdefault((first, second), v)
    return pairs


def text_outline(font_path: Path, text: str, size: float, cx: float, baseline: float,
                 tracking: float = 0.0):
    """Return (svg path d, contours, width) for text centred at cx on baseline."""
    font = TTFont(font_path)
    gs, cmap = font.getGlyphSet(), font.getBestCmap()
    upm = font["head"].unitsPerEm
    scale = size / upm
    kern = kerning_pairs(font)
    names = [cmap[ord(c)] for c in text]
    advances = []
    for i, g in enumerate(names):
        adv = font["hmtx"][g][0]
        if i + 1 < len(names):
            adv += kern.get((g, names[i + 1]), 0)
        advances.append(adv * scale + tracking)
    width = sum(advances) - tracking
    x = cx - width / 2
    svg_pen = SVGPathPen(gs, ntos=lambda v: f1(v))
    flat = FlattenPen(gs)
    for g, adv in zip(names, advances):
        m = (scale, 0, 0, -scale, x, baseline)
        gs[g].draw(TransformPen(svg_pen, m))
        gs[g].draw(TransformPen(flat, m))
        x += adv
    return svg_pen.getCommands(), flat.contours, width


def raster_mask(contours, ss: int = 2) -> np.ndarray:
    """Even-odd rasterisation of contours at W×H (supersampled)."""
    acc = Image.new("1", (W * ss, H * ss), 0)
    for c in contours:
        layer = Image.new("1", acc.size, 0)
        ImageDraw.Draw(layer).polygon([(x * ss, y * ss) for x, y in c], fill=1)
        acc = ImageChops.logical_xor(acc, layer)
    return np.asarray(acc.resize((W, H), Image.Resampling.BOX).convert("L")) > 100


# ───────────────────────────── background ───────────────────────────
def background_data_uri() -> str:
    img = Image.open(BACKGROUND).convert("RGB")
    img = img.resize((1600, int(1600 * img.height / img.width)), Image.Resampling.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.04)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=76, optimize=True, progressive=True, subsampling=2)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ───────────────────────────── banner ───────────────────────────────
def build_banner() -> str:
    rng = np.random.default_rng(SEED)
    name_d, name_contours, name_w = text_outline(FONT_TITLE, NAME, NAME_SIZE, W / 2, NAME_BASELINE,
                                                 tracking=0.5)
    sub_d, _, sub_w = text_outline(FONT_SUB, SUBTITLE, SUB_SIZE, W / 2, SUB_BASELINE, tracking=0.4)

    # Particle targets: random pixels inside the glyphs, slightly jittered.
    inside = np.argwhere(raster_mask(name_contours))[:, ::-1].astype(float)
    targets = inside[rng.choice(len(inside), PARTICLES, replace=False)]
    targets += rng.uniform(-0.4, 0.4, targets.shape)

    # Scatter: wide drift over the sky/valley, denser around the title band.
    wide = np.column_stack((rng.uniform(30, W - 30, PARTICLES), rng.uniform(20, H - 30, PARTICLES)))
    near = np.column_stack((rng.normal(W / 2, 260, PARTICLES), rng.normal(160, 80, PARTICLES)))
    pick = rng.random(PARTICLES) < 0.55
    scatter = np.where(pick[:, None], near, wide)
    scatter[:, 0] = np.clip(scatter[:, 0], 10, W - 10)
    scatter[:, 1] = np.clip(scatter[:, 1], 10, H - 10)
    # Optimal transport: every ember gets the letter point that minimises the
    # total squared travel, so the swarm flows coherently instead of crossing.
    rows, cols = linear_sum_assignment(cdist(scatter, targets, "sqeuclidean"))
    targets = targets[cols[np.argsort(rows)]]

    x_min, x_max = targets[:, 0].min(), targets[:, 0].max()
    parts: list[str] = []
    add = parts.append

    add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">')
    add('<title id="t">Alejo Almada — Full Stack Developer | Explorando IA y Automatización</title>')
    add('<desc id="d">Atardecer animado en Salta, Argentina: partículas de luz forman el nombre '
        'Alejo Almada y luego aparece el subtítulo.</desc>')
    add("<style>"
        ".static{display:none}"
        "@media (prefers-reduced-motion: reduce){"
        ".motion,.anim{display:none!important}"
        ".static{display:inline!important}"
        ".pan,.drift{transform:none!important}}"
        "</style>")

    # ── defs
    add("<defs>")
    add(f'<clipPath id="frame"><rect width="{W}" height="{H}" rx="18"/></clipPath>')
    add(f'<image id="photo" x="{BG_X}" y="{BG_Y}" width="{BG_W}" height="{BG_H}" '
        f'preserveAspectRatio="none" href="{background_data_uri()}"/>')
    add('<linearGradient id="skyFade" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#fff"/><stop offset=".15" stop-color="#fff"/>'
        '<stop offset=".22" stop-color="#000"/></linearGradient>')
    add(f'<mask id="skyMask" maskUnits="userSpaceOnUse" x="-80" y="0" width="{W + 160}" height="{H}">'
        f'<rect x="-80" width="{W + 160}" height="{H}" fill="url(#skyFade)"/></mask>')
    add('<radialGradient id="sun"><stop offset="0" stop-color="#FFE3A3" stop-opacity=".95"/>'
        '<stop offset=".18" stop-color="#FFB45C" stop-opacity=".55"/>'
        '<stop offset=".5" stop-color="#FF7A2E" stop-opacity=".16"/>'
        '<stop offset="1" stop-color="#FF5A1F" stop-opacity="0"/></radialGradient>')
    add('<linearGradient id="ray" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0" stop-color="#FFD08A" stop-opacity=".55"/>'
        '<stop offset="1" stop-color="#FFD08A" stop-opacity="0"/></linearGradient>')
    add('<radialGradient id="mist"><stop offset="0" stop-color="#FFB27A" stop-opacity=".2"/>'
        '<stop offset="1" stop-color="#FFB27A" stop-opacity="0"/></radialGradient>')
    add('<radialGradient id="scrim" cx=".5" cy=".5" r=".5">'
        '<stop offset="0" stop-color="#140806" stop-opacity=".52"/>'
        '<stop offset=".6" stop-color="#140806" stop-opacity=".22"/>'
        '<stop offset="1" stop-color="#140806" stop-opacity="0"/></radialGradient>')
    add('<radialGradient id="vignette" cx=".5" cy=".45" r=".75">'
        '<stop offset=".6" stop-color="#000" stop-opacity="0"/>'
        '<stop offset="1" stop-color="#000" stop-opacity=".38"/></radialGradient>')
    add(f'<linearGradient id="nameFill" x1="0" y1="{NAME_BASELINE - 66}" x2="0" y2="{NAME_BASELINE + 4}" '
        'gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#FFF8EC"/>'
        '<stop offset="1" stop-color="#F6DDB5"/></linearGradient>')
    shine = timeline([(0, "-700 0"), (6.0, "-700 0"), (7.6, "760 0"), (LOOP, "760 0")])
    add(f'<linearGradient id="shine" x1="{W / 2 - 420}" y1="0" x2="{W / 2 - 250}" y2="0" '
        'gradientUnits="userSpaceOnUse">'
        '<stop offset="0" stop-color="#fff" stop-opacity="0"/>'
        '<stop offset=".5" stop-color="#fff" stop-opacity=".85"/>'
        '<stop offset="1" stop-color="#fff" stop-opacity="0"/>'
        f'<animateTransform attributeName="gradientTransform" type="translate" dur="{f1(LOOP)}s" '
        f'repeatCount="indefinite" keyTimes="{shine[0]}" values="{shine[1]}" calcMode="spline" '
        f'keySplines="{EASE};{EASE};{EASE}"/></linearGradient>')
    add('<filter id="glow" x="-10%" y="-40%" width="120%" height="180%">'
        '<feGaussianBlur in="SourceAlpha" stdDeviation="7" result="b"/>'
        '<feFlood flood-color="#FF9A3C" flood-opacity=".55"/><feComposite in2="b" operator="in" result="g"/>'
        '<feGaussianBlur in="SourceAlpha" stdDeviation="2" result="s"/>'
        '<feFlood flood-color="#1A0905" flood-opacity=".55"/><feComposite in2="s" operator="in" result="sh"/>'
        '<feOffset in="sh" dy="2" result="sh2"/>'
        '<feMerge><feMergeNode in="g"/><feMergeNode in="sh2"/><feMergeNode in="SourceGraphic"/></feMerge></filter>')
    focus = [(0, "8"), (3.0, "8"), (3.9, "0"), (11.7, "0"), (12.5, "9")]
    add('<filter id="focus" x="-10%" y="-60%" width="120%" height="220%">'
        '<feGaussianBlur stdDeviation="0">' + anim("stdDeviation", focus, spline=True)
        + "</feGaussianBlur></filter>")
    add('<filter id="soft"><feGaussianBlur stdDeviation="1.1"/></filter>')
    add(f'<path id="nameP" d="{name_d}"/>')
    add(f'<path id="subP" d="{sub_d}"/>')
    add(f'<clipPath id="nameClip"><use href="#nameP"/></clipPath>')
    sub_x0 = W / 2 - sub_w / 2 - 4
    wipe = [(0, "0"), (4.0, "0"), (5.25, f1(sub_w + 10)), (13.0, f1(sub_w + 10)), (13.05, "0")]
    add(f'<clipPath id="subClip"><rect x="{f1(sub_x0)}" y="{SUB_BASELINE - 26}" '
        f'width="{f1(sub_w + 10)}" height="36">' + anim("width", wipe, spline=True) + "</rect></clipPath>")
    add("</defs>")

    add('<g clip-path="url(#frame)">')
    add(f'<rect width="{W}" height="{H}" fill="#2a120c"/>')

    # ── landscape: slow Ken Burns pan; the sky layer drifts further (parallax)
    add('<g class="pan"><animateTransform attributeName="transform" type="translate" dur="42s" '
        'repeatCount="indefinite" values="0 0;-22 -4;8 -6;0 0" keyTimes="0;.4;.75;1" calcMode="spline" '
        f'keySplines="{EASE};{EASE};{EASE}"/>')
    add('<use href="#photo"/>')
    add('<g mask="url(#skyMask)"><g class="drift"><animateTransform attributeName="transform" '
        'type="translate" dur="56s" repeatCount="indefinite" values="0 0;-26 0;0 0" keyTimes="0;.5;1" '
        f'calcMode="spline" keySplines="{EASE};{EASE}"/><use href="#photo"/></g></g>')

    # Sun: breathing glow + slowly turning rays.
    sx, sy = SUN
    add(f'<g style="mix-blend-mode:screen"><circle cx="{f1(sx)}" cy="{f1(sy)}" r="120" fill="url(#sun)" '
        'opacity=".8"><animate attributeName="opacity" dur="7s" repeatCount="indefinite" '
        f'values=".65;.95;.65" calcMode="spline" keySplines="{EASE};{EASE}"/>'
        '<animate attributeName="r" dur="7s" repeatCount="indefinite" values="110;132;110" '
        f'calcMode="spline" keySplines="{EASE};{EASE}"/></circle>')
    add(f'<g class="motion" opacity=".22"><animateTransform attributeName="transform" type="rotate" '
        f'dur="30s" repeatCount="indefinite" values="-4 {f1(sx)} {f1(sy)};4 {f1(sx)} {f1(sy)};'
        f'-4 {f1(sx)} {f1(sy)}" calcMode="spline" keySplines="{EASE};{EASE}"/>'
        '<animate attributeName="opacity" dur="11s" repeatCount="indefinite" values=".14;.28;.14"/>')
    for ang, length, width in ((-168, 520, 16), (-150, 600, 10), (-128, 440, 14), (-104, 380, 8),
                               (-60, 460, 12), (-34, 700, 18), (-14, 760, 10), (8, 640, 12)):
        add(f'<path d="M0 0L{length} {-width}L{length} {width}Z" fill="url(#ray)" '
            f'transform="translate({f1(sx)} {f1(sy)}) rotate({ang})"/>')
    add("</g></g>")

    # Valley mist bands drifting at different speeds (depth cue).
    for cy, rx, ry, dur, dx, op in ((262, 420, 34, 26, 90, .9), (300, 520, 42, 34, -120, .7)):
        add(f'<ellipse class="motion" cx="{W / 2}" cy="{cy}" rx="{rx}" ry="{ry}" fill="url(#mist)" '
            f'opacity="{op}"><animateTransform attributeName="transform" type="translate" dur="{dur}s" '
            f'repeatCount="indefinite" values="{-dx} 0;{dx} 0;{-dx} 0" calcMode="spline" '
            f'keySplines="{EASE};{EASE}"/></ellipse>')
    add("</g>")  # /pan

    add(f'<rect width="{W}" height="{H}" fill="url(#vignette)"/>')
    add(f'<ellipse cx="{W / 2}" cy="{(NAME_BASELINE + SUB_BASELINE) / 2 - 22}" rx="{f1(name_w * 0.78)}" '
        f'ry="105" fill="url(#scrim)"/>')

    # ── ambient embers rising (independent cycles, never synced = organic)
    add('<g class="motion" fill="#FFC77D">')
    for _ in range(EMBERS):
        x, y = rng.uniform(20, W - 20), rng.uniform(140, H)
        r = rng.uniform(0.7, 1.7)
        dur = rng.uniform(7, 13)
        dx, dy = rng.uniform(-30, 30), -rng.uniform(60, 150)
        peak = rng.uniform(.45, .85)
        begin = -rng.uniform(0, dur)
        add(f'<circle cx="{f1(x)}" cy="{f1(y)}" r="{r:.1f}" opacity="0">'
            f'<animateTransform attributeName="transform" type="translate" dur="{dur:.1f}s" '
            f'begin="{begin:.1f}s" repeatCount="indefinite" values="0 0;{f1(dx)} {f1(dy)}"/>'
            f'<animate attributeName="opacity" dur="{dur:.1f}s" begin="{begin:.1f}s" '
            f'repeatCount="indefinite" values="0;{peak:.2f};{peak * .6:.2f};0" keyTimes="0;.25;.7;1"/>'
            "</circle>")
    add("</g>")

    # ── particle name
    add('<g class="motion" fill="#FFE6BC">')
    for i in range(PARTICLES):
        sx_, sy_ = scatter[i]
        tx, ty = targets[i]
        frac = (tx - x_min) / max(x_max - x_min, 1)
        appear = rng.uniform(0.0, 0.9)
        move0 = 1.0 + 0.45 * frac + rng.uniform(0, 0.35)
        arrive = move0 + rng.uniform(1.55, 1.85)
        fade0 = rng.uniform(3.6, 4.1)
        back = 11.25 + 0.35 * frac + rng.uniform(0, 0.15)
        d0 = 11.6 + 0.75 * frac + rng.uniform(0, 0.15)
        d1 = d0 + rng.uniform(0.9, 1.15)
        ang = rng.uniform(-2.4, -0.7)
        dist = rng.uniform(25, 90)
        ex, ey = tx + np.cos(ang) * dist, ty + np.sin(ang) * dist - 18
        S, T, E = f"{f1(sx_)} {f1(sy_)}", f"{f1(tx)} {f1(ty)}", f"{f1(ex)} {f1(ey)}"
        pos = [(0, S), (move0, S), (arrive, T), (d0, T), (d1, E), (13.7, E), (13.72, S)]
        peak = f"{rng.uniform(.75, 1):.2f}"
        op = [(0, "0"), (appear, "0"), (appear + .5, peak), (arrive, "1"), (fade0, "1"),
              (fade0 + .5, "0"), (back, "0"), (back + .3, "1"), (d0, "1"), (d1, "0")]
        r = rng.choice((0.9, 1.1, 1.3, 1.6), p=(.3, .35, .25, .1))
        add(f'<circle r="{r}" opacity="0">'
            + anim("transform", pos, spline=True, tag="animateTransform", extra=' type="translate"')
            + anim("opacity", op) + "</circle>")
    add("</g>")

    # ── crisp name (focus pull) + light sweep
    name_op = [(0, "0"), (3.0, "0"), (3.9, "1"), (11.7, "1"), (12.5, "0")]
    add('<g class="anim" filter="url(#focus)" opacity="1">' + anim("opacity", name_op, spline=True))
    add('<use href="#nameP" fill="url(#nameFill)" filter="url(#glow)"/>')
    add(f'<rect x="{f1(W / 2 - name_w / 2 - 10)}" y="{NAME_BASELINE - 80}" width="{f1(name_w + 20)}" '
        f'height="110" fill="url(#shine)" clip-path="url(#nameClip)" opacity=".55"/>')
    add("</g>")

    # Twinkles resting on the letters while the name holds.
    add('<g class="motion" fill="#FFF3D6">')
    for j in range(16):
        tx, ty = targets[rng.integers(PARTICLES)]
        t0 = rng.uniform(4.6, 10.4)
        s = rng.uniform(3, 5.5)
        tw = [(0, "0"), (t0, "0"), (t0 + .35, ".95"), (t0 + .9, "0")]
        add(f'<path d="M{f1(tx)} {f1(ty - s)}Q{f1(tx)} {f1(ty)} {f1(tx + s)} {f1(ty)}Q{f1(tx)} {f1(ty)} '
            f'{f1(tx)} {f1(ty + s)}Q{f1(tx)} {f1(ty)} {f1(tx - s)} {f1(ty)}Q{f1(tx)} {f1(ty)} {f1(tx)} '
            f'{f1(ty - s)}Z" opacity="0">' + anim("opacity", tw) + "</path>")
    add("</g>")

    # ── subtitle: wipe reveal + caret
    sub_op = [(0, "0"), (3.95, "0"), (4.25, "1"), (12.4, "1"), (13.0, "0")]
    add('<g class="anim" opacity="1">' + anim("opacity", sub_op))
    add(f'<g clip-path="url(#subClip)"><use href="#subP" fill="#F2C063" filter="url(#glow)"/></g>')
    caret_x = [(0, f1(sub_x0)), (4.0, f1(sub_x0)), (5.25, f1(sub_x0 + sub_w + 10))]
    caret_op = [(0, "0"), (3.95, "0"), (4.05, "1"), (5.3, "1"), (5.55, "0"), (5.8, "1"), (6.05, "0")]
    add(f'<rect x="{f1(sub_x0)}" y="{SUB_BASELINE - 20}" width="2" height="25" rx="1" fill="#FFD58A" '
        'opacity="0">' + anim("x", caret_x, spline=True) + anim("opacity", caret_op) + "</rect>")
    add("</g>")

    # ── reduced-motion / no-animation fallback (shown only via CSS media query)
    add('<g class="static"><use href="#nameP" fill="url(#nameFill)" filter="url(#glow)"/>'
        '<use href="#subP" fill="#F2C063" filter="url(#glow)"/></g>')

    # Fine border light.
    add(f'<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="17.5" fill="none" '
        'stroke="#FFD7A0" stroke-opacity=".18"/>')
    add("</g></svg>")
    return "".join(parts)


# ───────────────────────────── arrow tiles ──────────────────────────
TILE_W, TILE_H = 112, 142
ICON = 48
ICON_Y = TILE_H - ICON - 12
TIP = (TILE_W / 2, ICON_Y - 9)

LINKS = {
    # name: (icon file, shaft path, draw delay)
    "portfolio": ("vercel", "M101 5C97 21 84 23 74 30S56 44 57 58 56 70 56 71", 0.25),
    "linkedin": ("linkedin", "M56 3C51 15 61 25 56 38S53 58 56 71", 0.0),
    "gmail": ("gmail", "M11 5C15 21 28 23 38 30S56 44 55 58 56 70 56 71", 0.5),
}

THEME_COLORS = {
    "dark": {"stroke": "#F2B45A", "glow": "#FF9A3C", "spark": "#FFF0C8", "ring": "#F2B45A"},
    "light": {"stroke": "#D07F1C", "glow": "#F0A040", "spark": "#B85A00", "ring": "#D07F1C"},
}


def icon_markup(name: str) -> str:
    raw = (ICONS / f"{name}.svg").read_text(encoding="utf-8")
    inner = re.search(r"<svg width=\"256\".*?>(.*)</svg>\s*</g>", raw, re.S)
    if not inner:
        raise SystemExit(f"Unexpected icon format: {name}")
    x = (TILE_W - ICON) / 2
    return (f'<svg x="{f1(x)}" y="{ICON_Y}" width="{ICON}" height="{ICON}" viewBox="0 0 256 256" '
            f'fill="none">{inner.group(1).strip()}</svg>')


def build_tile(link: str, theme: str) -> str:
    icon, shaft, delay = LINKS[link]
    c = THEME_COLORS[theme]
    tx, ty = TIP
    d0 = 5.0 + delay          # shaft starts drawing
    d1 = d0 + 1.05            # shaft done
    h1 = d1 + 0.3             # head done
    f0, f1_ = 11.8 + delay, 12.5 + delay

    dash = [(0, "100"), (d0, "100"), (d1, "0"), (13.6, "0"), (13.62, "100")]
    head_dash = [(0, "100"), (d1 - .05, "100"), (h1, "0"), (13.6, "0"), (13.62, "100")]
    fade = [(0, "0"), (d0 - .05, "0"), (d0, "1"), (f0, "1"), (f1_, "0")]
    # Spark: runs with the pen while drawing, then twice more during the hold.
    runs = [(d0, d1), (d1 + 2.1, d1 + 3.0), (d1 + 4.5, d1 + 5.4)]
    kp, sp_op = [(0, "0")], [(0, "0")]
    for a, b in runs:
        kp += [(a - .01, "0"), (a, "0"), (b, "1"), (b + .01, "1")]
        sp_op += [(a - .01, "0"), (a + .08, "1"), (b - .05, "1"), (b + .15, "0")]
    kp_keys, kp_vals = timeline(kp)
    ring_r = [(0, "25"), (h1 - .05, "25"), (h1 + .9, "33")]
    ring_op = [(0, "0"), (h1 - .05, "0"), (h1, ".7"), (h1 + .9, "0")]

    p: list[str] = []
    add = p.append
    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{TILE_W}" height="{TILE_H}" '
        f'viewBox="0 0 {TILE_W} {TILE_H}" role="img" aria-label="{link}">')
    add(f"<title>{link.capitalize()}</title>")
    add("<style>@media (prefers-reduced-motion: reduce){.motion{display:none}"
        ".draw{stroke-dashoffset:0!important;opacity:1!important}}</style>")
    add('<defs><filter id="g" x="-50%" y="-50%" width="200%" height="200%">'
        '<feGaussianBlur stdDeviation="2.4"/></filter>'
        f'<path id="shaft" d="{shaft}"/></defs>')
    head = f"M{f1(tx - 7)} {f1(ty - 9)}L{f1(tx)} {f1(ty)}L{f1(tx + 7)} {f1(ty - 9)}"
    stroke = (f'fill="none" stroke-linecap="round" stroke-linejoin="round" pathLength="100" '
              f'stroke-dasharray="100" class="draw"')
    add('<g class="draw" opacity="1">' + anim("opacity", fade, spline=True))
    # Soft amber glow under the line, then the line itself.
    for width, color, extra in ((5, c["glow"], ' filter="url(#g)" opacity=".55"'), (2.2, c["stroke"], "")):
        add(f'<g{extra}>')
        add(f'<use href="#shaft" {stroke} stroke="{color}" stroke-width="{width}" stroke-dashoffset="0">'
            + anim("stroke-dashoffset", dash, spline=True) + "</use>")
        add(f'<path d="{head}" {stroke} stroke="{color}" stroke-width="{width}" stroke-dashoffset="0">'
            + anim("stroke-dashoffset", head_dash, spline=True) + "</path>")
        add("</g>")
    add("</g>")
    # Travelling spark.
    add(f'<g class="motion"><circle r="2.4" fill="{c["spark"]}" opacity="0">'
        f'<animateMotion dur="{f1(LOOP)}s" repeatCount="indefinite" keyPoints="{kp_vals}" '
        f'keyTimes="{kp_keys}" calcMode="linear"><mpath href="#shaft"/></animateMotion>'
        + anim("opacity", sp_op) + "</circle>")
    # Arrival ring around the icon.
    add(f'<circle cx="{TILE_W / 2}" cy="{ICON_Y + ICON / 2}" r="24" fill="none" stroke="{c["ring"]}" '
        'stroke-width="1.5" opacity="0">' + anim("r", ring_r, spline=True) + anim("opacity", ring_op)
        + "</circle></g>")
    add(icon_markup(icon))
    add("</svg>")
    return "".join(p)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {"banner.svg": build_banner()}
    for link in LINKS:
        for theme in THEME_COLORS:
            outputs[f"link-{link}-{theme}.svg"] = build_tile(link, theme)
    for name, svg in outputs.items():
        path = OUT_DIR / name
        path.write_text(svg, encoding="utf-8")
        print(f"{path.relative_to(ROOT)}: {path.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()

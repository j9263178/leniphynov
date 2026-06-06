"""
Attractor-glyph pipeline (clean, standalone, works on ANY text)

    文字 → 語意向量 → 混沌吸引子點雲 → 密度場 → 門檻化/基元抽取 → 扁平字形

Extracted & generalised from the work/ attractor_* demos.  The original only
handled a fixed list of hand-tuned words; here the semantic vector and the
attractor parameters are derived deterministically from the text itself
(real Gemini embedding if available, else a stable hash), so the same text
always yields the same glyph and similar text yields similar glyphs.

run:  python attractor_glyph.py "床前明月光" "quantum mechanics" cat dog
"""
from __future__ import annotations
import sys
import math
import hashlib
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

# Optional: reuse the project's 256-d embedding. Falls back to hash if absent.
try:
    import lenia_text as _L
except Exception:
    _L = None


# ── 1. 文字 → 語意向量 ─────────────────────────────────────────────────────────
def semantic(text: str, dim: int = 10) -> np.ndarray:
    """Deterministic semantic vector in [0,1]^dim.

    Uses the real embedding (mean-pooled into `dim` bands) when available so
    that meaning drives the shape; otherwise a stable per-text hash vector.
    """
    v = _L._embed(text) if _L is not None else None
    if v is not None:
        v = np.asarray(v, dtype=np.float64)
        # fixed random projection 256→dim: preserves inter-text variation
        # (mean-pooling collapses different texts to near-identical vectors).
        W = np.random.default_rng(42).standard_normal((len(v), dim))
        z = v @ W
    else:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        z = np.random.default_rng(seed).standard_normal(dim)
    # squash each component to [0,1] with a fixed scale (keep spread)
    return 1.0 / (1.0 + np.exp(-z / 1.5))


def _stable_noise(text: str, n: int, scale: float = 0.04) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    return np.random.default_rng(seed).normal(0, scale, n)


# Known-good chaotic parameter sets (each yields a dense, compact attractor).
_PARAM_BANK = np.array([
    [-1.62, 1.72, -1.18, -1.42],
    [-1.72, 1.82, -1.24, -1.53],
    [1.54, -2.18, 1.87, -0.72],
    [-1.92, -1.31, 1.83, 0.88],
    [-1.36, 1.94, -0.82, -1.18],
    [1.42, -2.05, 1.64, -0.84],
], dtype=np.float64)


# ── 2. 語意向量 → 混沌吸引子點雲 ──────────────────────────────────────────────
def attractor_points(text: str, count: int = 220_000,
                     bank_offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Clifford/de-Jong style attractor; params derived from the semantic vector.

    Recurrence (same family as the original demos):
        x, y = sin(a·y) + c·cos(a·x),  sin(b·x) + d·cos(b·y)

    `bank_offset` rotates to a different known-good parameter set — used to
    retry when a chosen attractor comes out too sparse to fill a totem.
    """
    v = semantic(text)
    # Pick a KNOWN-GOOD chaotic parameter set from the bank (guarantees a
    # non-degenerate, compact attractor), then perturb by the semantic vector
    # so each text still gets its own distinct shape.  (Strategy from the
    # original sentence_glyph_line_demo — far more robust than free mapping.)
    seed = int.from_bytes(hashlib.sha256(("bank:" + text).encode()).digest()[:8], "big")
    params = _PARAM_BANK[(seed + bank_offset) % len(_PARAM_BANK)].copy()
    params += np.array([
        -0.18 + 0.36 * v[0] + 0.06 * v[7],
        -0.18 + 0.42 * v[8] - 0.05 * v[5],
        -0.16 + 0.34 * v[3] + 0.05 * v[6],
        -0.16 + 0.34 * v[2] - 0.05 * v[1],
    ])
    params += _stable_noise("params:" + text, 4, 0.08)
    a, b, c, d = params

    xy = np.empty((count, 2), dtype=np.float32)
    x = 0.08 + 0.4 * v[4]
    y = -0.11 + 0.3 * v[5]
    burn = 1600
    for i in range(count + burn):
        x, y = (math.sin(a * y) + c * math.cos(a * x),
                math.sin(b * x) + d * math.cos(b * y))
        if i >= burn:
            xy[i - burn] = (x, y)
    return xy, v


# ── 3. 點雲 → 密度場 ──────────────────────────────────────────────────────────
def normalize(xy: np.ndarray, size: int, pad: int) -> np.ndarray:
    lo = np.percentile(xy, 1.0, axis=0)
    hi = np.percentile(xy, 99.0, axis=0)
    out = np.clip((xy - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    out[:, 1] = 1.0 - out[:, 1]                       # flip y for image coords
    return out * (size - pad * 2) + pad


def density_field(xy: np.ndarray, size: int) -> np.ndarray:
    canvas = np.zeros((size, size), dtype=np.float32)
    xi = xy[:, 0].astype(np.int32)
    yi = xy[:, 1].astype(np.int32)
    good = (xi >= 0) & (xi < size) & (yi >= 0) & (yi < size)
    np.add.at(canvas, (yi[good], xi[good]), 1.0)
    canvas = np.log1p(canvas)                          # compress dynamic range
    return canvas / max(1e-6, canvas.max())


# ── 4. 門檻化 → 扁平字形 ──────────────────────────────────────────────────────
def _palette(v: np.ndarray) -> tuple[int, int, int]:
    if v[0] > 0.66:
        return (72, 228, 157)      # green
    if v[8] > 0.70:
        return (192, 122, 232)     # purple
    if v[5] > 0.66:
        return (230, 96, 184)      # pink
    return (88, 184, 235)          # blue


def _top_mask(field: np.ndarray, keep_percent: float) -> np.ndarray:
    nz = field[field > 0]
    thr = np.percentile(nz, 100 - keep_percent) if len(nz) else 1.0
    return (field >= thr)


def _fatten(mask_img: Image.Image, thickness: int) -> Image.Image:
    """Morphologically dilate a binary mask to thicken the strokes.

    thickness = total stroke growth in pixels (0 = no change).
    """
    if thickness <= 0:
        return mask_img
    out = mask_img
    # MaxFilter only takes odd kernels ≥3; apply repeatedly for larger growth.
    remaining = thickness
    while remaining > 0:
        k = min(9, remaining * 2 + 1)
        if k < 3:
            break
        out = out.filter(ImageFilter.MaxFilter(k))
        remaining -= (k - 1) // 2
    return out


def render_glyph(text: str, size: int = 480, keep_percent: float = 16.0,
                 thickness: int = 5, smooth: float = 1.8,
                 bg=(8, 10, 11)) -> tuple[Image.Image, np.ndarray]:
    """Full pipeline → a flat, colourised glyph image.

    `smooth`    blurs the density field BEFORE thresholding so nearby orbit
                lines fuse into continuous ribbons (no dotty fragments).
    `thickness` then fattens those ribbons (px) for a bold figure.
    `keep_percent` controls how much of the attractor survives.
    """
    raw, v = attractor_points(text)
    xy = normalize(raw, size, int(size * 0.11))
    field = density_field(xy, size)
    main = _palette(v)

    # smear the field so the orbit fuses into solid ribbons before masking
    field_s = field
    if smooth > 0:
        field_s = np.asarray(
            Image.fromarray(np.uint8(field * 255), "L")
                 .filter(ImageFilter.GaussianBlur(radius=smooth)),
            dtype=np.float32) / 255.0

    # flat ink mask from the densest ridges, then THICKEN
    core   = _top_mask(field_s, keep_percent)
    bright = _top_mask(field_s, keep_percent * 0.40)
    core_img   = _fatten(Image.fromarray(np.uint8(core * 255), "L"), thickness)
    bright_img = _fatten(Image.fromarray(np.uint8(bright * 255), "L"),
                         max(1, thickness - 2))

    img = Image.new("RGB", (size, size), bg)
    # soft glow halo (grows with thickness)
    glow = core_img.filter(ImageFilter.GaussianBlur(radius=3.0 + thickness)).point(lambda p: int(p * 0.22))
    img.paste(Image.new("RGB", (size, size), tuple(int(c * 0.5) for c in main)), mask=glow)
    # flat body
    img.paste(Image.new("RGB", (size, size), main),
              mask=core_img.filter(ImageFilter.GaussianBlur(radius=0.6)))
    # bright core
    img.paste(Image.new("RGB", (size, size), (232, 250, 240)),
              mask=bright_img.filter(ImageFilter.GaussianBlur(radius=0.4)).point(lambda p: int(p * 0.7)))
    return img, field


# ── 4b. 實心圓潤圖騰 (ported from sentence_glyph_line_demo.smooth_totem) ──────
def _remove_small(mask: np.ndarray, min_size: int = 3) -> np.ndarray:
    """Drop connected components (8-neighbour) smaller than `min_size`."""
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    out = np.zeros_like(mask, bool)
    nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1))
    for y in range(h):
        for x in np.flatnonzero(mask[y] & ~seen[y]):
            stack = [(x, y)]; seen[y, x] = True; pts = []
            while stack:
                px, py = stack.pop(); pts.append((px, py))
                for dx, dy in nbrs:
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True; stack.append((nx, ny))
            if len(pts) >= min_size:
                for px, py in pts:
                    out[py, px] = True
    return out


def _robust_mask(field: np.ndarray, preferred_keep: float = 10) -> np.ndarray:
    """Pick a keep% that yields enough connected ink; fall back gracefully."""
    for keep in (preferred_keep, 14, 18, 24, 32, 45):
        mask = _remove_small(_top_mask(field, keep), 3)
        if int(mask.sum()) >= 18:
            return mask
    mask = _top_mask(field, 45)
    if int(mask.sum()) >= 6:
        return mask
    y, x = np.unravel_index(int(np.argmax(field)), field.shape)
    mask = np.zeros_like(field, bool)
    mask[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = True
    return mask


def _smooth_totem_shape(mask: np.ndarray, size: int) -> Image.Image:
    """Low-res bool mask → big SOLID, ROUNDED, SMOOTH silhouette."""
    small = Image.fromarray(np.uint8(mask * 255), "L")
    shape = small.resize((size, size), Image.Resampling.BICUBIC)
    shape = shape.filter(ImageFilter.GaussianBlur(radius=2.0)).point(lambda p: 255 if p > 70 else 0)
    shape = shape.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))  # morphological close
    return shape


def _lenia_fill(text: str, size: int, steps: int = 130,
                tiles: int = 3) -> np.ndarray | None:
    """Generate the text's Lenia field as a (size,size) luminance map [0,1].

    `tiles` repeats the pattern tiles×tiles across the totem so the cells are
    FINER relative to the attractor's block area (tiles=1 → one big pattern,
    tiles=3 → 3× denser, finer texture).
    """
    if _L is None:
        return None
    try:
        z = _L.text_to_latent(text)
        params = _L.latent_to_params(z)
        sv = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2 ** 31)
        # Run at a STABLE resolution so the organism forms properly …
        A = _L.run_lenia(params, seed=sv, size=140, steps=steps)
        full = Image.fromarray(np.uint8(np.clip(A, 0, 1) * 255), "L")
        if tiles > 1:
            # … then SHRINK the finished pattern and repeat it → fine cells
            tile_px = max(24, int(math.ceil(size / tiles)))
            tile = full.resize((tile_px, tile_px), Image.Resampling.BILINEAR)
            reps = int(math.ceil(size / tile_px))
            arr = np.tile(np.asarray(tile), (reps, reps))[:size, :size]
            img = Image.fromarray(arr, "L")
        else:
            img = full.resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0
    except Exception:
        return None


def render_totem(text: str, size: int = 260, grid: int = 56, keep: float = 10.0,
                 thicken: int = 6, lenia_fill: bool = True, lenia_steps: int = 130,
                 fill_tiles: int = 3, mono: bool = False, binarize: bool = False,
                 fill_floor: float = 0.30, bg=(8, 10, 11)) -> tuple[Image.Image, np.ndarray]:
    """Full pipeline → a bold, solid, smooth totem glyph (not thin lines).

    Uses a LOW-RES density grid so the upscale+blur+threshold fuses the orbit
    into one rounded mass — the look from sentence_glyph_line_demo.

    `thicken`     extra dilation (px) to make the strokes even bolder.
    `lenia_fill`  fill the ink area with the text's own Lenia pattern (textured)
                  instead of a flat colour.
    """
    min_area = int(size * size * 0.04)        # require a reasonably full totem
    shape = None
    v = None
    # try several bank parameter sets until one fills the frame well enough
    for offset in range(len(_PARAM_BANK)):
        pts, v = attractor_points(text, bank_offset=offset)
        field = density_field(normalize(pts, grid, 2), grid)
        for cand in (keep, 14, 18, 24, 32, 45, 62):
            candidate = _smooth_totem_shape(_robust_mask(field, cand), size)
            if int((np.asarray(candidate) > 0).sum()) >= min_area:
                shape = candidate
                break
        if shape is not None:
            break
    if shape is None:                          # last resort
        field = density_field(normalize(attractor_points(text)[0], grid, 2), grid)
        shape = _smooth_totem_shape(_robust_mask(field, 70), size)
    main = (255, 255, 255) if mono else _palette(v)
    if mono:
        bg = (0, 0, 0)

    # extra thickening for an even bolder stroke
    if thicken > 0:
        shape = _fatten(shape, thicken).filter(ImageFilter.GaussianBlur(radius=1.2)) \
                                       .point(lambda p: 255 if p > 90 else 0)

    img = Image.new("RGB", (size, size), bg)
    if not mono:                                          # soft glow (skip in mono)
        glow = shape.filter(ImageFilter.GaussianBlur(radius=6.0 + thicken)).point(lambda p: int(p * 0.18))
        img.paste(Image.new("RGB", (size, size), tuple(int(c * 0.55) for c in main)), mask=glow)

    # ── fill the ink area with the text's own Lenia pattern ───────────────────
    lum = _lenia_fill(text, size, lenia_steps, fill_tiles) if lenia_fill else None
    shape_b = np.asarray(shape) > 127

    if binarize or mono:
        # strict 2-value output: white only where (inside totem) AND (Lenia alive)
        cell = (lum > fill_floor) if lum is not None else np.ones_like(shape_b)
        mask_b = shape_b & cell
        out_mask = Image.fromarray(np.uint8(mask_b * 255), "L")
        img.paste(Image.new("RGB", (size, size), main), mask=out_mask)
    elif lum is not None:
        body = np.empty((size, size, 3), dtype=np.uint8)
        for i, c in enumerate(main):                      # palette tint × Lenia
            body[..., i] = np.uint8(np.clip(c * (0.16 + 0.84 * lum), 0, 255))
        img.paste(Image.fromarray(body, "RGB"), (0, 0), mask=shape)
    else:
        img.paste(Image.new("RGB", (size, size), main), mask=shape)
    return img, field


# ── helpers: density preview + tiling ─────────────────────────────────────────
def _density_preview(field: np.ndarray, main) -> Image.Image:
    g = np.uint8(np.clip(field ** 0.5, 0, 1) * 255)
    base = Image.new("RGB", field.shape[::-1], (8, 10, 11))
    base.paste(Image.new("RGB", field.shape[::-1], main), mask=Image.fromarray(g, "L"))
    return base


def _label(img: Image.Image, text: str) -> Image.Image:
    tile = Image.new("RGB", (img.width, img.height + 44), (13, 15, 17))
    tile.paste(img, (0, 0))
    draw = ImageDraw.Draw(tile)
    try:
        font = ImageFont.truetype("msyh.ttc", 20)          # CJK-capable
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except OSError:
            font = ImageFont.load_default()
    draw.text((14, img.height + 12), text, fill=(220, 226, 230), font=font)
    return tile


def make_sheet(terms: list[str], out: str = "attractor_glyph_test.png",
               size: int = 360) -> str:
    """For each term: density preview + flat glyph, stacked in a grid."""
    pairs = []
    for t in terms:
        glyph, field = render_glyph(t, size=size)
        main = _palette(semantic(t))
        prev = _density_preview(field, main)
        col = Image.new("RGB", (size, size * 2 + 6), (8, 10, 11))
        col.paste(prev, (0, 0))
        col.paste(glyph, (0, size + 6))
        pairs.append(_label(col, t))
        print(f"  rendered: {t}")

    n = len(pairs); gap = 14; tw, th = pairs[0].size
    sheet = Image.new("RGB", (n * tw + (n + 1) * gap, th + 2 * gap + 40), (5, 7, 9))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    draw.text((gap, 12), "Attractor Glyphs  (top: density field · bottom: flat glyph)",
              fill=(235, 238, 240), font=font)
    for i, p in enumerate(pairs):
        sheet.paste(p, (gap + i * (tw + gap), 40 + gap))
    sheet.save(out)
    print(f"saved → {out}")
    return out


if __name__ == "__main__":
    terms = sys.argv[1:] or ["床前明月光", "明月", "quantum mechanics", "cat", "dog"]
    make_sheet(terms)

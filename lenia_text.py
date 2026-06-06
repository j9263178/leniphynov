import numpy as np
import hashlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys

bell = lambda x, m, s: np.exp(-((x - m) / s) ** 2 / 2)

# ── Progress reporting hook ───────────────────────────────────────────────────
# A UI (e.g. the Gradio app) can register a callback to receive status updates.
# Callback signature: cb(fraction: float in [0,1], desc: str).
_PROGRESS_CB = None
def set_progress_cb(cb):
    """Register a progress callback, or None to disable."""
    global _PROGRESS_CB
    _PROGRESS_CB = cb
def _progress(frac, desc):
    cb = _PROGRESS_CB
    if cb is not None:
        try:
            cb(min(max(float(frac), 0.0), 1.0), desc)
        except Exception:
            pass

# ── CJK font ──────────────────────────────────────────────────────────────────
def _setup_font():
    want = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "PingFang SC"]
    have = {f.name for f in fm.fontManager.ttflist}
    for f in want:
        if f in have:
            plt.rcParams["font.family"] = f
            plt.rcParams["axes.unicode_minus"] = False
            return
_setup_font()

# ── Anchors: known-stable Lenia regions, each with a SEMANTIC concept ─────────
# `concept` = multilingual phrase capturing the anchor's visual character.
# A text's latent = how similar its meaning is to each anchor concept, so
# semantically-similar text (across languages) → similar pattern.
ANCHORS = [
    dict(name="chain",   m=0.135, s=0.015, R=20, T=10, b=[1.0],   # 鏈環
         concept="鏈 連結 環扣 序列 chain link connection sequence"),
    dict(name="orbium",  m=0.150, s=0.015, R=26, T=10, b=[1.0],   # glider
         concept="飛 移動 旅行 滑行 奔跑 glider flight journey motion travel"),
    dict(name="blob",    m=0.240, s=0.022, R=24, T=10, b=[1.0],   # 緊密橢圓
         concept="石 堅實 厚重 緊密 凝固 stone solid heavy dense mass"),
    dict(name="scutium", m=0.270, s=0.030, R=16, T=10, b=[1.0],   # 流線迷宮
         concept="迷宮 複雜 交錯 紋理 intricate maze labyrinth complex texture"),
    dict(name="wave",    m=0.200, s=0.035, R=30, T=10, b=[1.0],   # 慢波橢圓
         concept="水 流動 波浪 平靜 河海 water flow wave calm liquid river"),
    dict(name="coral",   m=0.210, s=0.050, R=36, T=10, b=[1.0],   # 大圓圈
         concept="樹 枝 藤 分支 生長 草木 tree branch vine growth plant organic"),
    dict(name="turb",    m=0.170, s=0.045, R=32, T=5,  b=[1.0],   # 中型環
         concept="火 能量 混亂 熱 翻騰 爆 fire energy chaos heat turbulence burn"),
    dict(name="oval",    m=0.220, s=0.020, R=22, T=10, b=[1.0],   # 緊密橢圓群
         concept="群聚 種子 繁多 散布 cluster group seeds scatter multiple"),
]
NA = len(ANCHORS)

# ── Semantic embedding (Gemini) with local cache + hash fallback ──────────────
import os, json, threading
from urllib.request import Request, urlopen

_EMBED_DIM   = 256
_EMBED_MODEL = "gemini-embedding-001"
_CACHE_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embed_cache.json")
_ENV_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

def _load_env():
    if os.path.exists(_ENV_PATH):
        for line in open(_ENV_PATH, encoding="utf-8"):
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
_load_env()
_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

# disk cache: text → 256-d unit vector (list)
_cache_lock = threading.Lock()
try:
    _EMBED_CACHE = json.load(open(_CACHE_PATH, encoding="utf-8"))
except Exception:
    _EMBED_CACHE = {}

_SEMANTIC_OK = bool(_GEMINI_KEY)   # flips to False on first network failure

def _gemini_embed(text: str) -> np.ndarray:
    """Call Gemini embedding API → normalized 256-d vector. Raises on failure."""
    url = (f"https://generativelanguage.googleapis.com/v1beta/"
           f"models/{_EMBED_MODEL}:embedContent?key={_GEMINI_KEY}")
    body = json.dumps({"content": {"parts": [{"text": text}]},
                       "outputDimensionality": _EMBED_DIM}).encode()
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    vals = json.load(urlopen(req, timeout=30))["embedding"]["values"]
    v = np.asarray(vals, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-8)

def _embed(text: str) -> np.ndarray:
    """Cached embedding. Returns None if semantic mode is unavailable."""
    global _SEMANTIC_OK
    if not _SEMANTIC_OK:
        return None
    hit = _EMBED_CACHE.get(text)
    if hit is not None:
        return np.asarray(hit, dtype=np.float64)
    try:
        v = _gemini_embed(text)
    except Exception as e:
        _SEMANTIC_OK = False
        print(f"[semantic] embedding unavailable ({e}); falling back to hash latent")
        return None
    with _cache_lock:
        _EMBED_CACHE[text] = v.tolist()
        try:
            json.dump(_EMBED_CACHE, open(_CACHE_PATH, "w", encoding="utf-8"))
        except Exception:
            pass
    return v

# anchor concept vectors, embedded once (lazily) and cached on disk
_ANCHOR_VECS = None
def _anchor_vecs():
    global _ANCHOR_VECS
    if _ANCHOR_VECS is None:
        vecs = [_embed("[ANCHOR] " + a["concept"]) for a in ANCHORS]
        if any(v is None for v in vecs):
            return None
        _ANCHOR_VECS = np.array(vecs)            # (NA, 256)
    return _ANCHOR_VECS

# ── Hash fallback (original byte-projection latent) ───────────────────────────
_W = np.random.default_rng(42).standard_normal((256, NA))
def _hash_latent(text: str) -> np.ndarray:
    feat = np.zeros(256)
    enc  = text.encode("utf-8")
    for b in enc:
        feat[b] += 1.0
    for i in range(len(enc) - 1):
        feat[(enc[i] * 31 + enc[i + 1]) % 256] += 0.5
    for i, b in enumerate(enc):
        feat[(b + i * 7) % 256] += 0.3
    feat /= max(len(enc), 1)
    z = feat @ _W
    return (z - z.min()) / (z.max() - z.min() + 1e-8)

# ── Text → latent ─────────────────────────────────────────────────────────────
# Semantic: latent[i] = similarity(meaning(text), anchor_concept[i]).
# Cross-lingual by construction (火≈fire). Falls back to hash if API unavailable.
def text_to_latent(text: str) -> np.ndarray:
    v = _embed(text)
    if v is not None:
        C = _anchor_vecs()
        if C is not None:
            z = C @ v                                    # cosine sim → (NA,)
            return (z - z.min()) / (z.max() - z.min() + 1e-8)
    return _hash_latent(text)

# ── Granularity: what counts as one "unit" (one pattern) ──────────────────────
#   "char"  → each character          (一字一 pattern)
#   "sentence" → each clause/sentence (一句一 pattern, split on punctuation)
#   "whole" → the entire input        (整首一個 pattern)
# Each unit's pattern derives from the unit's own semantic embedding, so a
# sentence-unit reflects the meaning of the whole sentence, not just its chars.
import re as _re
_GRANULARITY = "char"
_SENT_SEP    = _re.compile(r"[，。、；：！？\,\.\;\:\!\?\n\r\t ]+")

def set_granularity(g: str):
    global _GRANULARITY
    _GRANULARITY = g if g in ("char", "sentence", "whole") else "char"

def _units(text: str):
    """Split input into the list of pattern-bearing units per current granularity."""
    if _GRANULARITY == "whole":
        t = text.strip()
        return [t] if t else []
    if _GRANULARITY == "sentence":
        return [p for p in (s.strip() for s in _SENT_SEP.split(text)) if p]
    return [c for c in text if c.strip() and c not in "，。、\n"]

# ── Latent → Lenia params (anchor interpolation) ─────────────────────────────
def latent_to_params(z: np.ndarray) -> dict:
    w = np.exp(z * 15)                                 # high temp → clear winner per text
    w /= w.sum()
    m = float(sum(w[i] * ANCHORS[i]["m"] for i in range(NA)))
    s = float(sum(w[i] * ANCHORS[i]["s"] for i in range(NA)))
    R = int(round(sum(w[i] * ANCHORS[i]["R"] for i in range(NA))))
    T = int(round(sum(w[i] * ANCHORS[i]["T"] for i in range(NA))))
    b = ANCHORS[int(np.argmax(w))]["b"]                # dominant anchor's kernel shape
    dominant = ANCHORS[int(np.argmax(w))]["name"]
    return dict(m=m, s=s, R=R, T=T, b=b, dominant=dominant, weights=w)

# ── Lenia ─────────────────────────────────────────────────────────────────────
def run_lenia(params: dict, seed: int, size: int = 64, steps: int = 300,
              revive: bool = True, max_tries: int = 8) -> np.ndarray:
    """Evolve a Lenia field.

    Narrow-growth parameter sets (small s) sometimes land the random initial
    condition in a "death basin" → the whole field decays to 0 → a solid black
    pattern.  When `revive` is on, we deterministically retry with alternate
    seeds (same order every time, so a given char still maps to one stable
    pattern) until the organism survives.
    """
    m, s, R, T, b = params["m"], params["s"], params["R"], params["T"], params["b"]
    b   = np.asarray(b)
    mid = size // 2
    # kernel is seed-independent → build its FFT once
    D  = np.linalg.norm(np.mgrid[-mid:mid, -mid:mid], axis=0) / R * len(b)
    K  = (D < len(b)) * b[np.minimum(D.astype(int), len(b) - 1)] * bell(D % 1, 0.5, 0.15)
    fK = np.fft.fft2(np.fft.fftshift(K / np.sum(K)))

    A = None
    for attempt in range(max_tries if revive else 1):
        sv = seed if attempt == 0 else (seed * 1000003 + attempt * 7919 + 1) % (2 ** 31)
        np.random.seed(sv)
        A = np.random.rand(size, size)
        for _ in range(steps):
            U = np.real(np.fft.ifft2(fK * np.fft.fft2(A)))
            A = np.clip(A + 1 / T * (bell(U, m, s) * 2 - 1), 0, 1)
        if not revive or float(A.max()) > 1e-3:   # alive → done
            return A
    return A                                        # gave up; return last attempt

# ── Display post-processing: threshold + single color ────────────────────────
COLOR = (0.96, 0.95, 0.90)  # ← 改這裡換顏色 (R, G, B) in [0,1]

def colorize(A, floor=0.15):
    """Binary: pixels above floor → COLOR, below → black. No gradients."""
    mask = (A > floor).astype(float)
    rgb = np.zeros((*A.shape, 3))
    for i, c in enumerate(COLOR):
        rgb[:, :, i] = mask * c
    return rgb


def _scaled_pattern(A, size, psize):
    """Colorise a Lenia field, optionally magnified so its tile spans `psize`
    canvas pixels (bigger pattern features, longer tiling period)."""
    if psize != size:
        from scipy.ndimage import zoom as _zoom
        return colorize(_zoom(A, psize / size, order=1))
    return colorize(A)

# ── Show anchor gallery ───────────────────────────────────────────────────────
def show_anchors(size=64, steps=300, out="lenia_anchors.png"):
    cols = 4
    rows = (NA + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.8))
    fig.patch.set_facecolor("#000000")
    axes = np.array(axes).flatten()

    for i, (ax, anc) in enumerate(zip(axes, ANCHORS)):
        A = run_lenia(anc, seed=i, size=size, steps=steps)
        ax.imshow(colorize(A), vmin=0, vmax=1, interpolation="bilinear")
        ax.set_title(f"#{i}  {anc['name']}", color="white", fontsize=12, fontweight="bold")
        ax.set_xlabel(
            f"m={anc['m']}  s={anc['s']}  R={anc['R']}\nb={anc['b']}",
            color="#777777", fontsize=8
        )
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for ax in axes[NA:]:
        ax.set_visible(False)

    plt.suptitle("Lenia Anchors  —  known-stable parameter regions", color="white", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="#0d0d0d")
    plt.show()
    print(f"saved → {out}")

# ── Show text gallery ─────────────────────────────────────────────────────────
def show_texts(texts, size=64, steps=300, cols=4, out="lenia_text.png"):
    rows = (len(texts) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 4.4))
    fig.patch.set_facecolor("#000000")
    axes = np.array(axes).flatten()

    for ax, text in zip(axes, texts):
        z      = text_to_latent(text)
        params = latent_to_params(z)
        seed   = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2 ** 31)
        A      = run_lenia(params, seed=seed, size=size, steps=steps)

        ax.imshow(colorize(A), vmin=0, vmax=1, interpolation="bilinear")
        ax.set_title(text, color="white", fontsize=16, fontweight="bold", pad=6)
        ax.set_xlabel(
            f"→ anchor: {params['dominant']}\nm={params['m']:.3f}  s={params['s']:.3f}  R={params['R']}",
            color="#666666", fontsize=8
        )
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for ax in axes[len(texts):]:
        ax.set_visible(False)

    plt.suptitle("Text → Latent Space → Lenia", color="white", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="#0d0d0d")
    plt.show()
    print(f"saved → {out}")

# ── Show poem: multi-line character grid ──────────────────────────────────────
def show_poem(text, line_length=5, size=56, steps=300, out="lenia_poem.png"):
    # strip punctuation, split into lines
    clean  = _units(text)
    lines  = [clean[i:i+line_length] for i in range(0, len(clean), line_length)]
    all_ch = list(dict.fromkeys(clean))          # unique chars, preserve order

    print(f"generating {len(all_ch)} unique characters …")
    patterns = {}
    for ch in all_ch:
        z      = text_to_latent(ch)
        params = latent_to_params(z)
        seed   = int(hashlib.md5(ch.encode()).hexdigest(), 16) % (2 ** 31)
        A      = run_lenia(params, seed=seed, size=size, steps=steps)
        patterns[ch] = colorize(A)

    gap_x = np.zeros((size, 5, 3))              # gap between chars
    gap_y = np.zeros((10, 1, 3))                # gap between lines (width filled later)

    row_imgs = []
    for line in lines:
        strips = []
        for i, ch in enumerate(line):
            strips.append(patterns[ch])
            if i < len(line) - 1:
                strips.append(gap_x)
        row_imgs.append(np.concatenate(strips, axis=1))

    # pad shorter rows to same width
    max_w = max(r.shape[1] for r in row_imgs)
    canvas_rows = []
    for r in row_imgs:
        if r.shape[1] < max_w:
            r = np.concatenate([r, np.zeros((size, max_w - r.shape[1], 3))], axis=1)
        canvas_rows.append(r)
        canvas_rows.append(np.zeros((10, max_w, 3)))   # line gap
    canvas = np.concatenate(canvas_rows[:-1], axis=0)

    # figure sized to canvas
    dpi   = 110
    fig_w = canvas.shape[1] / dpi
    fig_h = canvas.shape[0] / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h + 0.6))
    fig.patch.set_facecolor("#080808")
    ax.set_facecolor("#080808")
    ax.imshow(canvas, interpolation="bilinear")
    ax.axis("off")

    # character labels: one per tile, aligned to grid
    tile_w = size + gap_x.shape[1]
    tile_h = size + 10
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            px = c * tile_w + size / 2
            py = r * tile_h + size + 2
            ax.text(px, py, ch,
                    color=(0.96, 0.95, 0.90), fontsize=7,
                    ha="center", va="top", alpha=0.4)

    plt.tight_layout(pad=0.2)
    plt.savefig(out, dpi=dpi*2, bbox_inches="tight", facecolor="#000000")
    plt.show()
    print(f"saved → {out}")

# ── Global Physarum helper ────────────────────────────────────────────────────
def _global_physarum(W, H, n_steps=200, sa_deg=38.0, ra_deg=45.0, so=9,
                     dep=5.0, decay=0.91, diffuse_sigma=1.0, seed=0):
    """
    Run one Physarum simulation over the full W×H canvas.
    Returns a normalised trail map [0, 1].
    Uses wrap boundaries so the network covers the canvas uniformly.
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(int(seed) % (2**32))
    n_agents = min(max(W * H // 8, 4000), 80000)   # reasonable range
    T  = np.zeros((H, W), np.float32)
    ax = rng.uniform(0, W, n_agents).astype(np.float32)
    ay = rng.uniform(0, H, n_agents).astype(np.float32)
    aa = rng.uniform(0, 2 * np.pi, n_agents).astype(np.float32)
    SA = np.radians(sa_deg); RA = np.radians(ra_deg); SO = float(so)

    for _ in range(n_steps):
        def sense(off):
            sx = (ax + np.cos(aa + off) * SO).astype(int) % W
            sy = (ay + np.sin(aa + off) * SO).astype(int) % H
            return T[sy, sx]
        FL = sense(-SA); F = sense(0.); FR = sense(SA)
        tl=(F<FL)&(FL>=FR); tr=(F<FR)&(FR>FL); tr2=(FL==FR)&(FL>F)
        rs = rng.choice(np.array([-1., 1.], np.float32), n_agents)
        aa = np.where(tl, aa-RA, aa)
        aa = np.where(tr, aa+RA, aa)
        aa = np.where(tr2, aa+rs*RA, aa)
        ax = (ax + np.cos(aa)) % W
        ay = (ay + np.sin(aa)) % H
        np.add.at(T, (ay.astype(int) % H, ax.astype(int) % W), dep)
        T = gaussian_filter(T, sigma=diffuse_sigma, mode='wrap') * decay

    return T / (T.max() + 1e-8)


# ── Mini Physarum helper ──────────────────────────────────────────────────────
def _mini_physarum(region_mask, n_steps=100, seed=0, thresh_pct=82, dilate_px=3,
                   sa_deg=38.0, ra_deg=45.0, so=None):
    """
    Run a small Physarum simulation seeded inside a boolean region mask.
    Agents are free to wander up to ~1.5× the region bounding box — no bouncing —
    so trails can organically spill slightly outside, creating a sense of overgrowth.
    Returns a boolean mask of trail pixels (where to carve black), clipped to canvas.
    """
    from scipy.ndimage import gaussian_filter, binary_dilation
    H, W = region_mask.shape
    ys, xs = np.where(region_mask)
    if len(ys) < 30:
        return np.zeros_like(region_mask)

    rng      = np.random.default_rng(int(seed) % (2**32))
    # fewer agents so not every pixel gets a hole
    n_agents = max(20, len(ys) // 18)
    idx      = rng.integers(0, len(ys), n_agents)
    ax = xs[idx].astype(np.float32)
    ay = ys[idx].astype(np.float32)
    aa = rng.uniform(0, 2 * np.pi, n_agents).astype(np.float32)

    SA = np.radians(sa_deg)
    SO = float(so if so is not None else max(3, min(W, H) // 14))
    RA = np.radians(ra_deg)
    T  = np.zeros((H, W), np.float32)

    # allow agents to roam into ~1.5× the region's own bounding box
    h_span = max(ys.max() - ys.min(), 1)
    w_span = max(xs.max() - xs.min(), 1)
    y0_bb = max(0,     ys.min() - h_span // 2)
    y1_bb = min(H - 1, ys.max() + h_span // 2)
    x0_bb = max(0,     xs.min() - w_span // 2)
    x1_bb = min(W - 1, xs.max() + w_span // 2)

    for _ in range(n_steps):
        def sense(off):
            sx = np.clip((ax + np.cos(aa + off) * SO).astype(int), 0, W - 1)
            sy = np.clip((ay + np.sin(aa + off) * SO).astype(int), 0, H - 1)
            return T[sy, sx]

        FL = sense(-SA); F = sense(0.); FR = sense(SA)
        tl = (F < FL) & (FL >= FR)
        tr = (F < FR) & (FR >  FL)
        tr2= (FL == FR) & (FL > F)
        rs = rng.choice(np.array([-1., 1.], np.float32), n_agents)
        aa = np.where(tl,  aa - RA, aa)
        aa = np.where(tr,  aa + RA, aa)
        aa = np.where(tr2, aa + rs * RA, aa)

        # move freely — clip only to canvas edge, no region bounce
        ax = np.clip(ax + np.cos(aa), x0_bb, x1_bb).astype(np.float32)
        ay = np.clip(ay + np.sin(aa), y0_bb, y1_bb).astype(np.float32)

        ix = ax.astype(int); iy = ay.astype(int)
        np.add.at(T, (iy, ix), 5.0)
        T = gaussian_filter(T, sigma=0.8, mode='constant') * 0.91

    if T.max() < 1e-8:
        return np.zeros_like(region_mask)

    T_norm = T / T.max()
    region_vals = T_norm[region_mask]
    thresh = np.percentile(region_vals, thresh_pct) if len(region_vals) else 0.5
    trail  = binary_dilation(T_norm > thresh, iterations=dilate_px)
    # apply only within the canvas, not restricted to region boundary
    return trail & (np.ones_like(region_mask))


# ── Warped Voronoi helper ─────────────────────────────────────────────────────
def _warped_voronoi_assign(wx, wy, sub_xy, z_g, scale):
    """
    Nearest-seed assignment in smoothly-warped coordinate space.

    Instead of a straight perpendicular bisector between seeds, each pixel's
    coordinates are nudged by a smooth sine field (derived from z_g) before the
    distance is computed.  This curves the Voronoi boundaries organically while
    staying deterministic and character-specific.

    Parameters
    ----------
    wx, wy   : 1-D int/float arrays of pixel x and y coordinates
    sub_xy   : (n_chars, 2) float array of seed points [[x, y], ...]
    z_g      : group latent vector (8-dim, values in [0, 1])
    scale    : characteristic size of the region in pixels.
               Warp wavelength = 2.5 × scale  (gentle curves)
               Warp amplitude  = 0.10 × scale  (subtle displacement)

    Returns
    -------
    assign : 1-D int array, same length as wx/wy, value = seed index
    """
    sc  = 2 * np.pi / (scale * 2.5)   # spatial frequency → long wavelength = gentle
    amp = scale * 0.10                 # 10 % of block size → subtle
    wx_w = wx.astype(float)
    wy_w = wy.astype(float)
    for k in range(3):                 # 3 octaves: smooth, not noisy
        theta = float(z_g[k % 8])     * 2 * np.pi + k * 1.4
        phi   = float(z_g[(k+4) % 8]) * 2 * np.pi
        wave  = np.sin(np.cos(theta) * sc * wx + np.sin(theta) * sc * wy + phi)
        wx_w  = wx_w + amp * wave * float(np.cos(theta + np.pi / 2))
        wy_w  = wy_w + amp * wave * float(np.sin(theta + np.pi / 2))
    d2 = ((wy_w[:, None] - sub_xy[None, :, 1])**2 +
          (wx_w[:, None] - sub_xy[None, :, 0])**2)
    return d2.argmin(axis=1)


# ── Polygon layout ────────────────────────────────────────────────────────────
def _char_to_curve(z: np.ndarray, size: int, resolution: int = 320) -> np.ndarray:
    """
    Latent vector z (len 8) → smooth closed organic curve.
    Polar control points are B-spline interpolated (periodic) so the outline
    is continuously smooth — no corners.  Shape, radii, and global rotation
    are all deterministically derived from z.
    Returns (resolution, 2) array of (x, y) pixel coords.
    """
    from scipy.interpolate import splprep, splev

    n    = 7 + int(z[0] * 3.9999)   # 7–10 control points → varying complexity
    half = size / 2.0

    base    = np.linspace(0, 2 * np.pi, n, endpoint=False)
    max_jit = np.pi / n              # ≤ one full gap → allows concave dips
    perturb = np.array([(z[(i + 1) % 8] - 0.5) * 2 * max_jit for i in range(n)])
    angles  = (base + perturb) % (2 * np.pi)    # unsorted → organic non-convex
    angles += z[7] * 2 * np.pi                  # global rotation (full 360°)

    # Wide radius variation for blobby, amoeba-like silhouettes
    radii = np.array([(0.22 + z[(i + 4) % 8] * 0.26) * size for i in range(n)])

    cx = half + (z[2] - 0.5) * half * 0.22
    cy = half + (z[3] - 0.5) * half * 0.22

    ctrl_x = cx + radii * np.cos(angles)
    ctrl_y = cy + radii * np.sin(angles)

    # Periodic cubic B-spline through all control points
    tck, _ = splprep([ctrl_x, ctrl_y], s=0, per=True, k=3)
    u = np.linspace(0, 1, resolution, endpoint=False)
    xs, ys = splev(u, tck)

    return np.stack([xs, ys], axis=1)   # (resolution, 2)


def show_poem_poly(text, size=84, steps=300, cols=5, row_shift=0.5, out="lenia_poly.png"):
    """
    Each character → unique irregular 3-5 sided polygon whose geometry is
    fully derived from its latent vector.  The polygon is filled with the
    character's Lenia pattern; a thin light edge traces the silhouette.
    """
    from matplotlib.path import Path as MplPath
    import matplotlib.patches as mpatches

    clean  = _units(text)
    unique = list(dict.fromkeys(clean))
    n      = len(clean)

    print(f"generating {len(unique)} polygon-pattern pairs …")
    patt = {}   # ch → masked RGB array (size, size, 3)
    vmap = {}   # ch → vertices (n_sides, 2)

    for ch in unique:
        z      = text_to_latent(ch)
        params = latent_to_params(z)
        seed   = int(hashlib.md5(ch.encode()).hexdigest(), 16) % (2 ** 31)
        A      = run_lenia(params, seed=seed, size=size, steps=steps)
        rgb    = colorize(A)

        verts    = _char_to_curve(z, size)
        vmap[ch] = verts

        # Pixel mask: inside polygon → Lenia, outside → black
        path   = MplPath(verts)
        yy, xx = np.mgrid[0:size, 0:size]
        inside = path.contains_points(
                     np.stack([xx.ravel(), yy.ravel()], axis=1)
                 ).reshape(size, size)
        patt[ch] = rgb * inside[:, :, None]

    # ── centroid-anchored tight packing ──────────────────────────────────────
    # Use actual polygon bounding radius so cell spacing matches polygon size,
    # not the (larger) bounding box.
    centroids = {ch: vmap[ch].mean(axis=0) for ch in unique}
    brad      = {ch: float(np.max(np.linalg.norm(vmap[ch] - centroids[ch], axis=1)))
                 for ch in unique}
    r_max = max(brad.values())

    cell  = int(2 * r_max + 7)         # just enough room between polygon edges
    jit   = int(r_max * 0.24)          # ±24 % of r_max → mild chaotic scatter
    rows  = (n + cols - 1) // cols
    pad   = int(r_max * 1.3)
    W     = cols * cell + 2 * pad
    H     = rows * cell + 2 * pad

    max_row_shift = int(cell * 0.4 * row_shift)
    row_x_offs = {}
    for r in range(rows):
        row_chars = clean[r * cols: (r + 1) * cols]
        row_avg_z = np.mean([text_to_latent(ch) for ch in row_chars], axis=0)
        row_x_offs[r] = int((float(row_avg_z[0]) - 0.5) * 2 * max_row_shift)

    positions = []
    for idx, ch in enumerate(clean):
        z    = text_to_latent(ch)
        # grid centre in world coords
        gcx  = pad + (idx % cols) * cell + cell // 2 + row_x_offs[idx // cols]
        gcy  = pad + (idx // cols) * cell + cell // 2
        # jitter driven by z[5], z[6]
        wcx  = gcx + int((z[5] - 0.5) * 2 * jit)
        wcy  = gcy + int((z[6] - 0.5) * 2 * jit)
        # convert world centroid → top-left of (size×size) patch
        lcx, lcy = centroids[ch]
        ox   = int(np.clip(wcx - lcx, 0, W - size))
        oy   = int(np.clip(wcy - lcy, 0, H - size))
        positions.append((ox, oy))

    # ── paint onto numpy canvas (painters' order = poem order) ───────────────
    canvas = np.zeros((H, W, 3))
    for idx, ch in enumerate(clean):
        ox, oy = positions[idx]
        img    = patt[ch]
        x0, x1 = max(0, ox), min(W, ox + size)
        y0, y1 = max(0, oy), min(H, oy + size)
        ix0, ix1 = x0 - ox, x1 - ox
        iy0, iy1 = y0 - oy, y1 - oy
        if x1 > x0 and y1 > y0:
            patch  = img[iy0:iy1, ix0:ix1]
            region = canvas[y0:y1, x0:x1]
            active = np.any(patch > 0, axis=2)
            region[active] = patch[active]
            canvas[y0:y1, x0:x1] = region

    # ── render + polygon edge overlays ────────────────────────────────────────
    dpi  = 110
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi))
    fig.patch.set_facecolor("#080808")
    ax.set_facecolor("#080808")
    ax.imshow(canvas, interpolation="bilinear")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    for idx, ch in enumerate(clean):
        ox, oy = positions[idx]
        world = vmap[ch] + np.array([ox, oy])
        ax.add_patch(mpatches.Polygon(
            world, closed=True, fill=False,
            edgecolor=(0.96, 0.95, 0.90), linewidth=0.9, alpha=0.55, zorder=5))

    plt.tight_layout(pad=0.1)
    plt.savefig(out, dpi=dpi*2, bbox_inches="tight", facecolor="#000000")
    plt.show()
    print(f"saved → {out}")


# ── Shared building blocks for grouped / waves / mixed ────────────────────────
# These are the single source of truth for the blob & wave pipelines so that
# show_poem_grouped, show_poem_waves and show_poem_mixed all behave identically
# (no more divergent inline copies).

def _prepare_patterns(text, size, steps, group_size, want_raw=False,
                      tag="poem", pat_hi=0.85, pat_scale=1.0):
    """units → groups → embeddings → Lenia patterns (+ optional raw fields).

    Returns (clean, unique, groups, n_g, zmap, patterns, raw_A, g_z, psize).
    `pat_scale` magnifies each Lenia pattern so its tile spans more canvas
    pixels (bigger, less dense) without touching the page/layout geometry.
    `psize` is the side length of the (possibly magnified) pattern images and
    is what callers must use as the modulo when sampling `patterns[ch]`.
    Reports progress for the embedding (①) and pattern-evolution (②) phases.
    """
    clean  = _units(text)
    unique = list(dict.fromkeys(clean))
    n      = len(clean)
    groups = [clean[i:i + group_size] for i in range(0, n, group_size)]
    n_g    = len(groups)

    pat_scale = max(1.0, float(pat_scale))
    psize     = max(1, int(round(size * pat_scale)))

    print(f"[{tag}] {n} chars → {n_g} groups, generating {len(unique)} patterns "
          f"(scale ×{pat_scale:.2f} → {psize}px tiles) …")
    nU = len(unique)
    zmap = {}
    for i, ch in enumerate(unique):
        _progress(0.15 * i / max(nU, 1),
                  f"① 取得語意向量 (embedding API) {i+1}/{nU}")
        zmap[ch] = text_to_latent(ch)

    patterns, raw_A = {}, {}
    for i, ch in enumerate(unique):
        _progress(0.15 + (pat_hi - 0.15) * i / max(nU, 1),
                  f"② 演化 Lenia pattern {i+1}/{nU}")
        params = latent_to_params(zmap[ch])
        sv = int(hashlib.md5(ch.encode()).hexdigest(), 16) % (2 ** 31)
        A = run_lenia(params, seed=sv, size=size, steps=steps)
        if psize != size:                              # smoothly magnify field
            from scipy.ndimage import zoom as _zoom
            A_big = _zoom(A, psize / size, order=1)
            patterns[ch] = colorize(A_big)
        else:
            patterns[ch] = colorize(A)
        if want_raw:
            raw_A[ch] = A                              # wave edges stay at `size`

    g_z = {g: np.mean([zmap[ch] for ch in grp], axis=0)
           for g, grp in enumerate(groups)}
    return clean, unique, groups, n_g, zmap, patterns, raw_A, g_z, psize


def _lenia_wave(sig, target_len, smooth=3.0):
    """Smooth, normalise, and interpolate a 1-D Lenia field slice → wave offsets."""
    from scipy.ndimage import gaussian_filter1d
    s = gaussian_filter1d(np.asarray(sig, dtype=float), sigma=smooth)
    mu, sd = s.mean(), s.std() + 1e-8
    norm = (s - mu) / sd
    return np.interp(np.linspace(0, len(norm) - 1, target_len),
                     np.arange(len(norm)), norm)


def _fill_wave_voronoi(canvas, char_idx_c, patterns, zmap, size, row_h,
                       grp, wy_abs, wx, cx_b, cy_b, w_g, base_idx, z_g):
    """Wave-block sub-character fill (single source for waves & mixed)."""
    if not len(wy_abs):
        return
    if len(grp) == 1:
        canvas[wy_abs, wx]     = patterns[grp[0]][wy_abs % size, wx % size]
        char_idx_c[wy_abs, wx] = base_idx
        return
    cr = min(w_g, row_h) * 0.28
    sub_xy = np.array([
        [cx_b + cr * np.cos(float(zmap[ch][1]) * 2 * np.pi + k * 2 * np.pi / len(grp)),
         cy_b + cr * np.sin(float(zmap[ch][1]) * 2 * np.pi + k * 2 * np.pi / len(grp))]
        for k, ch in enumerate(grp)])
    asgn = _warped_voronoi_assign(wx, wy_abs, sub_xy, z_g, w_g)
    for k, ch in enumerate(grp):
        m = asgn == k
        canvas[wy_abs[m], wx[m]]     = patterns[ch][wy_abs[m] % size, wx[m] % size]
        char_idx_c[wy_abs[m], wx[m]] = base_idx + k


def _fill_blob_voronoi(canvas, char_idx_c, patterns, zmap, size,
                       grp, wy_l, wx_l, wy_w, wx_w, base_idx, z_g):
    """Blob sub-character fill (single source for grouped & mixed)."""
    if not len(wy_w):
        return
    if len(grp) == 1:
        canvas[wy_w, wx_w]     = patterns[grp[0]][wy_w % size, wx_w % size]
        char_idx_c[wy_w, wx_w] = base_idx
        return
    cx, cy = float(wx_l.mean()), float(wy_l.mean())
    cr = float(max(np.std(wx_l), np.std(wy_l), 1.0))
    sub_xy = []
    for k, ch in enumerate(grp):
        z = zmap[ch]
        angle = float(z[1]) * 2 * np.pi + k * 2 * np.pi / len(grp)
        r = cr * (0.32 + float(z[3]) * 0.16)
        sub_xy.append([cx + r * np.cos(angle), cy + r * np.sin(angle)])
    sub_xy = np.array(sub_xy)
    scale = float(max(np.std(wx_l), np.std(wy_l), 1.0)) * 2
    asgn = _warped_voronoi_assign(wx_l, wy_l, sub_xy, z_g, scale)
    for k, ch in enumerate(grp):
        m = asgn == k
        canvas[wy_w[m], wx_w[m]]     = patterns[ch][wy_w[m] % size, wx_w[m] % size]
        char_idx_c[wy_w[m], wx_w[m]] = base_idx + k


def _blob_mask(verts, bsz, z_g):
    """Rasterise a blob polygon → (bsz,bsz) bool mask, with optional punched hole.
    Single source for grouped & mixed blob shapes."""
    from matplotlib.path import Path as MplPath
    yy_l, xx_l = np.mgrid[0:bsz, 0:bsz]
    bmask = MplPath(verts).contains_points(
        np.stack([xx_l.ravel(), yy_l.ravel()], axis=1).astype(np.float32)
    ).reshape(bsz, bsz)
    if float(z_g[4]) > 0.60:                       # ~40% of blobs get one hole
        a    = float(z_g[0]) * 2 * np.pi
        dist = bsz * (0.06 + float(z_g[2]) * 0.22)
        rad  = bsz * (0.05 + float(z_g[7]) * 0.16)
        cx_h = bsz / 2 + dist * np.cos(a)
        cy_h = bsz / 2 + dist * np.sin(a)
        # ── wobbly hole: radius undulates with angle → petal / amoeba shape ────
        lobes = 3 + int(float(z_g[1]) * 3.999)     # 3–6 lobes
        amp   = 0.22 + float(z_g[5]) * 0.38        # wobble depth [0.22, 0.60]
        phase = float(z_g[6]) * 2 * np.pi          # rotation
        dxh   = xx_l - cx_h
        dyh   = yy_l - cy_h
        ang   = np.arctan2(dyh, dxh)
        r_th  = rad * (1.0 + amp * np.sin(lobes * ang + phase))
        bmask[(dxh ** 2 + dyh ** 2) < r_th ** 2] = False
    return bmask


def _draw_group_boundary(canvas, char_idx_c, group_size, mask_before=False):
    """Black hairline between groups only (not chars within a group)."""
    from scipy.ndimage import binary_dilation
    grp_idx_c = np.where(char_idx_c >= 0, char_idx_c // group_size, -1)
    vc = grp_idx_c[1:, :] != grp_idx_c[:-1, :]
    hc = grp_idx_c[:, 1:] != grp_idx_c[:, :-1]
    bm = np.zeros(char_idx_c.shape, bool)
    bm[1:, :] |= vc; bm[:-1, :] |= vc; bm[:, 1:] |= hc; bm[:, :-1] |= hc
    if mask_before:                                # grouped variant
        bm &= (char_idx_c >= 0)
        canvas[binary_dilation(bm, iterations=1)] = 0.0
    else:                                          # waves / mixed variant
        canvas[binary_dilation(bm, iterations=1) & (char_idx_c >= 0)] = 0.0


def _canvas_rgba(canvas, floor=0.02):
    """(H,W,3) float canvas → (H,W,4) RGBA: pattern pixels opaque, the empty
    (black) background → fully transparent alpha."""
    rgb   = np.clip(canvas, 0.0, 1.0)
    alpha = (rgb.max(axis=2) > floor).astype(np.float32)
    return np.dstack([rgb, alpha])


def _render_save(canvas, W, H, out, pad=0.0):
    """Shared figure render + save (transparent bg, 2× dpi)."""
    dpi = 110
    rgba = _canvas_rgba(canvas)
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi))
    fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
    ax.imshow(rgba, interpolation="bilinear")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    plt.tight_layout(pad=pad)
    plt.savefig(out, dpi=dpi * 2, bbox_inches="tight", transparent=True)
    plt.show();  print(f"saved → {out}")


# ── Wave-boundary row layout ──────────────────────────────────────────────────
def show_poem_waves(text, size=160, steps=300, cols=5, group_size=1, row_shift=0.5, pat_scale=1.0, out="lenia_waves.png"):
    """
    Characters grouped into blocks, laid out in rows.
    Each block has flat top/bottom (row-aligned) and wavy left/right edges.
    Within each block, characters are sub-divided via mini-Voronoi.
    Blocks can have at most one circular hole/notch (derived from group z).
    Visible gap between blocks; rows separated by a black horizontal strip.
    """
    (clean, unique, groups, n_g, zmap, patterns,
     raw_A, g_z, psize) = _prepare_patterns(text, size, steps, group_size,
                                            want_raw=True, tag="waves",
                                            pat_scale=pat_scale)
    _progress(0.85, "③ 排版與分組填色")

    rows    = (n_g + cols - 1) // cols
    block_w = int(size * max(group_size * 0.65, 0.85))  # thin per char
    gap_bw  = int(size * 0.32)
    row_h   = int(size * 1.60)      # tall rows
    gap_h   = max(5, int(size * 0.40))
    pad     = int(size * 0.45)

    W = cols * block_w + (cols - 1) * gap_bw + 2 * pad
    H = rows * (row_h + gap_h) - gap_h

    canvas     = np.zeros((H, W, 3))
    char_idx_c = np.full((H, W), -1, np.int32)
    xs_all     = np.arange(W)

    max_row_shift = int(gap_bw * row_shift)

    for r in range(rows):
        y0         = r * (row_h + gap_h)
        y1         = y0 + row_h
        row_g_idxs = list(range(r * cols, min((r + 1) * cols, n_g)))

        row_avg_z  = np.mean([g_z[g] for g in row_g_idxs], axis=0)
        row_x_off  = int((float(row_avg_z[0]) - 0.5) * 2 * max_row_shift)

        for li, g in enumerate(row_g_idxs):
            grp  = groups[g]
            z_g  = g_z[g]
            # per-block width [0.70, 1.30] × block_w and x offset within cell
            w_g   = int(block_w * (0.40 + float(z_g[4]) * 1.20))
            x_off = int(gap_bw  * (float(z_g[6]) - 0.5)  * 0.60)
            cell_cx = pad + li * (block_w + gap_bw) + block_w // 2 + row_x_off
            blx  = cell_cx - w_g // 2 + x_off
            brx  = blx + w_g

            # ── Lenia-derived wavy edges ──────────────────────────────────────
            avg_A = np.mean([raw_A[ch] for ch in grp], axis=0)   # (size, size)
            lw = np.full(row_h, float(blx))                      # straight left
            # Right boundary: Lenia-derived wave (z_g[3] → wavelength)
            smooth_r = 1.0 + float(z_g[3]) * 9.0          # [1, 10]
            sig_r    = avg_A[:, size // 2:].mean(axis=1)
            wv_r     = _lenia_wave(sig_r, row_h, smooth=smooth_r)
            amp_r    = w_g * (0.06 + float(z_g[5]) * 0.07)
            rw       = brx - amp_r * wv_r

            # ── block pixel mask ─────────────────────────────────────────────
            lw2 = lw[:, None];  rw2 = rw[:, None]
            bmask = (xs_all[None, :] >= lw2) & (xs_all[None, :] < rw2)  # (row_h, W)
            iy_rel, wx = np.where(bmask)
            wy_abs = iy_rel + y0
            if not len(wy_abs):
                continue

            cx_b = float(blx + brx) / 2
            cy_b = float(y0  + y1)  / 2
            _fill_wave_voronoi(canvas, char_idx_c, patterns, zmap, psize, row_h,
                               grp, wy_abs, wx, cx_b, cy_b, w_g,
                               g * group_size, z_g)

    # group-to-group seam removed (groups now merge seamlessly where they touch)
    # _draw_group_boundary(canvas, char_idx_c, group_size)
    _progress(0.95, "④ 輸出影像")
    _render_save(canvas, W, H, out, pad=0)
    _progress(1.0, "完成")


# ── Mixed: wave + blob per group ──────────────────────────────────────────────
def show_poem_mixed(text, size=160, steps=300, cols=5, group_size=1, row_shift=0.5,
                    germ=False, germ_steps=100,
                    germ_sa=38.0, germ_ra=45.0, germ_so=9, pat_scale=1.0,
                    out="lenia_mixed.png"):
    """
    Waves layout as base; groups with z_g[2] > 0.5 render as grouped blobs.
    Both styles use their respective algorithms faithfully:
      WAVE → same algorithm as show_poem_waves
      BLOB → same algorithm as show_poem_grouped, centred on the grid cell
    """
    (clean, unique, groups, n_g, zmap, patterns,
     raw_A, g_z, psize) = _prepare_patterns(text, size, steps, group_size,
                                            want_raw=True, tag="mixed",
                                            pat_hi=0.80, pat_scale=pat_scale)
    _progress(0.80, "③ 排版與分組填色")

    # ── shared grid layout (same as show_poem_waves) ──────────────────────────
    rows    = (n_g + cols - 1) // cols
    block_w = int(size * max(group_size * 0.65, 0.85))
    gap_bw  = int(size * 0.32)
    row_h   = int(size * 1.60)
    gap_h   = max(5, int(size * 0.40))
    pad     = int(size * 0.45)
    W = cols * block_w + (cols - 1) * gap_bw + 2 * pad
    H = rows * (row_h + gap_h) - gap_h

    canvas     = np.zeros((H, W, 3))
    char_idx_c = np.full((H, W), -1, np.int32)
    xs_all     = np.arange(W)

    max_row_shift = int(gap_bw * row_shift)

    for r in range(rows):
        y0 = r * (row_h + gap_h);  y1 = y0 + row_h
        row_gs = list(range(r * cols, min((r+1)*cols, n_g)))

        row_avg_z = np.mean([g_z[g] for g in row_gs], axis=0)
        row_x_off = int((float(row_avg_z[0]) - 0.5) * 2 * max_row_shift)

        for li, g in enumerate(row_gs):
            grp = groups[g];  z_g = g_z[g]
            w_g     = int(block_w * (0.40 + float(z_g[4]) * 0.75))
            x_off   = int(block_w * (float(z_g[6]) - 0.5) * 1.20)
            cell_cx = pad + li * (block_w + gap_bw) + block_w // 2 + row_x_off
            blx = cell_cx - w_g//2 + x_off
            brx = blx + w_g
            cy_b = (y0 + y1) / 2.0
            base = g * group_size

            if float(z_g[2]) > 0.75:
                # ── BLOB (shared show_poem_grouped pipeline) ──────────────────
                bsz   = int(size * (1.0 + float(z_g[4]) * 0.7))
                verts = _char_to_curve(z_g, bsz)
                centroid  = verts.mean(axis=0)
                blob_xoff = int(block_w * (float(z_g[5]) - 0.5) * 0.80)
                ox = int(np.clip(cell_cx - centroid[0] + blob_xoff, 0, W - bsz))
                oy = int(np.clip(cy_b    - centroid[1],          0, H - bsz))

                bmask = _blob_mask(verts, bsz, z_g)
                wy_l, wx_l = np.where(bmask)
                wy_w = wy_l + oy;  wx_w = wx_l + ox
                valid = (wy_w >= 0) & (wy_w < H) & (wx_w >= 0) & (wx_w < W)
                _fill_blob_voronoi(canvas, char_idx_c, patterns, zmap, psize,
                                   grp, wy_l[valid], wx_l[valid],
                                   wy_w[valid], wx_w[valid], base, z_g)

            else:
                # ── WAVE (shared show_poem_waves pipeline) ────────────────────
                avg_A = np.mean([raw_A[ch] for ch in grp], axis=0)
                sm    = 1.0 + float(z_g[3]) * 9.0
                wv_r  = _lenia_wave(avg_A[:, size//2:].mean(axis=1), row_h, sm)
                amp_r = w_g * (0.06 + float(z_g[5]) * 0.07)
                rw    = brx - amp_r * wv_r
                lw2 = np.full(row_h, float(blx))[:, None];  rw2 = rw[:, None]
                bmask = (xs_all[None, :] >= lw2) & (xs_all[None, :] < rw2)
                iy_rel, wx = np.where(bmask)
                wy_abs = iy_rel + y0
                cx_b = float(blx + brx) / 2
                _fill_wave_voronoi(canvas, char_idx_c, patterns, zmap, psize, row_h,
                                   grp, wy_abs, wx, cx_b, cy_b, w_g, base, z_g)

    # ── Physarum germ: one global run over the full canvas, then overlay ─────
    if germ:
        from scipy.ndimage import binary_dilation as _bd
        _progress(0.85, f"④ 演化黏菌 Physarum ({germ_steps} 步)")
        print(f"[mixed_germ] running global Physarum on {W}×{H} canvas…")
        sv = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
        T  = _global_physarum(W, H, n_steps=germ_steps,
                              sa_deg=germ_sa, ra_deg=germ_ra, so=germ_so, seed=sv)
        thresh     = np.percentile(T, 80)
        trail_mask = _bd(T > thresh, iterations=3)

        # only ~1/5 of groups get carved; the rest are immune
        rng_g   = np.random.default_rng(sv % (2**32))
        affected = rng_g.random(n_g) < (1 / 5)
        carved_px = np.zeros((H, W), bool)
        for g in range(n_g):
            if not affected[g]:
                continue
            grp        = groups[g]
            group_mask = ((char_idx_c >= g * group_size) &
                          (char_idx_c <  g * group_size + len(grp)))
            carved_px |= group_mask
        canvas[trail_mask & carved_px] = 0.0

    # group-to-group seam removed (groups now merge seamlessly where they touch)
    # _draw_group_boundary(canvas, char_idx_c, group_size)
    _progress(0.95, "⑤ 輸出影像")
    _render_save(canvas, W, H, out, pad=0)
    _progress(1.0, "完成")


# ── Path + warped-Voronoi layout ──────────────────────────────────────────────
def show_poem_path(text, size=80, steps=300, cols=5, out="lenia_path.png"):
    """
    Seed points follow a smooth snake path (L→R, R→L alternating rows) whose
    control points are organically perturbed by the poem's avg latent vector.
    Regions are assigned by warped Voronoi (sine-field coordinate distortion),
    giving curved, rhythmic cell boundaries that flow along the path.
    """
    from scipy.spatial import cKDTree
    from scipy.interpolate import splprep, splev
    from scipy.ndimage import binary_dilation
    from matplotlib.path import Path as MplPath

    clean  = _units(text)
    unique = list(dict.fromkeys(clean))
    n      = len(clean)

    print(f"generating {len(unique)} patterns …")
    patterns = {}
    zmap     = {}
    for ch in unique:
        z         = text_to_latent(ch)
        zmap[ch]  = z
        params    = latent_to_params(z)
        sv        = int(hashlib.md5(ch.encode()).hexdigest(), 16) % (2 ** 31)
        A         = run_lenia(params, seed=sv, size=size, steps=steps)
        patterns[ch] = colorize(A)

    rows = (n + cols - 1) // cols
    cell = int(size * 1.8)
    W, H = cols * cell, rows * cell
    mg   = cell * 0.70

    avg_z = np.mean([zmap[ch] for ch in unique], axis=0)

    # ── snake path control points ──────────────────────────────────────────────
    # One horizontal band per row, alternating direction; organic y-jitter from avg_z
    ctrl = []
    for r in range(rows):
        y_c = mg + (r + 0.5) * (H - 2 * mg) / rows
        xs  = np.linspace(mg, W - mg, cols + 2)
        if r % 2 == 1:
            xs = xs[::-1]
        for x in xs:
            k     = len(ctrl)
            y_jit = (float(avg_z[k % 8]) - 0.5) * cell * 0.32
            ctrl.append([float(x), y_c + y_jit])
        # smooth U-turn: add a bridging point between rows
        if r < rows - 1:
            bridge_x = (mg if r % 2 == 0 else W - mg)
            bridge_y = y_c + (H - 2 * mg) / rows / 2
            ctrl.append([bridge_x, bridge_y])

    ctrl    = np.array(ctrl)
    tck_p, _ = splprep([ctrl[:, 0], ctrl[:, 1]], s=len(ctrl) * 0.5, k=3)
    xs_s, ys_s = splev(np.linspace(0, 1, n), tck_p)
    seeds = np.stack([
        np.clip(xs_s, mg * 0.4, W - mg * 0.4),
        np.clip(ys_s, mg * 0.4, H - mg * 0.4)
    ], axis=1)

    # ── warped Voronoi ─────────────────────────────────────────────────────────
    yy, xx = np.mgrid[0:H, 0:W]
    sc   = 2 * np.pi / (cell * 2.8)
    amp  = cell * 0.30
    dx_w = np.zeros((H, W), np.float32)
    dy_w = np.zeros((H, W), np.float32)
    for k in range(6):
        θ    = float(avg_z[k % 8]) * 2 * np.pi + k * 1.4
        φ    = float(avg_z[(k + 4) % 8]) * 2 * np.pi
        wave = np.sin(np.cos(θ) * sc * xx + np.sin(θ) * sc * yy + φ)
        dx_w += amp * wave * float(np.cos(θ + np.pi / 2))
        dy_w += amp * wave * float(np.sin(θ + np.pi / 2))

    tree  = cKDTree(seeds)
    pts   = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    pts_w = np.stack([(xx + dx_w).ravel(), (yy + dy_w).ravel()], axis=1).astype(np.float32)
    _, lbl = tree.query(pts_w)
    labels = lbl.reshape(H, W)

    # ── fill canvas ────────────────────────────────────────────────────────────
    canvas = np.zeros((H, W, 3))
    for i, ch in enumerate(clean):
        wy, wx = np.where(labels == i)
        sy, sx = int(seeds[i, 1]), int(seeds[i, 0])
        canvas[wy, wx] = patterns[ch][(wy - sy) % size, (wx - sx) % size]

    # ── boundaries ────────────────────────────────────────────────────────────
    vd  = labels[1:, :] != labels[:-1, :]
    hd  = labels[:, 1:] != labels[:, :-1]
    bnd = np.zeros((H, W), dtype=bool)
    bnd[1:,  :] |= vd;  bnd[:-1, :] |= vd
    bnd[:,  1:] |= hd;  bnd[:, :-1] |= hd
    bnd = binary_dilation(bnd, iterations=2)
    canvas[bnd] = 0.0

    # ── outer silhouette ───────────────────────────────────────────────────────
    n_o  = 13
    rx, ry = W * 0.46, H * 0.46
    a_b  = np.linspace(0, 2 * np.pi, n_o, endpoint=False)
    jo   = np.pi / n_o * 0.80
    ao   = (a_b + np.array([(avg_z[(i+2)%8]-0.5)*2*jo for i in range(n_o)])) % (2*np.pi)
    ao  += avg_z[0] * 2 * np.pi
    re   = (rx * ry) / np.sqrt((ry * np.cos(ao))**2 + (rx * np.sin(ao))**2)
    ro   = re * (1.0 + np.array([(avg_z[(i+5)%8]-0.5)*0.22 for i in range(n_o)]))
    cxo  = W/2 + (avg_z[3]-0.5)*cell*0.12
    cyo  = H/2 + (avg_z[4]-0.5)*cell*0.12
    tck_o, _ = splprep([cxo + ro*np.cos(ao), cyo + ro*np.sin(ao)], s=0, per=True, k=3)
    xf, yf   = splev(np.linspace(0, 1, 1200, endpoint=False), tck_o)
    canvas[~MplPath(np.stack([xf, yf], axis=1)).contains_points(pts).reshape(H, W)] = 0.0

    # ── render ─────────────────────────────────────────────────────────────────
    _render_save(canvas, W, H, out, pad=0)


# ── C: anisotropic competitive growth ─────────────────────────────────────────
def show_poem_growth(text, size=80, steps=300, cols=5, out="lenia_growth.png"):
    """
    Approach C: each character has its own elliptical 'growth front' —
    direction, aspect ratio, and speed all derived from its latent vector.
    The region each character claims is the set of pixels where its weighted
    distance wins.  Coordinate warp adds additional organic curvature.
    """
    from scipy.interpolate import splprep, splev
    from scipy.ndimage import binary_dilation
    from matplotlib.path import Path as MplPath

    clean  = _units(text)
    unique = list(dict.fromkeys(clean))
    n      = len(clean)

    print(f"[growth] generating {len(unique)} patterns …")
    patterns = {}
    zmap     = {}
    for ch in unique:
        z = text_to_latent(ch);  zmap[ch] = z
        params = latent_to_params(z)
        sv = int(hashlib.md5(ch.encode()).hexdigest(), 16) % (2**31)
        patterns[ch] = colorize(run_lenia(params, seed=sv, size=size, steps=steps))

    rows = (n + cols - 1) // cols
    cell = int(size * 1.7);  W, H = cols * cell, rows * cell;  mg = cell // 5
    avg_z = np.mean([zmap[ch] for ch in unique], axis=0)

    seeds = np.zeros((n, 2))
    for idx, ch in enumerate(clean):
        z = zmap[ch]
        gx = (idx % cols + 0.5) * cell;  gy = (idx // cols + 0.5) * cell
        jit = cell * 0.35
        seeds[idx] = [np.clip(gx + (z[5]-0.5)*2*jit, mg, W-mg),
                      np.clip(gy + (z[6]-0.5)*2*jit, mg, H-mg)]

    # coordinate warp
    yy, xx = np.mgrid[0:H, 0:W]
    sc = 2*np.pi / (cell*2.8);  amp = cell*0.30
    dx_w = np.zeros((H, W), np.float32);  dy_w = np.zeros((H, W), np.float32)
    for k in range(6):
        θ = float(avg_z[k%8])*2*np.pi + k*1.4;  φ = float(avg_z[(k+4)%8])*2*np.pi
        wave = np.sin(np.cos(θ)*sc*xx + np.sin(θ)*sc*yy + φ)
        dx_w += amp*wave*float(np.cos(θ+np.pi/2));  dy_w += amp*wave*float(np.sin(θ+np.pi/2))
    xxw = (xx + dx_w).astype(np.float32);  yyw = (yy + dy_w).astype(np.float32)

    # anisotropic weighted distance (elliptical growth fronts)
    best_d  = np.full((H, W), np.inf, np.float32)
    labels  = np.zeros((H, W), np.int32)
    for i, ch in enumerate(clean):
        z = zmap[ch];  sx, sy = seeds[i]
        θ = float(z[7]) * np.pi
        ratio = 0.45 + float(z[3]) * 1.10   # axis ratio [0.45, 1.55]
        scale = 0.55 + float(z[2]) * 0.90   # growth speed [0.55, 1.45]
        ca, sa = np.cos(θ), np.sin(θ)
        dx = (xxw - sx)*ca + (yyw - sy)*sa
        dy = -(xxw - sx)*sa + (yyw - sy)*ca
        d  = np.sqrt(dx**2 + (dy/ratio)**2) / scale
        better = d < best_d;  labels[better] = i;  best_d[better] = d[better]

    canvas = np.zeros((H, W, 3))
    for i, ch in enumerate(clean):
        wy, wx = np.where(labels == i)
        sy, sx = int(seeds[i,1]), int(seeds[i,0])
        canvas[wy, wx] = patterns[ch][(wy-sy)%size, (wx-sx)%size]

    vd = labels[1:,:] != labels[:-1,:];  hd = labels[:,1:] != labels[:,:-1]
    bnd = np.zeros((H,W), bool)
    bnd[1:,:]|=vd; bnd[:-1,:]|=vd; bnd[:,1:]|=hd; bnd[:,:-1]|=hd
    bnd = binary_dilation(bnd, iterations=2);  canvas[bnd] = 0.0

    n_o=13; rx,ry=W*.46,H*.46
    a_b=np.linspace(0,2*np.pi,n_o,endpoint=False)
    jo=np.pi/n_o*.80
    ao=(a_b+np.array([(avg_z[(i+2)%8]-.5)*2*jo for i in range(n_o)]))%(2*np.pi)+avg_z[0]*2*np.pi
    re=(rx*ry)/np.sqrt((ry*np.cos(ao))**2+(rx*np.sin(ao))**2)
    ro=re*(1.+np.array([(avg_z[(i+5)%8]-.5)*.22 for i in range(n_o)]))
    cxo=W/2+(avg_z[3]-.5)*cell*.12;  cyo=H/2+(avg_z[4]-.5)*cell*.12
    tck_o,_=splprep([cxo+ro*np.cos(ao),cyo+ro*np.sin(ao)],s=0,per=True,k=3)
    xf,yf=splev(np.linspace(0,1,1200,endpoint=False),tck_o)
    pts=np.stack([xx.ravel(),yy.ravel()],axis=1).astype(np.float32)
    canvas[~MplPath(np.stack([xf,yf],axis=1)).contains_points(pts).reshape(H,W)] = 0.0

    _render_save(canvas, W, H, out, pad=0)


# ── Grouped blobs: each organic blob contains 2-3 chars ───────────────────────
def show_poem_grouped(text, size=144, steps=300, cols=4, group_size=3, pat_scale=1.0, out="lenia_grouped.png"):
    """
    Characters are grouped (2-3 per group).  Each group gets ONE organic blob
    shape (from the group's average latent vector, via _char_to_curve).
    Inside the blob the characters carve out sub-regions via mini-Voronoi;
    each sub-region is filled with its character's own Lenia pattern.
    Blobs float separately on a dark background — no single giant canvas.
    """
    (clean, unique, groups, n_g, zmap, patterns,
     _raw, g_z, psize) = _prepare_patterns(text, size, steps, group_size,
                                           want_raw=False, tag="grouped",
                                           pat_scale=pat_scale)
    _progress(0.85, "③ 排版與繪製 blob")

    # per-group blob sizes derived from z[4]: range [1.3, 2.2] × size
    g_blob = {g: int(size * (1.3 + float(g_z[g][4]) * 0.9)) for g in range(n_g)}

    # blob curves: one per group, using per-group blob size
    g_crv = {g: _char_to_curve(g_z[g], g_blob[g]) for g in range(n_g)}

    # tight-pack positions (centroid-anchored, same as show_poem_poly)
    centroids = {g: g_crv[g].mean(axis=0) for g in range(n_g)}
    brad      = {g: float(np.max(np.linalg.norm(g_crv[g] - centroids[g], axis=1)))
                 for g in range(n_g)}
    r_max = max(brad.values())
    cell  = int(2 * r_max + 8);  jit = int(r_max * 0.55);  pad = int(r_max * 1.6)
    rows  = (n_g + cols - 1) // cols
    W     = cols * cell + 2 * pad;  H = rows * cell + 2 * pad

    positions = []
    for g in range(n_g):
        z   = g_z[g]
        gcx = pad + (g % cols) * cell + cell // 2
        gcy = pad + (g // cols) * cell + cell // 2
        wcx = gcx + int((z[5] - 0.5) * 2 * jit)
        wcy = gcy + int((z[6] - 0.5) * 2 * jit)
        lcx, lcy = centroids[g]
        bsz = g_blob[g]
        positions.append((int(np.clip(wcx - lcx, 0, W - bsz)),
                          int(np.clip(wcy - lcy, 0, H - bsz))))

    # rasterise
    canvas     = np.zeros((H, W, 3))
    char_idx_c = np.full((H, W), -1, np.int32)   # for boundary drawing

    for g, grp in enumerate(groups):
        ox, oy = positions[g]
        verts  = g_crv[g]
        blob   = g_blob[g]   # per-group blob size

        # blob mask (polygon + optional punched hole) — shared helper
        z_g = g_z[g]
        bmask = _blob_mask(verts, blob, z_g)

        wy_l, wx_l = np.where(bmask)
        if not len(wy_l):
            continue

        # map to world coords
        wy_w = wy_l + oy;  wx_w = wx_l + ox
        valid = (wy_w < H) & (wx_w < W)
        wy_l, wx_l, wy_w, wx_w = wy_l[valid], wx_l[valid], wy_w[valid], wx_w[valid]

        _fill_blob_voronoi(canvas, char_idx_c, patterns, zmap, psize,
                           grp, wy_l, wx_l, wy_w, wx_w, g * group_size, z_g)

    # group-to-group seam removed (groups now merge seamlessly where they touch)
    # _draw_group_boundary(canvas, char_idx_c, group_size, mask_before=True)
    _progress(0.95, "④ 輸出影像")
    _render_save(canvas, W, H, out, pad=0.1)
    _progress(1.0, "完成")


# ── D-blend: noise banding where each group shares ONE blended Lenia ──────────
def show_poem_noise_blend(text, size=80, steps=300, cols=5, group_size=3, out="lenia_noise_blend.png"):
    """
    Group every `group_size` characters.  Average their latent vectors to get
    a single blended latent, generate ONE Lenia pattern per group, and fill
    each noise-field band with that unified pattern.
    Result: 2-3 characters 'speak together' with one visual voice.
    """
    from scipy.interpolate import splprep, splev
    from scipy.ndimage import binary_dilation
    from matplotlib.path import Path as MplPath

    clean  = _units(text)
    n      = len(clean)
    groups = [clean[i:i+group_size] for i in range(0, n, group_size)]
    n_groups = len(groups)

    print(f"[noise-blend] {n} chars → {n_groups} groups, generating {n_groups} blended patterns …")

    # one blended Lenia pattern per group
    group_patterns = {}
    zmap_all = {ch: text_to_latent(ch) for ch in set(clean)}
    avg_z    = np.mean(list(zmap_all.values()), axis=0)

    for g, group in enumerate(groups):
        z_blend = np.mean([zmap_all[ch] for ch in group], axis=0)  # avg latent
        params  = latent_to_params(z_blend)
        sv      = int(hashlib.md5("".join(group).encode()).hexdigest(), 16) % (2**31)
        A       = run_lenia(params, seed=sv, size=size, steps=steps)
        group_patterns[g] = colorize(A)

    # canvas
    rows = (n + cols - 1) // cols
    cell = int(size * 1.7);  W, H = cols * cell, rows * cell

    # smooth field → n_groups equal-area bands
    yy, xx = np.mgrid[0:H, 0:W]
    field  = np.zeros((H, W), np.float64)
    for k in range(4):
        θ = float(avg_z[k*2 % 8]) * np.pi + k * 0.55
        λ = cell * (1.3 + float(avg_z[(k*2+1) % 8]) * 0.9)
        φ = float(avg_z[(k+4) % 8]) * 2 * np.pi
        field += (0.75**k) * np.sin(np.cos(θ)*2*np.pi/λ*xx + np.sin(θ)*2*np.pi/λ*yy + φ)

    flat    = field.ravel();  order = np.argsort(flat);  npx = H * W
    raw_lbl = np.empty(npx, np.int32)
    for b in range(n_groups):
        raw_lbl[order[b*npx//n_groups:(b+1)*npx//n_groups]] = b
    raw_lbl = raw_lbl.reshape(H, W)

    # sort bands top-left → bottom-right
    band_order = [b for _, b in sorted(
        (( np.where(raw_lbl==b)[0].mean() + np.where(raw_lbl==b)[1].mean()*0.15
           if (raw_lbl==b).any() else np.inf ), b)
        for b in range(n_groups))]
    remap = np.empty(n_groups, np.int32)
    for i, ob in enumerate(band_order): remap[ob] = i
    group_labels = remap[raw_lbl]

    # fill — each band with its group's single blended pattern
    canvas = np.zeros((H, W, 3))
    for g in range(n_groups):
        wy, wx = np.where(group_labels == g)
        if len(wy):
            canvas[wy, wx] = group_patterns[g][wy % size, wx % size]

    # single-level boundary between groups
    vd = group_labels[1:,:] != group_labels[:-1,:]
    hd = group_labels[:,1:] != group_labels[:,:-1]
    bnd = np.zeros((H,W), bool)
    bnd[1:,:]|=vd; bnd[:-1,:]|=vd; bnd[:,1:]|=hd; bnd[:,:-1]|=hd
    canvas[binary_dilation(bnd, iterations=2)] = 0.0

    # outer silhouette
    n_o=13; rx,ry=W*.46,H*.46
    a_b=np.linspace(0,2*np.pi,n_o,endpoint=False)
    jo=np.pi/n_o*.80
    ao=(a_b+np.array([(avg_z[(i+2)%8]-.5)*2*jo for i in range(n_o)]))%(2*np.pi)+avg_z[0]*2*np.pi
    re=(rx*ry)/np.sqrt((ry*np.cos(ao))**2+(rx*np.sin(ao))**2)
    ro=re*(1.+np.array([(avg_z[(i+5)%8]-.5)*.22 for i in range(n_o)]))
    cxo=W/2+(avg_z[3]-.5)*cell*.12;  cyo=H/2+(avg_z[4]-.5)*cell*.12
    tck_o,_=splprep([cxo+ro*np.cos(ao),cyo+ro*np.sin(ao)],s=0,per=True,k=3)
    xf,yf=splev(np.linspace(0,1,1200,endpoint=False),tck_o)
    pts=np.stack([xx.ravel(),yy.ravel()],axis=1).astype(np.float32)
    canvas[~MplPath(np.stack([xf,yf],axis=1)).contains_points(pts).reshape(H,W)] = 0.0

    _render_save(canvas, W, H, out, pad=0)


# ── D: smooth noise-field banding ─────────────────────────────────────────────
def show_poem_noise(text, size=160, steps=300, cols=5, group_size=3,
                    octaves=4, wave_base=1.3, pat_scale=1.0, out="lenia_noise.png"):
    """
    Approach D: a smooth 2D scalar field is built from a few low-frequency
    sinusoids (direction and wavelength from the poem's latent vectors).
    The field is sliced into N equal-area value bands — one per character —
    producing contiguous wood-grain / contour-line shaped regions.
    Bands are sorted top-left→bottom-right to preserve reading rhythm.
    octaves: number of sinusoid layers (2-6); more = more complex grain.
    wave_base: base wavelength multiplier (0.5-3.0); larger = wider bands.
    """
    from scipy.interpolate import splprep, splev
    from scipy.ndimage import binary_dilation
    from matplotlib.path import Path as MplPath

    clean  = _units(text)
    unique = list(dict.fromkeys(clean))
    n      = len(clean)

    pat_scale = max(1.0, float(pat_scale))
    psize     = max(1, int(round(size * pat_scale)))
    print(f"[noise] generating {len(unique)} patterns (×{pat_scale:.2f} → {psize}px) …")
    patterns = {}
    zmap     = {}
    for ch in unique:
        z = text_to_latent(ch);  zmap[ch] = z
        params = latent_to_params(z)
        sv = int(hashlib.md5(ch.encode()).hexdigest(), 16) % (2**31)
        patterns[ch] = _scaled_pattern(
            run_lenia(params, seed=sv, size=size, steps=steps), size, psize)

    rows = (n + cols - 1) // cols
    cell = int(size * 1.7);  W, H = cols * cell, rows * cell
    avg_z = np.mean([zmap[ch] for ch in unique], axis=0)

    # smooth directional field: `octaves` layers, long wavelengths → ribbon-like bands
    yy, xx = np.mgrid[0:H, 0:W]
    field  = np.zeros((H, W), np.float64)
    for k in range(octaves):
        θ = float(avg_z[k*2 % 8]) * np.pi + k * 0.55
        λ = cell * (wave_base + float(avg_z[(k*2+1) % 8]) * 0.9)
        φ = float(avg_z[(k+4) % 8]) * 2 * np.pi
        field += (0.75**k) * np.sin(np.cos(θ)*2*np.pi/λ*xx + np.sin(θ)*2*np.pi/λ*yy + φ)

    # ── group characters ──────────────────────────────────────────────────────
    groups   = [clean[i:i+group_size] for i in range(0, n, group_size)]
    n_groups = len(groups)

    # quantile-based equal-area banding — one band per GROUP
    flat    = field.ravel()
    order   = np.argsort(flat)
    npx     = H * W
    raw_lbl = np.empty(npx, np.int32)
    for b in range(n_groups):
        raw_lbl[order[b*npx//n_groups : (b+1)*npx//n_groups]] = b
    raw_lbl = raw_lbl.reshape(H, W)

    # sort group bands by centroid → reading order
    centroids = []
    for b in range(n_groups):
        wy, wx = np.where(raw_lbl == b)
        centroids.append((wy.mean() + wx.mean()*0.15 if len(wy) else np.inf, b))
    band_order = [b for _, b in sorted(centroids)]
    remap = np.empty(n_groups, np.int32)
    for new_idx, old_b in enumerate(band_order):
        remap[old_b] = new_idx
    group_labels = remap[raw_lbl]   # (H, W) — group index in reading order

    # ── sub-divide each group band among its characters (mini-Voronoi) ────────
    char_labels = np.full((H, W), -1, np.int32)
    for g, group in enumerate(groups):
        wy, wx = np.where(group_labels == g)
        if not len(wy):
            continue
        base = g * group_size
        if len(group) == 1:
            char_labels[wy, wx] = base
            continue
        cx, cy = float(wx.mean()), float(wy.mean())
        cr     = float(max(np.std(wx), np.std(wy), 1.0))
        sub_xy = []
        for k, ch in enumerate(group):
            z     = zmap[ch]
            angle = float(z[5]) * 2*np.pi + k * 2*np.pi / len(group)
            sub_xy.append([cx + cr*0.42*np.cos(angle),
                           cy + cr*0.42*np.sin(angle)])
        sub_xy = np.array(sub_xy)
        dist2  = ((wy[:,None] - sub_xy[None,:,1])**2 +
                  (wx[:,None] - sub_xy[None,:,0])**2)
        for k in range(len(group)):
            char_labels[wy[dist2.argmin(1)==k], wx[dist2.argmin(1)==k]] = base + k

    # ── fill canvas ────────────────────────────────────────────────────────────
    canvas = np.zeros((H, W, 3))
    for g, group in enumerate(groups):
        for k, ch in enumerate(group):
            wy, wx = np.where(char_labels == g*group_size + k)
            if len(wy):
                canvas[wy, wx] = patterns[ch][wy % psize, wx % psize]

    # ── two-level boundaries ──────────────────────────────────────────────────
    def _bnd(lbl, itr):
        vd = lbl[1:,:] != lbl[:-1,:];  hd = lbl[:,1:] != lbl[:,:-1]
        b  = np.zeros(lbl.shape, bool)
        b[1:,:]|=vd; b[:-1,:]|=vd; b[:,1:]|=hd; b[:,:-1]|=hd
        return binary_dilation(b, iterations=itr)

    canvas[_bnd(char_labels,  1)] = 0.0   # thin: within-group char borders
    canvas[_bnd(group_labels, 3)] = 0.0   # thick: between-group borders (override)

    n_o=13; rx,ry=W*.46,H*.46
    a_b=np.linspace(0,2*np.pi,n_o,endpoint=False)
    jo=np.pi/n_o*.80
    ao=(a_b+np.array([(avg_z[(i+2)%8]-.5)*2*jo for i in range(n_o)]))%(2*np.pi)+avg_z[0]*2*np.pi
    re=(rx*ry)/np.sqrt((ry*np.cos(ao))**2+(rx*np.sin(ao))**2)
    ro=re*(1.+np.array([(avg_z[(i+5)%8]-.5)*.22 for i in range(n_o)]))
    cxo=W/2+(avg_z[3]-.5)*cell*.12;  cyo=H/2+(avg_z[4]-.5)*cell*.12
    tck_o,_=splprep([cxo+ro*np.cos(ao),cyo+ro*np.sin(ao)],s=0,per=True,k=3)
    xf,yf=splev(np.linspace(0,1,1200,endpoint=False),tck_o)
    pts=np.stack([xx.ravel(),yy.ravel()],axis=1).astype(np.float32)
    canvas[~MplPath(np.stack([xf,yf],axis=1)).contains_points(pts).reshape(H,W)] = 0.0

    _render_save(canvas, W, H, out, pad=0)


# ── Watershed tessellation layout ─────────────────────────────────────────────
def show_poem_watershed(text, size=80, steps=300, cols=5, out="lenia_watershed.png"):
    """
    Build a 2D height field from anisotropic Gaussian basins — one per character,
    shaped by its latent vector (size, aspect ratio, orientation).  Watershed
    flooding from each character's seed produces organically curved cell boundaries
    whose shapes reflect the latent geometry, not a grid.
    """
    from skimage.segmentation import watershed
    from scipy.ndimage import binary_dilation
    from scipy.interpolate import splprep, splev
    from matplotlib.path import Path as MplPath

    clean  = _units(text)
    unique = list(dict.fromkeys(clean))
    n      = len(clean)

    print(f"generating {len(unique)} patterns for {n} watershed cells …")
    patterns = {}
    zmap     = {}
    for ch in unique:
        z         = text_to_latent(ch)
        zmap[ch]  = z
        params    = latent_to_params(z)
        seed_v    = int(hashlib.md5(ch.encode()).hexdigest(), 16) % (2 ** 31)
        A         = run_lenia(params, seed=seed_v, size=size, steps=steps)
        patterns[ch] = colorize(A)

    # ── canvas & seed positions ────────────────────────────────────────────────
    rows = (n + cols - 1) // cols
    cell = int(size * 1.7)
    W, H = cols * cell, rows * cell
    mg   = cell // 5

    seeds = np.zeros((n, 2))
    for idx, ch in enumerate(clean):
        z  = zmap[ch]
        gx = (idx % cols  + 0.5) * cell
        gy = (idx // cols + 0.5) * cell
        jit = cell * 0.35
        seeds[idx, 0] = float(np.clip(gx + (z[5] - 0.5) * 2 * jit, mg, W - mg))
        seeds[idx, 1] = float(np.clip(gy + (z[6] - 0.5) * 2 * jit, mg, H - mg))

    # ── height field: sum of anisotropic Gaussian basins ──────────────────────
    yy, xx = np.mgrid[0:H, 0:W]
    field  = np.zeros((H, W), dtype=np.float32)

    for i, ch in enumerate(clean):
        z  = zmap[ch]
        sx, sy  = seeds[i]
        sigma   = cell * (0.45 + z[1] * 0.35)   # basin width  [0.45, 0.80] × cell
        depth   = 0.8  + z[2] * 0.6              # basin depth  [0.8,  1.4]
        ratio   = 0.55 + z[3] * 0.90             # aspect ratio [0.55, 1.45]
        angle   = z[7] * np.pi                   # tilt angle   [0, π]
        ca, sa  = np.cos(angle), np.sin(angle)

        # rotate local coords
        dx = (xx - sx) * ca + (yy - sy) * sa
        dy = -(xx - sx) * sa + (yy - sy) * ca

        r2 = (dx / sigma) ** 2 + (dy / (sigma * ratio)) ** 2
        field -= (depth * np.exp(-r2 / 2)).astype(np.float32)

    # ── watershed from seed markers ────────────────────────────────────────────
    markers = np.zeros((H, W), dtype=np.int32)
    for i in range(n):
        mx = int(np.clip(seeds[i, 0], 0, W - 1))
        my = int(np.clip(seeds[i, 1], 0, H - 1))
        markers[my, mx] = i + 1

    labels = watershed(field, markers=markers) - 1   # 0-indexed

    # ── fill canvas ────────────────────────────────────────────────────────────
    canvas = np.zeros((H, W, 3))
    for i, ch in enumerate(clean):
        wy, wx = np.where(labels == i)
        sy, sx = int(seeds[i, 1]), int(seeds[i, 0])
        py = (wy - sy) % size
        px = (wx - sx) % size
        canvas[wy, wx] = patterns[ch][py, px]

    # ── cell boundaries ────────────────────────────────────────────────────────
    vd  = labels[1:, :] != labels[:-1, :]
    hd  = labels[:, 1:] != labels[:, :-1]
    bnd = np.zeros((H, W), dtype=bool)
    bnd[1:,  :] |= vd;  bnd[:-1, :] |= vd
    bnd[:,  1:] |= hd;  bnd[:, :-1] |= hd
    bnd = binary_dilation(bnd, iterations=2)
    canvas[bnd] = 0.0

    # ── outer irregular silhouette ─────────────────────────────────────────────
    avg_z = np.mean([zmap[ch] for ch in unique], axis=0)
    n_o   = 13
    rx, ry = W * 0.46, H * 0.46
    a_base = np.linspace(0, 2 * np.pi, n_o, endpoint=False)
    jit_o  = np.pi / n_o * 0.80
    po     = np.array([(avg_z[(i + 2) % 8] - 0.5) * 2 * jit_o for i in range(n_o)])
    a_out  = (a_base + po) % (2 * np.pi) + avg_z[0] * 2 * np.pi
    r_ell  = (rx * ry) / np.sqrt((ry * np.cos(a_out)) ** 2 + (rx * np.sin(a_out)) ** 2)
    r_out  = r_ell * (1.0 + np.array([(avg_z[(i + 5) % 8] - 0.5) * 0.22 for i in range(n_o)]))
    cxo    = W / 2 + (avg_z[3] - 0.5) * cell * 0.12
    cyo    = H / 2 + (avg_z[4] - 0.5) * cell * 0.12
    tck_o, _ = splprep([cxo + r_out * np.cos(a_out), cyo + r_out * np.sin(a_out)],
                        s=0, per=True, k=3)
    xf, yf = splev(np.linspace(0, 1, 1200, endpoint=False), tck_o)

    pts        = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    outer_mask = MplPath(np.stack([xf, yf], axis=1)).contains_points(pts).reshape(H, W)
    canvas[~outer_mask] = 0.0

    # ── render ─────────────────────────────────────────────────────────────────
    _render_save(canvas, W, H, out, pad=0)


# ── Voronoi tessellation layout ───────────────────────────────────────────────
def show_poem_voronoi(text, size=160, steps=300, cols=5, pat_scale=1.0, out="lenia_voronoi.png"):
    """
    Divide the canvas into N Voronoi cells — one per character occurrence.
    Seed positions come from the latent vector (jittered grid via z[5], z[6]).
    Each cell is filled with its character's Lenia pattern tiled from the seed.
    Result: a seamless mosaic where every pixel belongs to a character, with
    organic curved boundaries derived from the latent space geometry.
    """
    from scipy.spatial import cKDTree

    clean  = _units(text)
    unique = list(dict.fromkeys(clean))
    n      = len(clean)

    pat_scale = max(1.0, float(pat_scale))
    psize     = max(1, int(round(size * pat_scale)))
    print(f"generating {len(unique)} Lenia patterns for {n} cells "
          f"(×{pat_scale:.2f} → {psize}px) …")
    patterns = {}
    zmap     = {}
    for ch in unique:
        z          = text_to_latent(ch)
        zmap[ch]   = z
        params     = latent_to_params(z)
        seed_v     = int(hashlib.md5(ch.encode()).hexdigest(), 16) % (2 ** 31)
        A          = run_lenia(params, seed=seed_v, size=size, steps=steps)
        patterns[ch] = _scaled_pattern(A, size, psize)   # (psize, psize, 3)

    # ── canvas & seed positions ────────────────────────────────────────────────
    rows   = (n + cols - 1) // cols
    cell   = int(size * 1.7)          # canvas cells are larger than the Lenia tile
    W, H   = cols * cell, rows * cell
    mg     = cell // 5                # margin so seeds don't sit on the edge

    # Each character occurrence gets its own seed.
    # Base position = grid centre; jitter derived from z[5] and z[6].
    seeds = np.zeros((n, 2))          # columns: x, y
    for idx, ch in enumerate(clean):
        z   = zmap[ch]
        gx  = (idx % cols  + 0.5) * cell
        gy  = (idx // cols + 0.5) * cell
        jit = cell * 0.35
        seeds[idx, 0] = float(np.clip(gx + (z[5] - 0.5) * 2 * jit, mg, W - mg))
        seeds[idx, 1] = float(np.clip(gy + (z[6] - 0.5) * 2 * jit, mg, H - mg))

    # ── smooth coordinate warp → curved cell boundaries ──────────────────────
    # Before Voronoi assignment, distort pixel coordinates with a smooth field
    # derived from avg_z so that the "nearest seed" boundaries become curves.
    avg_z  = np.mean([zmap[ch] for ch in unique], axis=0)
    yy, xx = np.mgrid[0:H, 0:W]
    sc     = 2 * np.pi / (cell * 2.8)   # spatial frequency of warp waves
    amp    = cell * 0.32                 # warp amplitude in pixels
    dx_w   = np.zeros((H, W), np.float32)
    dy_w   = np.zeros((H, W), np.float32)
    for k in range(6):
        θ = float(avg_z[k % 8]) * 2 * np.pi + k * 1.4
        φ = float(avg_z[(k + 4) % 8]) * 2 * np.pi
        wave = np.sin(np.cos(θ) * sc * xx + np.sin(θ) * sc * yy + φ)
        dx_w += amp * wave * float(np.cos(θ + np.pi / 2))
        dy_w += amp * wave * float(np.sin(θ + np.pi / 2))

    # ── Voronoi labelling (in warped coordinate space) ─────────────────────────
    tree   = cKDTree(seeds)
    pts    = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)  # original coords (for outer mask)
    pts_w  = np.stack([(xx + dx_w).ravel(), (yy + dy_w).ravel()], axis=1).astype(np.float32)
    _, lbl = tree.query(pts_w)
    labels = lbl.reshape(H, W)   # (H, W)  int, value = character index

    # ── fill canvas ────────────────────────────────────────────────────────────
    canvas = np.zeros((H, W, 3))
    for i, ch in enumerate(clean):
        wy, wx = np.where(labels == i)
        # Tile the Lenia pattern from the seed position
        sy, sx = int(seeds[i, 1]), int(seeds[i, 0])
        py = (wy - sy) % psize
        px = (wx - sx) % psize
        canvas[wy, wx] = patterns[ch][py, px]

    # ── cell boundary: widen to ~3 px ─────────────────────────────────────────
    from scipy.ndimage import binary_dilation
    vd  = labels[1:, :] != labels[:-1, :]
    hd  = labels[:, 1:] != labels[:, :-1]
    bnd = np.zeros((H, W), dtype=bool)
    bnd[1:,  :] |= vd;  bnd[:-1, :] |= vd
    bnd[:,  1:] |= hd;  bnd[:, :-1] |= hd
    bnd = binary_dilation(bnd, iterations=2)   # 2-pass dilation → ~5 px wide
    canvas[bnd] = 0.0

    # ── outer irregular boundary ───────────────────────────────────────────────
    from scipy.interpolate import splprep, splev
    from matplotlib.path import Path as MplPath
    # avg_z already computed above
    n_o   = 13
    # Ellipse semi-axes sized to cover the full rectangular canvas interior
    rx, ry  = W * 0.46, H * 0.46
    a_base  = np.linspace(0, 2 * np.pi, n_o, endpoint=False)
    jit_o   = np.pi / n_o * 0.80
    perturb_o = np.array([(avg_z[(i + 2) % 8] - 0.5) * 2 * jit_o for i in range(n_o)])
    a_out   = (a_base + perturb_o) % (2 * np.pi)
    a_out  += avg_z[0] * 2 * np.pi              # global rotation from avg latent

    # Ellipse radius at each angle, then perturbed ±20 % per control point
    r_ell  = (rx * ry) / np.sqrt((ry * np.cos(a_out)) ** 2 + (rx * np.sin(a_out)) ** 2)
    r_var  = np.array([(avg_z[(i + 5) % 8] - 0.5) * 0.22 for i in range(n_o)])
    r_out  = r_ell * (1.0 + r_var)

    cxo = W / 2 + (avg_z[3] - 0.5) * cell * 0.12
    cyo = H / 2 + (avg_z[4] - 0.5) * cell * 0.12
    ctrl_x = cxo + r_out * np.cos(a_out)
    ctrl_y = cyo + r_out * np.sin(a_out)

    tck_o, _ = splprep([ctrl_x, ctrl_y], s=0, per=True, k=3)
    u_o      = np.linspace(0, 1, 1200, endpoint=False)
    xf, yf   = splev(u_o, tck_o)

    outer_path = MplPath(np.stack([xf, yf], axis=1))
    outer_mask = outer_path.contains_points(pts).reshape(H, W)
    canvas[~outer_mask] = 0.0               # everything outside → dark background

    # ── render ─────────────────────────────────────────────────────────────────
    _render_save(canvas, W, H, out, pad=0)


# ── Triangular mosaic layout ──────────────────────────────────────────────────
def show_poem_triangles(text, size=72, steps=300, cols=5, out="lenia_triangles.png"):
    """
    Each character's Lenia square is sliced into 4 triangles (N/E/S/W) by
    both diagonals.  Every grid cell assembles 4 different characters' slices
    into a pinwheel — creating a flat geometric mosaic.
    """
    clean  = _units(text)
    unique = list(dict.fromkeys(clean))
    n      = len(clean)

    print(f"generating {len(unique)} unique patterns for {n} characters …")
    patterns = {}
    for ch in unique:
        z      = text_to_latent(ch)
        params = latent_to_params(z)
        seed   = int(hashlib.md5(ch.encode()).hexdigest(), 16) % (2 ** 31)
        A      = run_lenia(params, seed=seed, size=size, steps=steps)
        patterns[ch] = colorize(A)

    # ── Triangle masks (both diagonals through centre) ────────────────────────
    rs = np.arange(size).reshape(-1, 1)   # (size, 1)
    cs = np.arange(size).reshape(1, -1)   # (1, size)
    #   \ diagonal: cs > rs  → NE half
    #   / diagonal: rs+cs < size → NW half
    mask_N = ((cs >  rs) & (rs + cs <  size))[:, :, None].astype(float)  # top
    mask_E = ((cs >  rs) & (rs + cs >= size))[:, :, None].astype(float)  # right
    mask_S = ((cs <= rs) & (rs + cs >= size))[:, :, None].astype(float)  # bottom
    mask_W = ((cs <= rs) & (rs + cs <  size))[:, :, None].astype(float)  # left

    # thin black separator along both diagonals
    sep = ((np.abs(rs + cs - (size - 1)) < 1.5) |
           (np.abs(rs - cs)              < 1.5))

    # ── Build one cell per character ──────────────────────────────────────────
    # Each cell mixes 4 chars offset by ~n/4 steps in the poem sequence
    q = max(1, n // 4)
    cells = []
    for k in range(n):
        cN = clean[ k            % n]
        cE = clean[(k +     q)   % n]
        cS = clean[(k + 2 * q)   % n]
        cW = clean[(k + 3 * q)   % n]
        cell = (patterns[cN] * mask_N + patterns[cE] * mask_E +
                patterns[cS] * mask_S + patterns[cW] * mask_W)
        cell[sep] = 0.0   # carve dividers
        cells.append(cell)

    # ── Grid assembly ─────────────────────────────────────────────────────────
    rows_needed = (n + cols - 1) // cols
    while len(cells) < rows_needed * cols:
        cells.append(np.zeros((size, size, 3)))

    gap = 3
    canvas_rows = []
    for r in range(rows_needed):
        row_pieces = []
        for ci in range(cols):
            row_pieces.append(cells[r * cols + ci])
            if ci < cols - 1:
                row_pieces.append(np.zeros((size, gap, 3)))
        row_arr = np.concatenate(row_pieces, axis=1)
        canvas_rows.append(row_arr)
        if r < rows_needed - 1:
            canvas_rows.append(np.zeros((gap, row_arr.shape[1], 3)))
    canvas = np.concatenate(canvas_rows, axis=0)

    dpi = 110
    fig, ax = plt.subplots(figsize=(canvas.shape[1] / dpi, canvas.shape[0] / dpi))
    fig.patch.set_facecolor("#080808")
    ax.set_facecolor("#080808")
    ax.imshow(canvas, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0.1)
    plt.savefig(out, dpi=dpi*2, bbox_inches="tight", facecolor="#000000")
    plt.show()
    print(f"saved → {out}")


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:]
    poem = "床前明月光，疑是地上霜。舉頭望明月，低頭思故鄉。"
    if args == ["--anchors"]:
        show_anchors()
    elif args == ["--triangles"]:
        show_poem_triangles(poem)
    elif args == ["--poly"]:
        show_poem_poly(poem)
    elif args == ["--voronoi"]:
        show_poem_voronoi(poem)
    elif args == ["--waves"]:
        show_poem_waves(poem)
    elif args == ["--mixed"]:
        show_poem_mixed(poem)
    elif args == ["--path"]:
        show_poem_path(poem)
    elif args == ["--growth"]:
        show_poem_growth(poem)
    elif args == ["--noise"]:
        show_poem_noise(poem)
    elif args == ["--noise-blend"]:
        show_poem_noise_blend(poem)
    elif args == ["--grouped"]:
        show_poem_grouped(poem)
    elif not args or args == ["--poem"]:
        show_poem(poem)
    elif len(args) == 1 and len(args[0]) > 1:
        show_poem_voronoi(args[0])
    else:
        show_texts(args)

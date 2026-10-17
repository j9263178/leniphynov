"""
注音拼接造字 demo(獨立路線,和 Lenia 無關)

    注音音節 → 拆成 聲母 / 介音 / 韻母 / 聲調 → 把各符號的字形依不同「拼法」組成一個方塊字

不靠任何模型,純粹用字型裡的注音符號當基元,看拼接規則能不能造出像字的東西。

run:
    python zhuyin_glyph.py                     # 內建範例,輸出 zhuyin_demo.png
    python zhuyin_glyph.py ㄇㄠ ㄒㄧㄠˇ ㄕㄨㄟˇ   # 自己指定音節
    python zhuyin_glyph.py --random 8          # 隨機合法音節
    python zhuyin_glyph.py ㄏㄨㄢ --out a.png        # 只給檔名時存進 gallery/
"""
from __future__ import annotations
import argparse
import math
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage, signal
from skimage.morphology import skeletonize

# ── 注音符號分類 ──────────────────────────────────────────────────────────────
INITIALS = "ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙ"
MEDIALS = "ㄧㄨㄩ"
FINALS = "ㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ"
TONES = "ˊˇˋ˙"

# 隨機造音節時用的簡化組合
_MED_FIN = {
    "": "ㄚㄛㄜㄞㄟㄠㄡㄢㄣㄤㄥ",
    "ㄧ": "ㄚㄝㄠㄡㄢㄣㄤㄥ",
    "ㄨ": "ㄚㄛㄞㄟㄢㄣㄤㄥ",
    "ㄩ": "ㄝㄢㄣ",
}

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msjhbd.ttc",
    r"C:\Windows\Fonts\msjh.ttc",
    r"C:\Windows\Fonts\kaiu.ttf",
    r"C:\Windows\Fonts\mingliu.ttc",
]
LABEL_FONT_CANDIDATES = [r"C:\Windows\Fonts\msjh.ttc", r"C:\Windows\Fonts\kaiu.ttf"]


def _load_font(cands, size):
    for p in cands:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# ── 1. 音節解析 ───────────────────────────────────────────────────────────────
def parse(syllable: str) -> dict:
    """'ㄒㄧㄠˇ' → {'units': [('I','ㄒ'),('M','ㄧ'),('F','ㄠ')], 'tones': ['ˇ']}

    多個音節用 + 連接,會合成「一個字」,符號數可以超過 3:'ㄒㄧㄠˇ+ㄌㄧㄤˊ' → 6 個符號、2 個聲調。
    """
    units, tones = [], []
    for seg in syllable.split("+"):
        body = []
        for ch in seg.strip():
            if ch in TONES:
                tones.append(ch)
            elif ch in INITIALS + MEDIALS + FINALS:
                body.append(ch)
        for i, ch in enumerate(body):
            if ch in INITIALS and i == 0:
                units.append(("I", ch))
            elif ch in MEDIALS and i + 1 < len(body) and body[i + 1] in FINALS:
                units.append(("M", ch))
            else:
                units.append(("F", ch))
    if not units:
        raise ValueError(f"不是合法注音: {syllable!r}")
    return {"units": units, "tones": tones}


def random_syllable(rng: random.Random) -> str:
    s = ""
    if rng.random() < 0.85:
        s += rng.choice(INITIALS)
    med = rng.choice(["", "", "ㄧ", "ㄨ", "ㄩ"])
    s += med + rng.choice(_MED_FIN[med])
    s += rng.choice(["", "", "ˊ", "ˇ", "ˋ"])
    return s


def random_word(rng: random.Random, parts: int) -> str:
    return "+".join(random_syllable(rng) for _ in range(parts))


# ── 2. 單一符號 → 點陣(裁到 bbox) ───────────────────────────────────────────
_SYM_CACHE: dict[str, np.ndarray] = {}


def render_symbol(ch: str, px: int = 360) -> np.ndarray:
    """回傳 float32 0..1,黑字=1,已裁到字形外框。"""
    if ch in _SYM_CACHE:
        return _SYM_CACHE[ch]
    font = _load_font(FONT_CANDIDATES, px)
    im = Image.new("L", (px * 2, px * 2), 0)
    ImageDraw.Draw(im).text((px // 2, px // 2), ch, fill=255, font=font)
    a = np.asarray(im, dtype=np.float32) / 255.0
    ys, xs = np.where(a > 0.05)
    if len(ys) == 0:
        raise RuntimeError(f"字型沒有 {ch!r}")
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    _SYM_CACHE[ch] = a
    return a


# ── 3. 放置:把符號塞進畫布的一個方框 ────────────────────────────────────────
def place(canvas: np.ndarray, sym: np.ndarray, box, stroke: float, stretch: float = 0.6):
    """把符號縮放進方框,只取「中心線」(骨架)疊進 canvas(bool)。

    筆畫粗度不跟著縮放走:這裡只放 1px 的中心線,最後由 finalize() 統一以固定粗度重畫。
    box=(x0,y0,x1,y1) 為 0..1 的畫布比例。stretch: 0=保持比例 1=拉滿方框。
    """
    S = canvas.shape[0]
    x0, y0, x1, y1 = box
    bw, bh = (x1 - x0) * S, (y1 - y0) * S
    h, w = sym.shape
    s_fit = min(bw / w, bh / h)
    sx = s_fit * (bw / w / s_fit) ** stretch
    sy = s_fit * (bh / h / s_fit) ** stretch
    nw, nh = max(1, int(round(w * sx))), max(1, int(round(h * sy)))
    # 極端長寬比(如 ㄧ 這種單一橫)只沿長邊拉長
    thin = max(3, int(round(stroke)))
    if w / h > 2.5:
        nw, nh = max(1, int(round(bw))), thin
    elif h / w > 2.5:
        nw, nh = thin, max(1, int(round(bh)))
    img = Image.fromarray((sym * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)
    mask = np.pad(np.asarray(img) > 127, 2)
    skel = skeletonize(mask)[2:-2, 2:-2]
    cx, cy = x0 * S + bw / 2, y0 * S + bh / 2
    px0, py0 = int(round(cx - nw / 2)), int(round(cy - nh / 2))
    cx0, cy0 = max(px0, 0), max(py0, 0)
    cx1, cy1 = min(px0 + nw, S), min(py0 + nh, S)
    if cx1 > cx0 and cy1 > cy0:
        canvas[cy0:cy1, cx0:cx1] |= skel[cy0 - py0:cy1 - py0, cx0 - px0:cx1 - px0]


def _dilate(mask: np.ndarray, r: float) -> np.ndarray:
    return ndimage.distance_transform_edt(~mask) <= r


def _degree(sk: np.ndarray) -> np.ndarray:
    nb = ndimage.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - sk
    return nb


def prune(sk: np.ndarray, k: int) -> np.ndarray:
    """剪掉長度 < k 的細短分支(骨架化產生的毛刺),保留主幹的真正末端。"""
    cur = sk.copy()
    for _ in range(k):                       # 從所有端點往內啃 k 步:毛刺啃光,主幹縮短
        cur &= ~(cur & (_degree(cur) <= 1))
    for _ in range(k):                       # 只從剩下的端點沿原骨架長回 k 步:主幹復原,毛刺不會長回
        ep = cur & (_degree(cur) <= 1)
        cur |= ndimage.binary_dilation(ep, np.ones((3, 3), bool)) & sk & ~cur
    # 整段都比 k 短的獨立小筆畫(如輕聲點)會被啃光,把它們原樣放回
    lab, n = ndimage.label(sk, np.ones((3, 3), int))
    for i in range(1, n + 1):
        comp = lab == i
        if not (cur & comp).any():
            cur |= comp
    return cur


def nib_stamp(center: np.ndarray, stroke: float, angle_deg: float,
               width: float = 2.1, hair: float = 0.2,
               extra: np.ndarray | None = None, round_ends=None,
               balls=None) -> np.ndarray:
    """寬筆尖(扁頭筆):讓一支斜放的扁筆尖沿中心線掃過去。

    筆尖是一條斜線段(長 = width×筆畫粗,厚 = hair×筆畫粗,與水平夾 angle_deg、向右上升)。
    筆畫方向和筆尖垂直時最粗,平行時只剩髮絲;筆畫端點自然成為斜切的刀口。
    這正是歌德體(blackletter)粗細與稜角的來源。
    """
    L, t = stroke * width, max(2.0, stroke * hair)
    n = int(math.ceil(L)) + 6
    c = n / 2
    a = math.radians(angle_deg)
    ux, uy = math.cos(a), -math.sin(a)          # 筆尖方向(影像座標 y 向下)
    vx, vy = -uy, ux                             # 垂直方向
    pts = [(c + sx * ux * L / 2 + sy * vx * t / 2, c + sx * uy * L / 2 + sy * vy * t / 2)
           for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    k = Image.new("L", (n, n), 0)
    ImageDraw.Draw(k).polygon(pts, fill=255)
    kernel = (np.asarray(k, dtype=np.float32) > 127).astype(np.float32)
    out = signal.fftconvolve(center.astype(np.float32), kernel, mode="same") > 0.5
    if round_ends:
        out = _round_caps(out, round_ends, L, t, angle_deg)
    if balls:
        out = _trim_to_balls(out, balls, math.hypot(L / 2, t / 2))
    if extra is not None:
        out = out | extra
    # 骨架是階梯狀的,掃出來的邊緣會鋸齒:輕輕模糊再門檻化,磨平鋸齒但保留刀口稜角
    out = ndimage.gaussian_filter(out.astype(np.float32), stroke * 0.07)
    return (out > 0.5).astype(np.float32)


K8 = np.ones((3, 3), bool)


def _rdp(pts: np.ndarray, eps: float) -> np.ndarray:
    """Ramer–Douglas–Peucker:把彎曲的點列簡化成少數幾個轉折點(折角)。"""
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    n = np.hypot(*ab)
    if n < 1e-6:
        d = np.hypot(*(pts - a).T)
    else:
        d = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / n
    i = int(np.argmax(d))
    if d[i] <= eps:
        return np.array([a, b])
    left, right = _rdp(pts[:i + 1], eps), _rdp(pts[i:], eps)
    return np.vstack([left[:-1], right])


def _walk(pixels: set) -> list:
    """把一段(無分叉)骨架像素依序走成一條點列。"""
    def nbrs(q):
        y, x = q
        return [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dy or dx) and (y + dy, x + dx) in pixels]
    ends = [q for q in pixels if len(nbrs(q)) <= 1]
    cur = ends[0] if ends else next(iter(pixels))
    path, seen = [cur], {cur}
    while True:
        nxt = [q for q in nbrs(cur) if q not in seen]
        if not nxt:
            break
        cur = min(nxt, key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
        path.append(cur)
        seen.add(cur)
    if len(path) > 2 and max(abs(path[0][0] - path[-1][0]), abs(path[0][1] - path[-1][1])) <= 1:
        path.append(path[0])                     # 封閉的圈
    return path


def skeleton_polylines(sk: np.ndarray, eps: float):
    """骨架 → 折線列表。回傳 [(點列 ndarray[(y,x)...], (起點是否為真正末端, 終點是否為真正末端))]。

    先在分叉處切開、各段走成點列,再把「假分叉」(只連兩段的點)接回成一條,
    最後用 RDP 把每條折成直線段。
    """
    deg = _degree(sk)
    jd = ndimage.binary_dilation(sk & (deg >= 3), K8)
    jlab, jn = ndimage.label(jd, K8)
    cent = {i + 1: np.array(c) for i, c in
            enumerate(ndimage.center_of_mass(jd, jlab, range(1, jn + 1)))} if jn else {}
    jnear = ndimage.maximum_filter(jlab, size=5)
    blab, bn = ndimage.label(sk & ~jd, K8)

    polys = []
    for i, sl in enumerate(ndimage.find_objects(blab), start=1):
        ys, xs = np.nonzero(blab[sl] == i)
        pix = {(int(y) + sl[0].start, int(x) + sl[1].start) for y, x in zip(ys, xs)}
        path = _walk(pix)
        polys.append({"pts": [np.array(q, float) for q in path],
                      "a": int(jnear[path[0]]), "b": int(jnear[path[-1]])})

    # 只連兩段的「假分叉」:接成一條
    while True:
        att = {}
        for k, pl in enumerate(polys):
            if pl is None:
                continue
            for end in ("a", "b"):
                if pl[end]:
                    att.setdefault(pl[end], []).append((k, end))
        two = [(c, v) for c, v in att.items() if len(v) == 2]
        if not two:
            break
        c, ((i, ei), (j, ej)) = two[0]
        if i == j:
            polys[i]["pts"].append(polys[i]["pts"][0])
            polys[i]["a"] = polys[i]["b"] = 0
            continue
        pi, pj = polys[i], polys[j]
        if ei == "a":
            pi = {"pts": pi["pts"][::-1], "a": pi["b"], "b": pi["a"]}
        if ej == "b":
            pj = {"pts": pj["pts"][::-1], "a": pj["b"], "b": pj["a"]}
        polys[i] = {"pts": pi["pts"] + [cent[c]] + pj["pts"], "a": pi["a"], "b": pj["b"]}
        polys[j] = None

    out = []
    for pl in polys:
        if pl is None:
            continue
        pts = list(pl["pts"])
        if pl["a"]:
            pts.insert(0, cent[pl["a"]])
        if pl["b"]:
            pts.append(cent[pl["b"]])
        arr = _rdp(np.array(pts), eps)
        out.append((arr, (pl["a"] == 0, pl["b"] == 0)))
    return out


def rasterize(polys, shape) -> np.ndarray:
    im = Image.new("L", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(im)
    for arr, _ in polys:
        xy = [(float(x), float(y)) for y, x in arr]
        if len(xy) == 1:
            d.point(xy[0], fill=255)
        else:
            d.line(xy, fill=255, width=1)
    return np.asarray(im) > 0


FOOT_DX, FOOT_DY = 1.5, 1.2     # 菱形尖腳半對角線(單位:筆畫粗)
FOOT_R = 1.25                   # 圓形腳的半徑(單位:筆畫粗)


def feet_mask(polys, stroke: float, shape, kind: str = "diamond",
              all_ends: bool = False, r_mult: float | None = None,
              balls: list | None = None) -> np.ndarray:
    """在筆畫的真正末端畫上腳(kind: diamond 菱形 / circle 圓形)。

    預設只加在「接近垂直」的筆畫;all_ends=True 則每個末端都加(聲調記號用),
    單點(輕聲)也會畫成一個圓。
    balls:若給一個 list,會把每顆圓球記成 (x, y, r, 向外方向 (ux, uy) 或 None) 加進去,
    讓寬筆尖把伸出圓外的刀口修掉。
    """
    im = Image.new("L", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(im)
    dx, dy = stroke * FOOT_DX, stroke * FOOT_DY
    r_ball = stroke * (FOOT_R if r_mult is None else r_mult)
    for arr, (free_a, free_b) in polys:
        if len(arr) < 2:
            if all_ends and kind == "circle" and len(arr) == 1:
                y, x = arr[0]
                d.ellipse([x - r_ball, y - r_ball, x + r_ball, y + r_ball], fill=255)
                if balls is not None:
                    balls.append((x, y, r_ball, None))
            continue
        for free, tip, rest in ((free_a, arr[0], arr[1:]), (free_b, arr[-1], arr[-2::-1])):
            if not free:
                continue
            far = next((q for q in rest if np.hypot(*(q - tip)) >= stroke * 1.2), rest[-1])
            vy, vx = far - tip
            if all_ends or abs(vx) < 0.45 * abs(vy):     # 接近垂直(或指定全部末端)
                y, x = tip
                if kind == "circle":
                    d.ellipse([x - r_ball, y - r_ball, x + r_ball, y + r_ball], fill=255)
                    if balls is not None:
                        n = math.hypot(vx, vy) or 1.0
                        balls.append((x, y, r_ball, (-vx / n, -vy / n)))
                else:
                    d.polygon([(x - dx, y), (x, y - dy), (x + dx, y), (x, y + dy)], fill=255)
    return np.asarray(im) > 0


def _round_caps(out: np.ndarray, ends, L: float, t: float, angle_deg: float) -> np.ndarray:
    """把筆畫末端「雙斜」的刀口改成圓頭(不是另外加圓球)。

    ends: [(尖端 (y,x), 筆畫由內向外的單位方向 (x,y))]。
    做法:在筆畫剛好達到完整粗度的位置把多出來的斜切部分切掉,再補上半徑 = 半粗的圓。
    """
    H, W = out.shape
    yy, xx = np.mgrid[:H, :W]
    a_ = math.radians(angle_deg)
    nx, ny = math.cos(a_), -math.sin(a_)                 # 筆尖方向
    out = out.copy()
    for (ty, tx), (ux, uy) in ends:
        cos_d = abs(ux * nx + uy * ny)
        sin_d = abs(ux * ny - uy * nx)
        r = 0.5 * (L * sin_d + t * cos_d)                # 筆畫半粗
        a = 0.5 * (L * cos_d + t * sin_d)                # 斜切刀口沿筆畫方向的半長
        cx, cy = tx - ux * a, ty - uy * a                # 完整粗度開始的位置
        rx, ry = xx - cx, yy - cy
        along = rx * ux + ry * uy
        perp = np.abs(-rx * uy + ry * ux)
        out &= ~((along > 0) & (along <= a + L) & (perp <= r * 1.25))
        out |= rx * rx + ry * ry <= r * r
    return out


def horizontal_ends(polys, stroke: float):
    """所有「接近水平」的筆畫真正末端:[(尖端 (y,x), 由內向外單位方向 (x,y))]。"""
    ends = []
    for arr, (free_a, free_b) in polys:
        if len(arr) < 2:
            continue
        for free, tip, rest in ((free_a, arr[0], arr[1:]), (free_b, arr[-1], arr[-2::-1])):
            if not free:
                continue
            far = next((q for q in rest if np.hypot(*(q - tip)) >= stroke * 1.2), rest[-1])
            vy, vx = tip - far                           # 由內向外
            n = math.hypot(vx, vy)
            if n > 1e-6 and abs(vy) < 0.45 * abs(vx):
                ends.append(((float(tip[0]), float(tip[1])), (vx / n, vy / n)))
    return ends


def split_horizontal(polys, stroke: float):
    """把折線拆成「橫向線段」和「其餘筆畫」。橫向 = 偏離水平約 24° 以內、且夠長。"""
    rest, hsegs = [], []
    for arr, flags in polys:
        if len(arr) == 1:                               # 單點(輕聲點等)原樣保留
            rest.append((arr, flags))
            continue
        cur = [arr[0]]
        for p, q in zip(arr[:-1], arr[1:]):
            dy, dx = q - p
            if abs(dy) < 0.45 * abs(dx) and math.hypot(dx, dy) >= stroke * 1.5:
                if len(cur) > 1:
                    rest.append((np.array(cur), flags))
                hsegs.append((p, q))
                cur = [q]
            else:
                cur.append(q)
        if len(cur) > 1:
            rest.append((np.array(cur), flags))
    return rest, _merge_collinear(hsegs, stroke)


def _merge_collinear(hsegs, stroke: float):
    """首尾相接、仍然接近水平的橫線段合併成一條(例如被豎筆畫從中間切開的橫槓)。"""
    segs = [(p, q) if p[1] <= q[1] else (q, p) for p, q in hsegs]
    tol = stroke * 0.5
    merged = True
    while merged:
        merged = False
        for i in range(len(segs)):
            for j in range(len(segs)):
                if i == j:
                    continue
                (p1, q1), (p2, q2) = segs[i], segs[j]
                if np.hypot(*(q1 - p2)) <= tol:
                    dy, dx = q2 - p1
                    if abs(dy) < 0.45 * abs(dx):
                        segs[i] = (p1, q2)
                        segs.pop(j)
                        merged = True
                        break
            if merged:
                break
    return segs


ARC_THICK = 1.3     # 彎曲橫筆畫的粗度(單位:筆畫粗),全長一致
ARC_RISE = 0.9      # 彎曲幅度(單位:筆畫粗);長橫筆畫最多不超過長度的 16%


def arc_mask(hsegs, stroke: float, shape, direction: str = "up") -> np.ndarray:
    """橫筆畫 → 輕微彎曲、粗細一致、兩端圓頭。

    direction: "up" 中間向上拱(∩);"down" 兩端向上翹(∪)。
    """
    im = Image.new("L", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(im)
    w = stroke * ARC_THICK
    sign = 1.0 if direction == "up" else -1.0
    for p, q in hsegs:
        if q[1] < p[1]:
            p, q = q, p
        px, py, qx, qy = p[1], p[0], q[1], q[0]
        ln = math.hypot(qx - px, qy - py)
        tx, ty = (qx - px) / ln, (qy - py) / ln
        nx, ny = ty, -tx                              # 垂直於筆畫、朝上
        rise = min(stroke * ARC_RISE, ln * 0.16) * sign
        n = max(16, int(ln / 4))
        pts = []
        for i in range(n + 1):
            u = i / n
            off = rise * math.sin(math.pi * u)        # 兩端為 0、中間最高
            pts.append((px + tx * ln * u + nx * off, py + ty * ln * u + ny * off))
        d.line(pts, fill=255, width=1)
    # 先畫 1px 中心線,再統一以圓形膨脹:邊緣平滑、粗細一致、兩端自然是圓頭
    return _dilate(np.asarray(im) > 0, w / 2)


def _trim_to_balls(out: np.ndarray, balls, reach: float) -> np.ndarray:
    """寬筆尖的刀口會比筆畫末端多伸出最多 reach;把末端圓球外、伸出去的那一截修掉。

    只修末端往外的扇形範圍,避免削到旁邊別的筆畫。
    """
    H, W = out.shape
    out = out.copy()
    for x, y, r, u in balls:
        pad = max(r, reach) + 3
        x0, x1 = max(0, int(x - pad)), min(W, int(x + pad) + 1)
        y0, y1 = max(0, int(y - pad)), min(H, int(y + pad) + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rx, ry = xx - x, yy - y
        dist = np.hypot(rx, ry)
        cut = (dist > r) & (dist <= pad)
        if u is not None:
            ux, uy = u
            along = rx * ux + ry * uy
            cut &= along > dist * 0.5             # 只修末端往外 ±60° 的扇形
        out[y0:y1, x0:x1] &= ~cut
    return out


def finalize(skel: np.ndarray, stroke: float, merge: float = 0.75,
             nib: float | None = None, angular: bool = False,
             feet: bool | str = False, all_ends: bool = False,
             ball: float | None = None, round_h: bool = False,
             arc: str | None = None) -> np.ndarray:
    """中心線 → 固定規則的字。

    1. 先把彼此很近(< merge×筆畫粗)的中心線膨脹再細化:同一筆畫被疊成兩條略有位移的線,
       會合併成一條,不會再出現「一處粗一處細」。
    2. 剪掉骨架化產生的短毛刺。
    3. angular:把彎曲的筆畫折成直線段;feet:在垂直筆畫的末端加腳;
       arc:橫向筆畫改成輕微彎曲、粗細一致的圓頭筆畫;
       round_h:橫向筆畫的斜切末端改成圓頭(僅寬筆尖模式)。
    4. 等粗圓頭筆畫;若指定 nib(筆尖角度),改用寬筆尖掃出有粗細對比的筆畫。
    """
    near = _dilate(skel, stroke * merge)
    center = prune(skeletonize(near), int(stroke * 0.9))
    extra = None
    round_ends = None
    balls = []
    if angular or feet or round_h or arc:
        polys = skeleton_polylines(center, stroke * (0.7 if angular else 1.5))
        center = rasterize(polys, center.shape)
        if round_h and nib is not None and not arc:
            round_ends = horizontal_ends(polys, stroke)
        if feet:
            extra = feet_mask(polys, stroke, center.shape, "diamond" if feet is True else feet,
                              all_ends=all_ends, r_mult=ball, balls=balls)
            if balls:
                # 骨架在末端分岔留下的短殘枝:整條都落在某顆圓球附近,就不畫,免得從圓球旁邊戳出來
                def near_ball(arr):
                    return any(np.hypot(arr[:, 1] - bx, arr[:, 0] - by).max() <= br + stroke * 0.8
                               for bx, by, br, _ in balls)
                polys = [(arr, fl) for arr, fl in polys if len(arr) < 2 or not near_ball(arr)]
                center = rasterize(polys, center.shape)
        if arc:
            rest, hsegs = split_horizontal(polys, stroke)
            center = rasterize(rest, center.shape)
            dm = arc_mask(hsegs, stroke, center.shape, arc)
            extra = dm if extra is None else (extra | dm)
    if nib is not None:
        return nib_stamp(center, stroke, nib, extra=extra, round_ends=round_ends, balls=balls)
    ink = _dilate(center, stroke / 2)
    if extra is not None:
        ink = ink | extra
    return ink.astype(np.float32)


# ── 4. 五種拼法 ───────────────────────────────────────────────────────────────
def _sub(main, fx0, fy0, fx1, fy1):
    X0, Y0, X1, Y1 = main
    w, h = X1 - X0, Y1 - Y0
    return (X0 + fx0 * w, Y0 + fy0 * h, X0 + fx1 * w, Y0 + fy1 * h)


OVERLAP = 0.13   # 相鄰符號各往對方擠進的比例(佔主體邊長);越大交疊越多,越不像並排


def layout_lr(syms, main):
    """左右:3 個以內並排;4 個以上改成高低交錯的鋸齒,避免擠成一排細直條。"""
    n = len(syms)
    step = 1 / n
    ov = OVERLAP
    boxes = []
    for i in range(n):
        x0, x1 = max(0.0, i * step - ov), min(1.0, (i + 1) * step + ov)
        if n <= 3:
            y0, y1 = 0.0, 1.0
        elif i % 2 == 0:
            y0, y1 = 0.0, 0.72
        else:
            y0, y1 = 0.28, 1.0
        boxes.append(_sub(main, x0, y0, x1, y1))
    return boxes


def layout_pin(syms, main):
    """品字:2 個斜對角,3 個品字形,4 個田字,5 個梅花,6 個以上排成格子。"""
    n = len(syms)
    ov = OVERLAP
    if n == 1:
        return [_sub(main, 0, 0, 1, 1)]
    if n == 2:
        return [_sub(main, 0, 0, 0.5 + ov * 1.4, 0.5 + ov * 1.4),
                _sub(main, 0.5 - ov * 1.4, 0.5 - ov * 1.4, 1, 1)]
    if n == 3:
        return [_sub(main, 0.16, 0, 0.84, 0.5 + ov),
                _sub(main, 0, 0.5 - ov, 0.5 + ov, 1),
                _sub(main, 0.5 - ov, 0.5 - ov, 1, 1)]
    if n == 5:      # 四角 + 中心
        c = [(0, 0), (1, 0), (0, 1), (1, 1)]
        boxes = [_sub(main, 0 if cx == 0 else 0.5 - ov, 0 if cy == 0 else 0.5 - ov,
                      0.5 + ov if cx == 0 else 1, 0.5 + ov if cy == 0 else 1) for cx, cy in c]
        return boxes + [_sub(main, 0.25, 0.25, 0.75, 0.75)]
    cols = 2 if n == 4 else 3
    rows = -(-n // cols)
    boxes = []
    for r in range(rows):
        k = min(cols, n - r * cols)             # 最後一列不滿時置中
        for c in range(k):
            cw = 1 / cols
            x0 = (c + (cols - k) / 2) * cw
            y0 = r / rows
            boxes.append(_sub(main, max(0, x0 - ov / 2), max(0, y0 - ov / 2),
                              min(1, x0 + cw + ov / 2), min(1, y0 + 1 / rows + ov / 2)))
    return boxes


def layout_han(syms, main):
    """漢字式:第一個符號當偏旁佔左側,其餘遞迴地在剩下的空間裡上下、左右交替切分。"""
    ov = OVERLAP

    def peel(n, axis, x0, y0, x1, y1):
        if n == 1:
            return [(x0, y0, x1, y1)]
        sp = 0.5 if n == 2 else 0.42
        w, h = x1 - x0, y1 - y0
        if axis == 0:
            a = (x0, y0, x0 + w * (sp + ov), y1)
            b = (x0 + w * (sp - ov), y0, x1, y1)
        else:
            a = (x0, y0, x1, y0 + h * (sp + ov))
            b = (x0, y0 + h * (sp - ov), x1, y1)
        return [a] + peel(n - 1, 1 - axis, *b)

    return [_sub(main, *b) for b in peel(len(syms), 0, 0, 0, 1, 1)]


def layout_overlay(syms, main):
    """疊合:3 個以內全部疊在同一格;4 個以上分兩疊,沿對角線錯開,避免整團糊掉。"""
    n = len(syms)
    if n <= 3:
        return [_sub(main, 0.02, 0.02, 0.98, 0.98) for _ in syms]
    half = (n + 1) // 2
    a = _sub(main, 0.0, 0.0, 0.78, 0.78)
    b = _sub(main, 0.22, 0.22, 1.0, 1.0)
    return [a if i < half else b for i in range(n)]


def layout_enclose(syms, main):
    """包圍:第一個當外框,其餘擠成一團放在框內。

    內部要夠大:以前 3 個符號時是同心縮小(最裡面只剩 0.28),內部符號小到看不出來,
    整個字看起來就只剩外框那一個符號。
    """
    n = len(syms)
    if n == 1:
        return [_sub(main, 0, 0, 1, 1)]
    if n == 2:
        return [_sub(main, 0, 0, 1, 1), _sub(main, 0.2, 0.2, 0.8, 0.8)]
    inner = _sub(main, 0.12, 0.12, 0.88, 0.88)
    return [_sub(main, 0, 0, 1, 1)] + layout_pin(syms[1:], inner)


def layout_ring(syms, main):
    """環列:符號繞著中心排成一圈,彼此交疊。"""
    n = len(syms)
    if n == 1:
        return [_sub(main, 0, 0, 1, 1)]
    sz = 0.58 if n <= 4 else 0.5 if n <= 6 else 0.44
    r = 0.5 - sz / 2
    start = -90 + (45 if n % 2 == 0 else 0)
    out = []
    for i in range(n):
        a = math.radians(start + 360 * i / n)
        cx, cy = 0.5 + r * math.cos(a), 0.5 + r * math.sin(a)
        out.append(_sub(main, cx - sz / 2, cy - sz / 2, cx + sz / 2, cy + sz / 2))
    return out


STYLES = {
    "左右": layout_lr,
    "品字": layout_pin,
    "漢字式": layout_han,
    "疊合": layout_overlay,
    "包圍": layout_enclose,
    "環列": layout_ring,
}

STROKE = 0.048      # 全字統一筆畫粗度(佔畫布邊長的比例)
SUPERSAMPLE = 2     # 內部用 2 倍解析度畫,再縮回,邊緣較平滑


# ── 5. 組字 ───────────────────────────────────────────────────────────────────
TONE_BALL = 0.95    # 聲調末端圓球的半徑(單位:筆畫粗)


def place_tone(skel: np.ndarray, tone: str, stroke: float, rng: random.Random,
               target: np.ndarray | None = None):
    """聲調符號可出現在畫布任何地方:優先挑不碰到任何筆畫的空位,隨機取一處。

    字形放進 target(省略則放進 skel);挑位置時會避開 skel 與 target 上的所有筆畫。
    """
    W = skel.shape[0]
    t = render_symbol(tone)
    size = 0.12 if tone == "\u02d9" else 0.17          # 輕聲點較小
    tw = th = int(size * W)
    if tone != "\u02d9":
        th = int(size * 1.4 * W)                       # ˊˇˋ 略高
    occupied = skel if target is None else (skel | target)
    ink = _dilate(occupied, stroke * 1.3)              # 筆畫外加一圈留白
    step = max(4, W // 24)
    margin = int(0.03 * W)
    cands = []
    for y in range(margin, W - th - margin, step):
        for x in range(margin, W - tw - margin, step):
            cands.append((float(ink[y:y + th, x:x + tw].mean()), x, y))
    clear = [c for c in cands if c[0] == 0.0]
    if clear:
        _, x, y = rng.choice(clear)
    else:
        lo = min(c[0] for c in cands)
        _, x, y = rng.choice([c for c in cands if c[0] <= lo + 0.01])
    place(skel if target is None else target, t,
          (x / W, y / W, (x + tw) / W, (y + th) / W), stroke, stretch=0.0)


def compose(syllable: str, style: str, S: int = 240, nib: float | None = None,
            angular: bool = False, feet: bool | str = False,
            tone_circle: bool = False, round_h: bool = False,
            arc: str | None = None, rotations=None) -> np.ndarray:
    """回傳 S×S float 陣列,1=墨。全字筆畫粗度一致;聲調位置由 (音節, 拼法) 決定,可重現。

    rotations:每個符號逆時針轉幾個 90°(0~3),依符號順序;省略則都不轉。

    tone_circle:聲調字形不變(ˊ ˇ ˋ ˙ 照原樣寫),只在每個末端加實心圓球。
    """
    p = parse(syllable)
    layout = STYLES[style]
    syms = [render_symbol(ch) for _, ch in p["units"]]
    if rotations:
        syms = [np.rot90(sym, int(r) % 4) for sym, r in zip(syms, list(rotations) + [0] * len(syms))]
    main = (0.08, 0.08, 0.92, 0.92)

    W = S * SUPERSAMPLE
    stroke = STROKE * W
    skel = np.zeros((W, W), dtype=bool)
    for sym, box in zip(syms, layout(syms, main)):
        place(skel, sym, box, stroke)
    tskel = np.zeros((W, W), dtype=bool) if tone_circle else None
    for k, tone in enumerate(p["tones"]):
        place_tone(skel, tone, stroke, random.Random(f"{syllable}|{style}|{k}"), target=tskel)

    ink = finalize(skel, stroke, nib=nib, angular=angular, feet=feet, round_h=round_h,
                   arc=arc)
    if tone_circle and tskel.any():
        t_ink = finalize(tskel, stroke, nib=nib, angular=angular, feet="circle",
                         all_ends=True, ball=TONE_BALL)
        ink = np.maximum(ink, t_ink)
    img = Image.fromarray((ink * 255).astype(np.uint8)).resize((S, S), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


# ── 6. 排版輸出 ───────────────────────────────────────────────────────────────
GALLERY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gallery")


def out_path(p: str) -> str:
    """只給檔名(沒有資料夾)時,輸出放進 gallery/;給了路徑就照路徑。"""
    if os.path.dirname(p):
        return p
    os.makedirs(GALLERY, exist_ok=True)
    return os.path.join(GALLERY, p)


def contact_sheet(syllables, cell=200, out="zhuyin_demo.png", **opts):
    names = list(STYLES)
    head_h, pad = 44, 8
    f_lab = _load_font(LABEL_FONT_CANDIDATES, 22)
    lab_w = int(max(120, max(f_lab.getlength(x) for x in syllables) + 24))
    W = lab_w + len(names) * (cell + pad) + pad
    H = head_h + len(syllables) * (cell + pad) + pad
    sheet = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(sheet)
    f_head = _load_font(LABEL_FONT_CANDIDATES, 22)

    for j, n in enumerate(names):
        d.text((lab_w + pad + j * (cell + pad) + cell // 2, head_h // 2), n,
               fill=60, font=f_head, anchor="mm")
    for i, syl in enumerate(syllables):
        y = head_h + pad + i * (cell + pad)
        d.text((lab_w // 2, y + cell // 2), syl, fill=0, font=f_lab, anchor="mm")
        for j, n in enumerate(names):
            x = lab_w + pad + j * (cell + pad)
            g = compose(syl, n, S=cell, **opts)
            tile = Image.fromarray(((1 - g) * 255).astype(np.uint8))
            sheet.paste(tile, (x, y))
            d.rectangle([x, y, x + cell - 1, y + cell - 1], outline=210)
    out = out_path(out)
    sheet.save(out)
    return out


DEFAULT = ["ㄇㄠ", "ㄒㄧㄠˇ", "ㄏㄨㄢ", "ㄕㄨㄟˇ", "ㄌㄧㄤˊ", "ㄩㄝˋ"]
# 多個音節合成一個字(4~6 個符號)
DEFAULT_MULTI = ["ㄇㄠ+ㄇㄠ", "ㄅㄚ+ㄇㄚ˙", "ㄒㄧㄠˇ+ㄌㄧㄤˊ", "ㄏㄨㄢ+ㄩㄝˋ",
                 "ㄕㄨㄟˇ+ㄌㄧㄥˊ", "ㄐㄧㄣ+ㄔㄨㄤˊ"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("syllables", nargs="*", help="注音音節,例如 ㄒㄧㄠˇ;多音節用 + 連接")
    ap.add_argument("--random", type=int, default=0, help="隨機產生 N 個音節")
    ap.add_argument("--parts", type=int, default=1, help="--random 時,每個字由幾個音節組成")
    ap.add_argument("--multi", action="store_true", help="用內建的多音節範例")
    ap.add_argument("--nib", type=float, nargs="?", const=30.0, default=None,
                    help="寬筆尖(歌德體式)筆畫;可給筆尖角度(度),預設 30")
    ap.add_argument("--angular", action="store_true", help="把彎筆畫折成直線段(折角)")
    ap.add_argument("--feet", nargs="?", const="diamond", default=None,
                    choices=["diamond", "circle"], help="垂直筆畫末端加腳:diamond 菱形(預設)/ circle 圓形")
    ap.add_argument("--tone-circle", action="store_true",
                    help="聲調字形照原樣,只在每個末端加實心圓球")
    ap.add_argument("--round-h", action="store_true", help="橫向筆畫的斜切末端改成圓頭(需搭配 --nib/--gothic)")
    ap.add_argument("--arc", nargs="?", const="up", default=None, choices=["up", "down"],
                    help="橫筆畫輕微彎曲、粗細一致:up 中間上拱(預設)/ down 兩端上翹")
    ap.add_argument("--no-tone", action="store_true", help="拿掉所有聲調符號")
    ap.add_argument("--gothic", action="store_true", help="歌德體:寬筆尖 + 折角 + 尖腳")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cell", type=int, default=200)
    ap.add_argument("--out", default="zhuyin_demo.png")
    a = ap.parse_args()

    syls = list(a.syllables)
    if a.random:
        rng = random.Random(a.seed)
        syls += [random_word(rng, a.parts) for _ in range(a.random)]
    if not syls:
        syls = DEFAULT_MULTI if a.multi else DEFAULT
    if a.no_tone:
        syls = ["".join(ch for ch in x if ch not in TONES) for x in syls]
    print("音節:", " ".join(syls))
    nib = a.nib if a.nib is not None else (30.0 if a.gothic else None)
    print("輸出:", contact_sheet(syls, cell=a.cell, out=a.out, nib=nib,
                               angular=a.angular or a.gothic or bool(a.arc),
                               feet=a.feet or ("diamond" if a.gothic else False),
                               tone_circle=a.tone_circle, round_h=a.round_h, arc=a.arc))


if __name__ == "__main__":
    main()

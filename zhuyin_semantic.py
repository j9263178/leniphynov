"""
語意 → 注音造字

    文字 → 語意向量(Gemini 256 維,與 lenia_text 共用快取)→ 注音音節 + 拼法 → 字形

沿用 lenia_text / attractor_glyph 的設計:
  - 向量來自 lenia_text._embed(有 embed_cache.json 快取;沒有 API 時退回穩定的雜湊向量)。
  - 向量先減去平均、投到前 32 個主成分並白化(zhuyin_basis.npz,固定)。
  - direct 模式(預設):每個分量直接決定字形的一個面向:排法、符號數(3~4)、重複程度、各個符號。
  - radical 模式:先找最近的語意錨點當「部首」音節,再加一個個別音節。

run:
    python zhuyin_semantic.py                      # 內建範例(都是快取裡有的詞)
    python zhuyin_semantic.py fire water dog --chosen
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import numpy as np
from PIL import Image, ImageDraw

import zhuyin_glyph as Z

try:
    import lenia_text as _L
except Exception:
    _L = None

DIM = 256
_NONE = ""
OPT_INITIAL = [_NONE] + list(Z.INITIALS)          # 聲母可以沒有
OPT_MEDIAL = [_NONE] + list(Z.MEDIALS)            # 介音可以沒有
OPT_FINAL = list(Z.FINALS)
OPT_PARTS = [1, 2]                                # 一個字由幾個音節組成
STYLE_NAMES = list(Z.STYLES)

# 預設筆畫風格:目前定下來的那一套
LOOK = dict(nib=30.0, angular=True, feet="circle", arc="up")


_HERE = os.path.dirname(os.path.abspath(__file__))
_BASIS_PATH = os.path.join(_HERE, "zhuyin_basis_v2.npz")
_BASIS = None
K = 32          # 只保留前 32 個主成分


def _basis():
    """語意向量的「座標系」:共同方向(平均)+ 前 K 個主成分(白化)。

    Gemini 的向量彼此都很像(有一個很強的共同成分),直接投影的話所有文字都會選到同樣的符號。
    所以先減掉平均,再投到前 K 個主成分並把各成分拉到同樣的尺度:
    相近的詞(火/fire、水/water)相似度約 0.65~0.7,無關的詞約 0.15,差距最明顯。
    另外先去掉兩個「跟意思無關」的方向:語言(英文 vs 中文)和長度(單字 vs 詞句)。
    不去掉的話,前幾個主成分幾乎都在分語言和長度,fire 和 火 會被分到很遠的地方。
    座標系第一次從 embed_cache.json 算出後存成 zhuyin_basis_v2.npz,之後固定使用,
    快取再增加也不會讓已造好的字改變。
    """
    global _BASIS
    if _BASIS is None:
        if os.path.exists(_BASIS_PATH):
            f = np.load(_BASIS_PATH)
            _BASIS = (f["mean"], f["comp"], f["scale"], f["nuis"])
        else:
            try:
                cache = json.load(open(os.path.join(_HERE, "embed_cache.json"), encoding="utf-8"))
            except Exception:
                return None
            import re
            items = [(k, np.asarray(v, dtype=np.float64)) for k, v in cache.items()
                     if not k.startswith("[ANCHOR]") and len(v) == DIM]
            if len(items) < 2 * K:
                return None
            cjk = lambda k: re.search(r"[一-鿿]", k) is not None
            latin = [v for k, v in items if re.fullmatch(r"[A-Za-z][A-Za-z ]+", k)]
            zh1 = [v for k, v in items if cjk(k) and len(k) == 1]
            zhn = [v for k, v in items if cjk(k) and len(k) > 1]
            dirs = []
            if latin and zh1:
                dirs.append(np.mean(latin, 0) - np.mean(zh1, 0))      # 語言
            if zh1 and zhn:
                dirs.append(np.mean(zhn, 0) - np.mean(zh1, 0))        # 長度
            nuis, _ = np.linalg.qr(np.array(dirs).T) if dirs else (np.zeros((DIM, 0)), None)
            X = np.array([v for _, v in items])
            mean = X.mean(axis=0)
            Xc = X - mean
            Xc = Xc - (Xc @ nuis) @ nuis.T
            _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            _BASIS = (mean, Vt[:K], S[:K], nuis)
            np.savez(_BASIS_PATH, mean=mean, comp=Vt[:K], scale=S[:K], nuis=nuis)
    return _BASIS


def _embed_retry(text: str, tries: int = 4):
    """先查快取;沒有才呼叫 API,遇到暫時性網路錯誤會重試。

    lenia_text._embed 第一次失敗後就整個關掉語意模式(連快取都不查),
    一次網路抖動就會讓整篇文章退回雜湊,所以這裡自己處理。
    """
    if _L is None:
        return None
    hit = _L._EMBED_CACHE.get(text)
    if hit is not None:
        return np.asarray(hit, dtype=np.float64)
    if not _L._GEMINI_KEY:
        return None
    import time
    for k in range(tries):
        _L._SEMANTIC_OK = True
        v = _L._embed(text)
        if v is not None:
            return v
        time.sleep(1.5 * (k + 1))
    return None


def text_vector(text: str) -> tuple[np.ndarray, bool]:
    """回傳 (K 維單位向量, 是否為真正的語意向量)。"""
    v = _embed_retry(text)
    if v is not None:
        b = _basis()
        if b is not None:
            mean, comp, scale, nuis = b
            x = np.asarray(v, dtype=np.float64) - mean
            x = x - (x @ nuis) @ nuis.T
            z = comp @ x / scale
            return z / (np.linalg.norm(z) + 1e-12), True
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    z = np.random.default_rng(seed).standard_normal(K)
    return z / np.linalg.norm(z), False


def _proj(name: str, n: int) -> np.ndarray:
    """每個選擇項目一個固定投影矩陣(以名稱為種子,改程式不會互相影響)。"""
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
    return np.random.default_rng(seed).standard_normal((K, n))


def _pick(v: np.ndarray, name: str, options: list):
    return options[int(np.argmax(v @ _proj(name, len(options))))]


# ── 部首:跟 lenia_text 的 ANCHORS 同一個想法 ─────────────────────────────────
# 每個部首是一個「語意錨點」(用快取裡已有向量的詞),配一個部首音節與固定拼法。
# 文字先找語意上最近的錨點 → 得到部首(字的第一個音節,在漢字式裡就是左邊的偏旁),
# 意思相近的詞落在同一個部首,共用同一個偏旁和拼法,看起來是同一家族的字。
RADICALS = [
    # (錨點文字, 顯示名, 部首音節, 拼法)
    ("火", "火", "ㄏㄨㄛ", "漢字式"),
    ("水", "水", "ㄕㄨㄟ", "左右"),
    ("石", "石", "ㄕ", "包圍"),
    ("月", "月", "ㄩㄝ", "品字"),
    ("日", "日", "ㄖ", "包圍"),
    ("山", "山", "ㄕㄢ", "品字"),
    ("人", "人", "ㄖㄣ", "漢字式"),
    ("草", "草", "ㄘㄠ", "左右"),
    ("cat", "獸", "ㄇㄠ", "漢字式"),
    ("光", "光", "ㄍㄨㄤ", "環列"),
    ("Darkness", "暗", "ㄢ", "疊合"),
    ("hope", "心", "ㄒㄧㄣ", "疊合"),
    ("星辰", "星", "ㄒㄧㄥ", "環列"),
    ("城市", "城", "ㄔㄥ", "包圍"),
    ("記憶", "憶", "ㄧ", "品字"),
    ("quantum mechanics", "理", "ㄌㄧ", "環列"),
]
_RAD_VECS = None


def _radical_vecs():
    """部首錨點的向量;任何一個錨點沒有語意向量就回傳 None(退回純投影)。"""
    global _RAD_VECS
    if _RAD_VECS is None:
        vecs = []
        for anchor, *_ in RADICALS:
            v, real = text_vector(anchor)
            if not real:
                return None
            vecs.append(v)
        _RAD_VECS = np.array(vecs)
    return _RAD_VECS


def _syllable(v: np.ndarray, tag: str) -> str:
    return (_pick(v, f"initial{tag}", OPT_INITIAL) + _pick(v, f"medial{tag}", OPT_MEDIAL)
            + _pick(v, f"final{tag}", OPT_FINAL))


# ── 直接模式:向量的每個分量直接決定字形的一個面向(不走音節/部首的漢字邏輯)───────
# 分量依變異量排序(第 0 個最能區分文字),越前面的分量決定越顯眼的面向:
#   z0 → 排法    z1 → 符號數(3/4)    z2 → 重複程度(至少兩種不同符號)    z3 → 重複的位置
#   z4..z7 → 每個不同的符號
# 每個分量先換成 0~1 的均勻值 u(常態 CDF),再切成等寬的格子。分量是連續的,
# 意思相近的詞分量也相近,大多落在同一格;只有剛好在格子邊界附近才會換。
# 符號表依「外形」排序,相鄰的格子是長得像的符號,所以就算換格,也多半換成相似的符號。
from math import erf, sqrt
ALL_SYMBOLS = Z.INITIALS + Z.MEDIALS + Z.FINALS
_SYMBOL_ORDER = None


def symbol_order() -> str:
    """把 37 個注音符號排成一列,相鄰的符號外形最像(貪婪最近鄰鏈)。"""
    global _SYMBOL_ORDER
    if _SYMBOL_ORDER is None:
        feats = []
        for ch in ALL_SYMBOLS:
            a = Z.render_symbol(ch)
            im = Image.fromarray((a * 255).astype(np.uint8)).resize((24, 24), Image.LANCZOS)
            f = np.asarray(im, dtype=np.float64).ravel()
            feats.append((f - f.mean()) / (f.std() + 1e-9))
        F = np.array(feats)
        D = np.linalg.norm(F[:, None] - F[None], axis=2)
        _, _, Vt = np.linalg.svd(F - F.mean(0), full_matrices=False)
        cur = int(np.argmin(F @ Vt[0]))                # 從第一主成分的一端開始
        order, left = [cur], set(range(len(F))) - {cur}
        while left:
            cur = min(left, key=lambda j: D[cur, j])
            order.append(cur)
            left.remove(cur)
        _SYMBOL_ORDER = "".join(ALL_SYMBOLS[i] for i in order)
    return _SYMBOL_ORDER


def _u(z: float) -> float:
    return 0.5 * (1 + erf(z / sqrt(2)))


_CALIB_PATH = os.path.join(_HERE, "zhuyin_calib.npz")
_CALIB = None


def _calib():
    """每個分量在「真正的詞」(中文字詞、英文單字)上的分布,排好序存起來。

    快取裡混了很多單一字母和標點,它們的分布和真正的詞不一樣;用常態 CDF 切格子的話,
    真正的詞會擠在中間幾格(例如排法幾乎都是疊合)。改用真正的詞的排名來切,每一格才會平均。
    第一次算完存成 zhuyin_calib.npz,之後固定。
    """
    global _CALIB
    if _CALIB is None:
        if os.path.exists(_CALIB_PATH):
            _CALIB = np.load(_CALIB_PATH)["ref"]
        else:
            import re
            cache = json.load(open(os.path.join(_HERE, "embed_cache.json"), encoding="utf-8"))
            words = [k for k in cache if not k.startswith("[ANCHOR]") and
                     (re.search(r"[一-鿿]", k) or re.fullmatch(r"[A-Za-z][A-Za-z ]+", k))]
            Zs = np.array([text_vector(k)[0] * sqrt(K) for k in words])
            _CALIB = np.sort(Zs, axis=0)
            np.savez(_CALIB_PATH, ref=_CALIB)
    return _CALIB


def _uniform(z: np.ndarray, real: bool) -> list:
    """分量 → 0~1 的均勻值。真正的語意向量用詞的排名;雜湊向量用常態 CDF。"""
    if real:
        ref = _calib()
        n = len(ref)
        return [(np.searchsorted(ref[:, i], z[i]) + 0.5) / (n + 1) for i in range(len(z))]
    return [_u(x) for x in z]


def direct_glyph(v: np.ndarray, real: bool = True) -> tuple[str, str, str, list]:
    """向量 → (符號串, 排法, 說明, 每格旋轉)。"""
    z = v * sqrt(len(v))                               # 單位向量 → 各分量約為標準常態
    u = _uniform(z, real)
    bin_ = lambda x, n: min(n - 1, int(x * n))
    style = STYLE_NAMES[bin_(u[0], len(STYLE_NAMES))]
    n = 3 + bin_(u[1], 2)                              # 3 或 4 個符号
    distinct = n - bin_(u[2], n - 1)                   # n..2 個不同的符號(至少兩種,
                                                       # 只有一種的話疊合時會看起來像單一符號)
    order = symbol_order()
    syms = []
    for k in range(distinct):
        i = bin_(u[4 + k], len(order))
        while order[i] in syms:                        # 不同的符號不要撞在一起
            i = (i + 1) % len(order)
        syms.append(order[i])
    seq = list(syms)
    for k in range(n - distinct):                      # 補上重複的符號
        src = seq[bin_(u[3], len(seq))]
        pos = bin_((u[3] * 7 + k * 0.37) % 1.0, len(seq) + 1)
        seq.insert(pos, src)
    seq = "".join(seq)
    # 有些組合疊起來會被看成單一符號(例如 ㄏ 疊在 ㄕ 上,ㄏ 整個被 ㄕ 蓋住)。
    # 從向量選的排法開始依序試,挑第一個「不像任何單一符號」的排法。
    k0 = STYLE_NAMES.index(style)
    tried = []
    for k in range(len(STYLE_NAMES)):
        st = STYLE_NAMES[(k0 + k) % len(STYLE_NAMES)]
        sc = single_likeness(seq, st)
        if sc < SINGLE_MAX:
            style = st
            break
        tried.append((sc, st))
    else:
        style = min(tried)[1]
    return seq, style, f"{n}符 {distinct}種", None   # 試過逐格轉 90°,不好看,不轉


SINGLE_MAX = 0.55       # 和某個單一符號的相似度(IoU)超過這個,就算「看起來像單一符號」
_SINGLES = None


def single_likeness(seq: str, style: str, cell: int = 64) -> float:
    """字形和 37 個單一符號(各自撐滿整格)最像的那個的相似度(稍微加粗後的 IoU)。"""
    global _SINGLES
    from scipy import ndimage
    fat = lambda m: ndimage.binary_dilation(m, iterations=2)
    if _SINGLES is None:
        _SINGLES = [fat(Z.compose(ch, "左右", S=cell, **LOOK) > 0.5) for ch in ALL_SYMBOLS]
    g = fat(Z.compose(seq, style, S=cell, **LOOK) > 0.5)
    return max(float((g & m).sum()) / max(1, (g | m).sum()) for m in _SINGLES)


MODE = "direct"


def vector_to_glyph(v: np.ndarray, real: bool = True) -> tuple[str, str, str | None, list | None]:
    """向量 → (注音字串, 拼法, 部首名)。

    有語意向量:部首音節 + 個別音節(由「去掉部首方向後剩下的差異」決定),拼法跟著部首。
    沒有(雜湊向量):兩個音節和拼法都由固定投影決定,沒有部首。
    """
    if MODE == "direct":
        return direct_glyph(v, real)
    R = _radical_vecs() if real else None
    if R is None:
        parts = _pick(v, "parts", OPT_PARTS)
        sylls = [_syllable(v, str(k)) for k in range(parts)]
        return "+".join(sylls), _pick(v, "style", STYLE_NAMES), None, None
    i = int(np.argmax(R @ v))
    _, name, rad_syl, style = RADICALS[i]
    resid = v - (v @ R[i]) * R[i]
    resid = resid / (np.linalg.norm(resid) + 1e-12)
    return rad_syl + "+" + _syllable(resid, "0"), style, name, None


def sheet(words, cell=200, out="zhuyin_semantic.png", all_styles=True):
    """每列一個詞。all_styles=True 時列出所有拼法,並框出向量選中的那一種。"""
    rows = []
    for w in words:
        v, real = text_vector(w)
        syl, style, rad, rots = vector_to_glyph(v, real)
        rows.append((w, syl, style, real, rad, rots))
        tag = (f"[{rad}]" if rad else "") + ("" if real else " (無語意向量,用雜湊)")
        print(f"{w:<10} → {tag:<6} {syl:<12} {style}")

    cols = STYLE_NAMES if all_styles else ["語意選擇"]
    head_h, pad = 44, 8
    f_lab = Z._load_font(Z.LABEL_FONT_CANDIDATES, 24)
    f_sub = Z._load_font(Z.LABEL_FONT_CANDIDATES, 18)
    def sub(r):
        return (f"[{r[4]}] " if r[4] else "") + r[1]
    lab_w = int(max(150, max(max(f_lab.getlength(r[0]), f_sub.getlength(sub(r))) for r in rows) + 30))
    W = lab_w + len(cols) * (cell + pad) + pad
    H = head_h + len(rows) * (cell + pad) + pad
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    for j, c in enumerate(cols):
        d.text((lab_w + pad + j * (cell + pad) + cell // 2, head_h // 2), c,
               fill=60, font=f_sub, anchor="mm")
    for i, r in enumerate(rows):
        w, syl, style, real, rad, rots = r
        y = head_h + pad + i * (cell + pad)
        d.text((lab_w // 2, y + cell // 2 - 14), w, fill=0, font=f_lab, anchor="mm")
        d.text((lab_w // 2, y + cell // 2 + 18), sub(r), fill=110 if real else 170,
               font=f_sub, anchor="mm")
        for j, c in enumerate(cols):
            st = style if c == "語意選擇" else c
            x = lab_w + pad + j * (cell + pad)
            g = Z.compose(syl, st, S=cell, rotations=rots, **LOOK)
            img.paste(Image.fromarray(((1 - g) * 255).astype(np.uint8)), (x, y))
            chosen = st == style
            d.rectangle([x, y, x + cell - 1, y + cell - 1],
                        outline=0 if chosen and all_styles else 210, width=3 if chosen and all_styles else 1)
    out = Z.out_path(out)
    img.save(out)
    return out


# ── 文章排版 ─────────────────────────────────────────────────────────────────
import re as _re
_TOKEN = _re.compile(r"[一-鿿]|[A-Za-z]+(?:'[A-Za-z]+)?|\n|[^\sA-Za-z一-鿿]")
_PUNCT_STOP = set("。.!?！？")


def tokenize(text: str):
    """中文一字一個、英文一詞一個;標點與換行另外處理。"""
    return _TOKEN.findall(text)


def article(text: str, cell: int = 110, cols: int = 10, out: str = "zhuyin_article.png",
            gloss: bool = True):
    """把一段文章排成造字的頁面:每個字/詞一個字形,標點畫成小圓點,換行即分段。"""
    toks = tokenize(text)
    glyphs = {}
    for t in toks:
        if t not in glyphs and (_re.match(r"[一-鿿A-Za-z]", t)):
            v, real = text_vector(t)
            syl, style, _, rots = vector_to_glyph(v, real)
            g = Z.compose(syl, style, S=cell, rotations=rots, **LOOK)
            glyphs[t] = (g, real)
    pad, gap = 28, 10
    gl_h = 22 if gloss else 0
    row_h = cell + gl_h + gap
    punct_w = cell // 3
    f_gl = Z._load_font(Z.LABEL_FONT_CANDIDATES, 16)
    W = pad * 2 + cols * cell
    # 先算位置
    x, y, placed = 0, 0, []
    for t in toks:
        if t == "\n":
            x, y = 0, y + row_h + gap
            continue
        w = cell if t in glyphs else punct_w
        if x + w > cols * cell:
            x, y = 0, y + row_h
        placed.append((t, x, y))
        x += w
    H = pad * 2 + y + row_h
    page = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(page)
    for t, x, y in placed:
        px, py = pad + x, pad + y
        if t in glyphs:
            g, real = glyphs[t]
            tile = Image.fromarray(((1 - g) * 255).astype(np.uint8))
            page.paste(tile, (px, py), Image.fromarray((g * 255).astype(np.uint8)))
            if gloss:
                d.text((px + cell // 2, py + cell + 2), t, fill=120 if real else 190,
                       font=f_gl, anchor="mt")
        else:                                           # 標點:句號類大點,其他小點
            r = cell * (0.07 if t in _PUNCT_STOP else 0.045)
            cx, cy = px + punct_w / 2, py + cell * 0.82
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=0)
    out = Z.out_path(out)
    page.save(out)
    missing = [t for t, (_, real) in glyphs.items() if not real]
    return out, len(glyphs), missing


def article_irregular(text: str, cell: int = 110, width_cells: float = 9.0,
                      out: str = "zhuyin_article_free.png", gloss: bool = False):
    """一行一行整齊書寫,只有「字距」有變化(沿用 Gradio 版:每個字的寬度和間距來自它的向量)。

    - 字形不縮放、不旋轉、不扭曲,每行的中線對齊,行距固定。
    - 每個字佔的寬度 = 字形實際的墨跡寬度(比例字,窄的字佔得少)。
    - 字與下一個字之間的空隙由這個字自己的向量分量 z15 決定;同一個字每次出現,後面的空隙都一樣。
    """
    toks = tokenize(text)
    info = {}                                         # 字 → (墨跡 tile, 空隙, real)
    for t in toks:
        if t in info or not _re.match(r"[一-鿿A-Za-z]", t):
            continue
        v, real = text_vector(t)
        syl, style, _, rots = vector_to_glyph(v, real)
        g = Z.compose(syl, style, S=cell, rotations=rots, **LOOK)
        ys, xs = np.where(g > 0.05)
        g = g[:, xs.min():xs.max() + 1]               # 只裁左右,保留上下,讓各字中線一致
        u = _uniform(v * sqrt(len(v)), real)
        gap = 0.02 + 0.45 * u[15]                     # 空隙:0.02 ~ 0.47 格
        info[t] = (Image.fromarray((g * 255).astype(np.uint8)), gap, real)

    max_w = int(width_cells * cell)
    row_h = int(cell * 1.3)
    punct_w = cell * 0.3
    lines, cur, x = [], [], 0.0
    for t in toks:
        if t == "\n":
            lines.append(cur)
            cur, x = [], 0.0
            continue
        if t in info:
            im, gap, _ = info[t]
            w, adv = im.width, im.width + gap * cell
        else:
            w = adv = punct_w
        if x + w > max_w and cur:
            lines.append(cur)
            cur, x = [], 0.0
        cur.append((t, x))
        x += adv
    if cur:
        lines.append(cur)

    pad = int(cell * 0.5)
    gl_h = 22 if gloss else 0
    W = max_w + 2 * pad
    H = len(lines) * (row_h + gl_h) + 2 * pad
    page = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(page)
    f_gl = Z._load_font(Z.LABEL_FONT_CANDIDATES, 15)
    for r, ln in enumerate(lines):
        top = pad + r * (row_h + gl_h)
        for t, x in ln:
            px = int(pad + x)
            if t in info:
                im, _, real = info[t]
                page.paste(0, (px, top), im)
                if gloss:
                    d.text((px + im.width / 2, top + cell + 4), t, fill=130 if real else 190,
                           font=f_gl, anchor="mt")
            else:                                     # 標點:圓點貼在字的下緣
                rr = cell * (0.07 if t in _PUNCT_STOP else 0.045)
                cx, cy = px + punct_w / 2, top + cell * 0.82
                d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=0)
    out = Z.out_path(out)
    page.save(out)
    missing = [t for t, (_, _, real) in info.items() if not real]
    return out, len(info), missing


# 快取裡已有向量的詞,兩兩相關
DEFAULT_WORDS = ["fire", "火", "water", "河", "海", "dog", "tiger", "Light", "陽",
                 "陰", "黑貓", "despair", "belief", "明月", "秋"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("words", nargs="*", help="要造字的文字(中英文皆可)")
    ap.add_argument("--chosen", action="store_true", help="只輸出向量選中的拼法(一欄)")
    ap.add_argument("--mode", choices=["direct", "radical"], default="direct",
                    help="direct:向量分量直接決定字形(預設);radical:部首 + 音節")
    ap.add_argument("--article", metavar="TEXT", help="把一段文章排成造字頁面")
    ap.add_argument("--article-file", metavar="PATH", help="從檔案讀文章(UTF-8)")
    ap.add_argument("--cols", type=int, default=10, help="文章模式每行幾個字")
    ap.add_argument("--no-gloss", action="store_true", help="文章模式不在字形下方標原文")
    ap.add_argument("--irregular", action="store_true",
                    help="文章模式改成不規則排版(大小、浮動、旋轉、字距、整頁扭曲都由向量決定)")
    ap.add_argument("--gloss", action="store_true", help="不規則排版時在字形下方標原文")
    ap.add_argument("--cell", type=int, default=200)
    ap.add_argument("--out", default="zhuyin_semantic.png")
    a = ap.parse_args()
    global MODE
    MODE = a.mode
    if a.article or a.article_file:
        text = a.article or open(a.article_file, encoding="utf-8").read()
        cell = min(a.cell, 140) if a.cell != 200 else 110
        if a.irregular:
            out = a.out if a.out != "zhuyin_semantic.png" else "zhuyin_article_free.png"
            path, n, missing = article_irregular(text, cell=cell, width_cells=a.cols - 1,
                                                 out=out, gloss=a.gloss)
        else:
            out = a.out if a.out != "zhuyin_semantic.png" else "zhuyin_article.png"
            path, n, missing = article(text, cell=cell, cols=a.cols, out=out,
                                       gloss=not a.no_gloss)
        print(f"輸出: {path}({n} 個不同的字/詞)")
        if missing:
            print("沒有語意向量(用雜湊):", " ".join(missing))
        return
    words = a.words or DEFAULT_WORDS
    print("輸出:", sheet(words, cell=a.cell, out=a.out, all_styles=not a.chosen))


if __name__ == "__main__":
    main()

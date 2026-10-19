"""
注音造字 → 字型(產生器)

給 repo1 的 Tools/Conscript/build_font.py 用:它對字集裡每個字呼叫 glyph(),把字形描成外框寫進 TTF。
字形和 zhuyin_batch / zhuyin_semantic 的文章排版完全一樣(同一個語意向量 → 同一組符號、拼法、筆畫風格),
字寬也照文章排版:墨跡寬度 + 由向量 z15 決定的空隙,所以字型自帶那種不規則的字距。
中文字、英文字母、數字、標點和任何符號都一樣處理:每個字元各有自己的語意向量和字形。

產生器介面(任何造字都一樣):
    prepare(chars)            造字前呼叫一次;這裡用來補齊語意向量(一次 50 個字呼叫 Gemini,存進 embed_cache.json)
    glyph(ch, cell) -> dict   {"alpha": 高 = cell 的 0..1 陣列(1 = 墨), "advance": 字寬(像素),
                               "key": 字形的代號(用來找撞字), "info": 給人看的說明}
                              不支援的字回傳 None(遊戲會退回原本的字型)

run(單獨試用):
    python zhuyin_font.py 火水山        # 印出各字的符號與拼法
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
from math import sqrt
from urllib.request import Request, urlopen

import numpy as np

import zhuyin_glyph as Z
import zhuyin_semantic as S

def _cached_only(text: str, tries: int = 4):
    """造字時只查快取:向量都在 prepare() 一次批次補,這裡不再一個字一個字呼叫 API(額度用完時每個字會卡十幾秒)。"""
    hit = S._L._EMBED_CACHE.get(text) if S._L is not None else None
    return None if hit is None else np.asarray(hit, dtype=np.float64)


S._embed_retry = _cached_only

BATCH = 50
PACE = 40         # 秒;每批之間的間隔(免費額度:一次 100 個字會被擋,50 個可以)

# Standalone / poetry users retain the old single-glyph default. A font build
# opts into expansion explicitly; the shared semantic basis and cache are untouched.
_OPTIONS = {}


def configure(options: dict) -> None:
    global _OPTIONS
    unknown = set(options) - {"split_probability", "split_gap", "allow_embedding_requests"}
    if unknown:
        raise ValueError(f"Unknown font provider settings: {sorted(unknown)}")
    probability = float(options.get("split_probability", 0.0))
    gap = float(options.get("split_gap", 0.18))
    if not 0 <= probability <= 1 or not 0 <= gap <= 1:
        raise ValueError("split_probability and split_gap must be between zero and one")
    _OPTIONS = dict(options, split_probability=probability, split_gap=gap)


def part_count(ch: str) -> int:
    """Stable per codepoint, independent of embeddings, worker order and rebuilds."""
    code = ord(ch)
    cjk = 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or 0x20000 <= code <= 0x323AF
    value = int.from_bytes(hashlib.sha256(("zhuyin-length-v1:" + ch).encode()).digest()[:8], "big") / 2**64
    return 2 if cjk and value < _OPTIONS.get("split_probability", 0.0) else 1


def _batch_embed(texts: list[str]) -> list[np.ndarray]:
    L = S._L
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/{L._EMBED_MODEL}"
           f":batchEmbedContents?key={L._GEMINI_KEY}")
    body = json.dumps({"requests": [{"model": f"models/{L._EMBED_MODEL}",
                                     "content": {"parts": [{"text": t}]},
                                     "outputDimensionality": L._EMBED_DIM} for t in texts]}).encode()
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    out = json.load(urlopen(req, timeout=60))["embeddings"]
    vecs = [np.asarray(e["values"], dtype=np.float64) for e in out]
    return [v / (np.linalg.norm(v) + 1e-8) for v in vecs]


def prepare(chars) -> None:
    """快取裡沒有語意向量的字,分批向 Gemini 要(和 lenia_text._embed 同一個模型、維度、正規化)。"""
    if not _OPTIONS.get("allow_embedding_requests", True):
        print("[zhuyin_font] Cache-only build; no embedding requests or cache updates")
        return
    L = S._L
    if L is None or not L._GEMINI_KEY:
        print("[zhuyin_font] 沒有 GEMINI_API_KEY,沒有向量的字會用雜湊造字")
        return
    todo = [c for c in dict.fromkeys(chars) if c not in L._EMBED_CACHE]
    if not todo:
        return
    print(f"[zhuyin_font] 補 {len(todo)} 個字的語意向量({-(-len(todo) // BATCH)} 次呼叫)")
    for i in range(0, len(todo), BATCH):
        part = todo[i:i + BATCH]
        vecs = None
        for k in range(5):
            try:
                vecs = _batch_embed(part)
                break
            except Exception as e:                       # 每分鐘的速率限制或網路抖動:等一分鐘再試
                body = e.read().decode(errors="ignore") if hasattr(e, "read") else ""
                if "free_tier_requests" in body:         # 免費額度每天 1000 個字:用完就先停,其餘的字用雜湊造
                    print(f"  今天的免費額度用完了,還差 {len(todo) - i} 個字;沒有向量的字先用雜湊造,額度恢復後重建字型即可", flush=True)
                    return
                print(f"  第 {i // BATCH + 1} 批失敗({e}),一分鐘後重試", flush=True)
                time.sleep(60)
        if vecs is None:
            print(f"  語意向量拿不到,還差 {len(todo) - i} 個字;先用雜湊造", flush=True)
            return
        for c, v in zip(part, vecs):
            L._EMBED_CACHE[c] = v.tolist()
        json.dump(L._EMBED_CACHE, open(L._CACHE_PATH, "w", encoding="utf-8"))
        print(f"  {min(i + BATCH, len(todo))}/{len(todo)}", flush=True)
        if i + BATCH < len(todo):
            time.sleep(PACE)                             # 一批 50 個字,不撞到速率限制


def _part(v: np.ndarray, real: bool, cell: int) -> dict | None:
    seq, style, _, rots = S.vector_to_glyph(v, real)
    g = Z.compose(seq, style, S=cell, rotations=rots, **S.LOOK)
    xs = np.where(g.max(axis=0) > 0.05)[0]
    if len(xs) == 0:
        return None
    g = g[:, xs.min():xs.max() + 1]                    # 只裁左右,上下保留讓中線一致
    u = S._uniform(v * sqrt(len(v)), real)
    gap = 0.02 + 0.45 * u[15]                          # 與文章排版同一個字距規則
    return {"alpha": g, "advance": g.shape[1] + gap * cell, "key": f"{seq}|{style}",
            "info": f"{seq} {style}" + ("" if real else " (雜湊)")}


def glyph(ch: str, cell: int) -> dict | None:
    v, real = S.text_vector(ch)
    first = _part(v, real, cell)
    if first is None:
        return None
    first["parts"] = 1
    if part_count(ch) == 1:
        return first
    # A fixed orthogonal transform provides another view of the same vector;
    # it needs no new embedding and preserves neighbourhoods in vector space.
    # Keep the original first part, and avoid duplicating its symbol/layout key.
    for attempt in range(8):
        seed = int.from_bytes(hashlib.sha256(f"zhuyin-second-v1:{attempt}".encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        second = _part(v[rng.permutation(len(v))] * rng.choice([-1, 1], len(v)), real, cell)
        if second is not None and second["key"] != first["key"]:
            break
    else:
        raise ValueError(f"Cannot generate a distinct second part for U+{ord(ch):04X}")
    gap = max(1, round(cell * _OPTIONS["split_gap"]))
    alpha = np.concatenate([first["alpha"], np.zeros((cell, gap)), second["alpha"]], axis=1)
    return {"alpha": alpha, "advance": first["alpha"].shape[1] + gap + second["advance"],
            "key": first["key"] + " + " + second["key"],
            "info": first["info"] + " + " + second["info"], "parts": 2}


if __name__ == "__main__":
    for c in "".join(sys.argv[1:]) or "火水山":
        r = glyph(c, 128)
        print(c, r["info"] if r else "—")

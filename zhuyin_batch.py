"""
批量產生去背的造字圖

每張圖是一行 1~11 個字的造字(和 zhuyin_semantic 的文章排版同一套:比例字距,
字與字的空隙由各字的語意向量決定),透明背景,墨色預設 #FB01A3。

文字來源:
  - --source 文章.txt:從文章裡依序切出長度 1~11 的片段(略過標點與空白)
  - 不給:從 embed_cache.json 已有向量的中文字隨機組合(不呼叫 API)

輸出:<out>/0001.png ... 以及 <out>/manifest.csv(檔名 → 原文 → 造字符號)

run:
    python zhuyin_batch.py                              # 100 張,快取字隨機組合
    python zhuyin_batch.py -n 200 --source 文章.txt --out batch_story        # 輸出到 gallery/batch_story/
    python zhuyin_batch.py --color "#000000" --cell 160 --seed 7
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import random
import re
import numpy as np
from math import sqrt
from PIL import Image

import zhuyin_glyph as Z
import zhuyin_semantic as S

UNIT = re.compile(r"[一-鿿]|[A-Za-z]+(?:'[A-Za-z]+)?")


def _hex(c: str) -> tuple:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


class GlyphBank:
    """每個字只造一次字形(很慢的那一步),整批共用。"""

    def __init__(self, cell: int):
        self.cell = cell
        self.bank = {}

    def get(self, unit: str):
        if unit not in self.bank:
            v, real = S.text_vector(unit)
            syl, style, _, rots = S.vector_to_glyph(v, real)
            g = Z.compose(syl, style, S=self.cell, rotations=rots, **S.LOOK)
            xs = np.where(g.max(axis=0) > 0.05)[0]
            g = g[:, xs.min():xs.max() + 1]           # 只裁左右,上下保留讓中線一致
            u = S._uniform(v * sqrt(len(v)), real)
            gap = 0.02 + 0.45 * u[15]                 # 與文章排版同一個字距規則
            self.bank[unit] = (g, gap, syl, real)
        return self.bank[unit]


def render_line(units, bank: GlyphBank, color=(251, 1, 163), margin: float = 0.12) -> Image.Image:
    """一行字 → 去背 RGBA 圖(裁到墨跡外框,四周留一點邊)。"""
    cell = bank.cell
    parts, x = [], 0.0
    for i, t in enumerate(units):
        g, gap, _, _ = bank.get(t)
        parts.append((g, int(round(x))))
        x += g.shape[1] + (gap * cell if i < len(units) - 1 else 0)
    W = int(round(x)) + 1
    alpha = np.zeros((cell, W), dtype=np.float32)
    for g, px in parts:
        w = g.shape[1]
        alpha[:, px:px + w] = np.maximum(alpha[:, px:px + w], g[:, :W - px])
    ys, xs = np.where(alpha > 0.02)
    m = int(cell * margin)
    alpha = alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    alpha = np.pad(alpha, m)
    rgba = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    rgba[..., :3] = color
    rgba[..., 3] = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def snippets_from_text(text: str, n: int, rng: random.Random, lo: int, hi: int):
    """依序從文章切出 n 個長度 lo~hi 的片段;文章用完就從頭再來。"""
    units = UNIT.findall(text)
    if not units:
        raise SystemExit("文章裡沒有可用的字")
    out, i = [], 0
    while len(out) < n:
        k = rng.randint(lo, hi)
        if i + k > len(units):
            i = 0
            k = min(k, len(units))
        out.append(units[i:i + k])
        i += k
    return out


def snippets_from_cache(n: int, rng: random.Random, lo: int, hi: int):
    """從快取裡已有向量的中文字隨機組合(不呼叫 API)。"""
    cache = json.load(open(os.path.join(S._HERE, "embed_cache.json"), encoding="utf-8"))
    chars = sorted(k for k in cache if len(k) == 1 and re.match(r"[一-鿿]", k))
    if not chars:
        raise SystemExit("快取裡沒有中文字,請用 --source 指定文章")
    return [[rng.choice(chars) for _ in range(rng.randint(lo, hi))] for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=100, help="要產生幾張(預設 100)")
    ap.add_argument("--min", type=int, default=1, help="每張最少幾個字")
    ap.add_argument("--max", type=int, default=11, help="每張最多幾個字")
    ap.add_argument("--source", help="文章檔(UTF-8);不給就用快取裡的字隨機組合")
    ap.add_argument("--out", default="batch", help="輸出資料夾(只給名字時建在 gallery/ 底下)")
    ap.add_argument("--cell", type=int, default=128, help="每個字的大小(像素)")
    ap.add_argument("--color", default="#FB01A3", help="墨色")
    ap.add_argument("--seed", type=int, default=0, help="隨機種子(同樣的種子產生同樣的一批)")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    if a.source:
        snips = snippets_from_text(open(a.source, encoding="utf-8").read(), a.n, rng, a.min, a.max)
    else:
        snips = snippets_from_cache(a.n, rng, a.min, a.max)

    a.out = Z.out_path(a.out)
    os.makedirs(a.out, exist_ok=True)
    bank = GlyphBank(a.cell)
    color = _hex(a.color)
    width = len(str(a.n))
    rows = []
    for i, units in enumerate(snips, start=1):
        name = f"{i:0{max(4, width)}d}.png"
        render_line(units, bank, color).save(os.path.join(a.out, name))
        rows.append((name, "".join(units) if all(len(u) == 1 for u in units) else " ".join(units),
                     " ".join(bank.get(u)[2] for u in units)))
        if i % 10 == 0 or i == len(snips):
            print(f"{i}/{len(snips)}  (已造 {len(bank.bank)} 個不同的字)")
    with open(os.path.join(a.out, "manifest.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "text", "symbols"])
        w.writerows(rows)
    missing = [t for t, (_, _, _, real) in bank.bank.items() if not real]
    print(f"完成:{a.out}/ 共 {len(rows)} 張")
    if missing:
        print("沒有語意向量(用雜湊):", " ".join(missing))


if __name__ == "__main__":
    main()

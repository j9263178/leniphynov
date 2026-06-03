"""
造字系統 — Gradio 互動介面
run: python lenia_app.py
"""
import os, tempfile, functools

# Must set non-interactive backend before any matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **kw: None          # suppress any stray plt.show calls

import gradio as gr
from PIL import Image

from lenia_text import (
    show_poem_grouped,
    show_poem_voronoi,
    show_poem_noise,
    show_poem_waves,
    show_poem_mixed,
    set_granularity,
)

# 粒度標籤 → lenia_text 內部代號
GRAINS = {
    "字　｜ 一字一圖":   "char",
    "句　｜ 一句一圖":   "sentence",
    "整首｜ 全文一圖":   "whole",
}

# mixed_germ = show_poem_mixed with germ=True baked in
def _mixed_germ(text, **kwargs):
    kwargs.setdefault("germ", True)
    return show_poem_mixed(text, **kwargs)

MODES = {
    "mixed      ｜ wave + blob 混合":      show_poem_mixed,
    "mixed_germ ｜ wave + blob + 黏菌":   _mixed_germ,
    "waves      ｜ 水平行 + 波浪邊":       show_poem_waves,
    "grouped    ｜ 每組獨立 blob":         show_poem_grouped,
    "noise      ｜ 木紋帶狀":              show_poem_noise,
    "voronoi    ｜ warped 分割":           show_poem_voronoi,
}


def generate(text, mode_label, grain_label, group_size, cols, steps,
             octaves, wave_base, row_shift,
             germ_steps, germ_sa, germ_ra, germ_so):
    text = text.strip()
    if not text:
        return None

    set_granularity(GRAINS.get(grain_label, "char"))

    tmp = tempfile.mktemp(suffix=".png")
    fn  = MODES[mode_label]

    kwargs = dict(out=tmp, steps=int(steps), cols=int(cols))
    if any(k in mode_label for k in ("grouped", "waves", "mixed")):
        kwargs["group_size"] = int(group_size)
    if any(k in mode_label for k in ("waves", "mixed", "poly")):
        kwargs["row_shift"] = float(row_shift)
    if "germ" in mode_label:
        kwargs["germ_steps"] = int(germ_steps)
        kwargs["germ_sa"]    = float(germ_sa)
        kwargs["germ_ra"]    = float(germ_ra)
        kwargs["germ_so"]    = int(germ_so)
    if "noise" in mode_label:
        kwargs["octaves"]   = int(octaves)
        kwargs["wave_base"] = float(wave_base)

    fn(text, **kwargs)

    img = Image.open(tmp)
    return img


# ── UI ────────────────────────────────────────────────────────────────────────
css = """
/* ── 全黑底 全白字 無邊框 ── */
:root {
    --body-background-fill:                 #000;
    --background-fill-primary:              #000;
    --background-fill-secondary:            #000;
    --block-background-fill:                #000;
    --input-background-fill:                #000;
    --input-background-fill-focus:          #000;
    --border-color-primary:                 transparent;
    --border-color-accent:                  transparent;
    --color-accent:                         #fff;
    --body-text-color:                      #fff;
    --body-text-color-subdued:              #aaa;
    --block-label-text-color:               #fff;
    --block-label-text-size:                13px;
    --block-title-text-color:               #fff;
    --block-title-text-size:                13px;
    --input-placeholder-color:              #444;
    --checkbox-label-text-color:            #fff;
    --slider-color:                         #fff;
    --button-primary-background-fill:       #fff;
    --button-primary-background-fill-hover: #ddd;
    --button-primary-text-color:            #000;
    --button-primary-border-color:          transparent;
    --button-secondary-background-fill:     #000;
    --button-secondary-background-fill-hover: #111;
    --button-secondary-text-color:          #fff;
    --button-secondary-border-color:        #333;
    --table-border-color:                   #222;
    --table-even-background-fill:           #000;
    --table-odd-background-fill:            #000;
    --shadow-drop: none; --shadow-drop-lg: none; --shadow-inset: none;
    --radius-xs: 2px; --radius-sm: 2px; --radius-md: 2px;
    --radius-lg: 2px; --radius-xxl: 2px;
}

/* 全部背景黑、陰影全無 */
*, *::before, *::after {
    background-color: #000 !important;
    box-shadow: none !important;
    border-color: transparent !important;
}

/* 全部文字白 */
*, *::before, *::after {
    color: #fff !important;
}

body { -webkit-font-smoothing: antialiased; }

.gradio-container {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 40px 60px 80px !important;
    font-family: 'SF Pro Text', 'Helvetica Neue', 'PingFang TC',
                 'Noto Sans TC', system-ui, sans-serif !important;
}

/* ── Header ── */
.app-header { margin-bottom: 40px; padding-bottom: 28px; border-bottom: 1px solid #222 !important; }
.app-header h1 {
    font-size: 28px !important;
    font-weight: 200 !important;
    letter-spacing: 0.25em !important;
    margin: 0 0 8px !important;
    line-height: 1 !important;
}
.app-header p {
    color: #666 !important;
    font-size: 12px !important;
    letter-spacing: 0.18em !important;
    margin: 0 !important;
    text-transform: uppercase;
}

/* ── 輸入框 — 只用底線做輸入感 ── */
textarea, input[type="text"], input[type="number"] {
    border: none !important;
    border-bottom: 1px solid #333 !important;
    border-radius: 0 !important;
    font-size: 16px !important;
    line-height: 1.7 !important;
    padding: 10px 0 !important;
    caret-color: #fff;
}
textarea:focus, input:focus {
    border-bottom-color: #fff !important;
    outline: none !important;
}
textarea::placeholder, input::placeholder { color: #444 !important; }

/* ── 標籤字放大 ── */
label > span, .block-label, .block-label span,
label span, span.svelte-1gfkn6j, .svelte-pbokmd {
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    text-transform: none !important;
    color: #fff !important;
}

/* ── Radio 選項 ── */
fieldset { border: none !important; padding: 0 !important; }
.wrap label {
    border: 1px solid #333 !important;
    border-radius: 2px !important;
    font-size: 13px !important;
    padding: 7px 14px !important;
    cursor: pointer !important;
    transition: border-color 0.1s, opacity 0.1s !important;
    opacity: 0.45 !important;
}
.wrap label:has(input[type="radio"]:checked) {
    border-color: #fff !important;
    opacity: 1 !important;
}
.wrap label input { display: none !important; }

/* ── Slider 數值 ── */
input[type="range"] { accent-color: #fff !important; }
.numeral { font-size: 13px !important; color: #aaa !important; }

/* ── Accordion ── */
details { border-top: 1px solid #222 !important; border-radius: 0 !important; }
details > summary {
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    padding: 14px 0 !important;
    cursor: pointer !important;
    list-style: none !important;
}
details > summary::-webkit-details-marker { display: none; }

/* ── 生成按鈕 ── */
button.primary {
    background: #fff !important;
    color: #000 !important;
    border: none !important;
    border-radius: 2px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    padding: 16px !important;
    width: 100% !important;
    margin-top: 8px !important;
    transition: opacity 0.12s !important;
}
button.primary:hover { opacity: 0.8 !important; }
button.primary:active { opacity: 0.55 !important; }

/* ── 其他按鈕 ── */
button.secondary {
    border: 1px solid #333 !important;
    font-size: 13px !important;
    padding: 8px 16px !important;
}
button.secondary:hover { border-color: #666 !important; }

/* ── 圖片區 ── */
.image-frame, .image-container, [data-testid="image"] {
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    background: transparent !important;
}

/* ── 狀態列 ── */
textarea[readonly], .readonly textarea {
    font-size: 13px !important;
    color: #aaa !important;
    border-bottom-color: #1a1a1a !important;
}

/* ── Examples 表格 ── */
.examples { border-top: 1px solid #222 !important; }
.examples-header { font-size: 12px !important; padding: 12px 0 6px !important; }
thead th {
    font-size: 12px !important;
    border-bottom: 1px solid #222 !important;
    padding: 8px 12px !important;
    color: #aaa !important;
}
tbody tr { border-bottom: 1px solid #111 !important; cursor: pointer !important; }
tbody tr:hover { background: #0d0d0d !important; }
tbody td { font-size: 13px !important; padding: 10px 12px !important; }

/* ── 隱藏頂部系統進度條，只保留按鈕 loading 狀態 ── */
#progress-bar, .progress-bar, .progress-bar-wrap,
div[id="progress"], .generating { display: none !important; }

/* ── 按鈕 loading 動畫 ── */
button.primary.running,
button.primary[disabled] {
    opacity: 0.45 !important;
    cursor: wait !important;
}

/* ── 隱藏 footer ── */
footer, .footer, .built-with { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: #000; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }
"""

_HEADER = """
<div class="app-header">
  <h1>造字系統</h1>
  <p>Lenia · Cellular Automata · Text Rendering</p>
</div>
"""

with gr.Blocks(title="造字系統") as demo:
    gr.HTML(_HEADER)

    with gr.Row(equal_height=False):
        # ── left panel ────────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):
            text_in = gr.Textbox(
                label="輸入文字",
                placeholder="床前明月光，疑是地上霜。",
                lines=4,
            )
            grain_sel = gr.Radio(
                choices=list(GRAINS.keys()),
                value=list(GRAINS.keys())[0],
                label="生成粒度",
            )
            mode_sel = gr.Radio(
                choices=list(MODES.keys()),
                value=list(MODES.keys())[0],
                label="排版模式",
            )

            with gr.Row():
                cols_sel = gr.Slider(
                    2, 8, value=4, step=1,
                    label="每列字數",
                    scale=1,
                )
                steps_sel = gr.Slider(
                    100, 500, value=300, step=50,
                    label="演化步數",
                    scale=2,
                )

            with gr.Accordion("進階參數", open=False):
                group_sel = gr.Slider(
                    2, 3, value=3, step=1,
                    label="每組字數（grouped 模式）",
                )
                row_shift_sel = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05,
                    label="行偏移量（waves / mixed）",
                )
                with gr.Row():
                    octaves_sel = gr.Slider(
                        2, 6, value=4, step=1,
                        label="倍頻（noise）",
                        scale=1,
                    )
                    wave_base_sel = gr.Slider(
                        0.5, 3.0, value=1.3, step=0.1,
                        label="波長係數（noise）",
                        scale=1,
                    )
                germ_steps_sel = gr.Slider(
                    30, 200, value=100, step=10,
                    label="黏菌步數（mixed_germ）",
                )
                with gr.Row():
                    germ_sa_sel = gr.Slider(
                        5, 70, value=38, step=1,
                        label="SA 感測角°",
                        scale=1,
                    )
                    germ_ra_sel = gr.Slider(
                        5, 90, value=45, step=1,
                        label="RA 轉向角°",
                        scale=1,
                    )
                    germ_so_sel = gr.Slider(
                        2, 30, value=9, step=1,
                        label="SO 距離 px",
                        scale=1,
                    )

            btn = gr.Button("生成", variant="primary", size="lg")

        # ── right panel ───────────────────────────────────────────────────────
        with gr.Column(scale=2):
            img_out = gr.Image(label="輸出", type="pil", height=300)

    btn.click(
        fn=generate,
        inputs=[text_in, mode_sel, grain_sel, group_sel, cols_sel, steps_sel,
                octaves_sel, wave_base_sel, row_shift_sel,
                germ_steps_sel, germ_sa_sel, germ_ra_sel, germ_so_sel],
        outputs=[img_out],
    )

    _G = list(GRAINS.keys())
    gr.Examples(
        examples=[
            ["床前明月光，疑是地上霜。舉頭望明月，低頭思故鄉。", list(MODES.keys())[0], _G[0], 3, 4, 300, 4, 1.3, 0.5, 100, 38, 45, 9],
            ["床前明月光，疑是地上霜。舉頭望明月，低頭思故鄉。", list(MODES.keys())[1], _G[1], 3, 4, 300, 4, 1.3, 0.5, 100, 38, 45, 9],
            ["春眠不覺曉，處處聞啼鳥。夜來風雨聲，花落知多少。", list(MODES.keys())[2], _G[1], 3, 5, 300, 4, 1.3, 0.5, 100, 38, 45, 9],
            ["白日依山盡，黃河入海流。欲窮千里目，更上一層樓。", list(MODES.keys())[4], _G[0], 3, 5, 300, 6, 2.0, 0.5, 100, 38, 45, 9],
        ],
        inputs=[text_in, mode_sel, grain_sel, group_sel, cols_sel, steps_sel,
                octaves_sel, wave_base_sel, row_shift_sel,
                germ_steps_sel, germ_sa_sel, germ_ra_sel, germ_so_sel],
        label="範例",
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Base(), css=css)

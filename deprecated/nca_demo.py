"""
nca_demo.py  ——  Neural Cellular Automata (NCA) 自學示範
==========================================================

參考 Emergent Garden 影片裡的 NCA 藍圖，實作步驟 1~3（不含第 4 步的遺傳演算法
搜尋），跑出「幾個模式」的 demo，最後拼成一張 montage 圖 (nca_demo.png)。

藍圖對照
--------
1. 多維度狀態矩陣      -> NCA.state，形狀 (C, H, W)，前 3 個 channel 當 RGB
2. 感知周圍環境        -> PERCEPTION：固定卷積核 identity / sobel_x / sobel_y /
                          laplacian / gaussian（低通，幫忙長出 Turing 紋理）
3. 神經網路決策        -> NCA._net：兩層 1x1 卷積 (= channel 維度上的小 MLP) + tanh
   隨機異步更新        -> 每步以 fire_rate 機率遮罩，只更新部分細胞

關於「第 4 步」
---------------
影片裡會分裂、像黏菌的「生物」，是用遺傳演算法『搜尋』權重找出來的；這支腳本
不做那個搜尋，所以：
  * 隨機權重 NCA -> 多半是 Turing 風格紋理（斑點 / 條紋 / 迷宮）。這裡用一個
    「結構分數」自動從多組隨機規則挑出最有結構的幾組（取代第 4 步的搜尋）。
  * 想看明確、可重現的「細胞分裂 / 蠕蟲 / 迷宮」，看下面的 Gray-Scott 反應擴散
    （NCA 的確定性表親：Laplacian = 感知，反應式 = 固定的網路）。

依賴：numpy / scipy / matplotlib （都已在專案裡，不需要 torch）
執行：python nca_demo.py
"""

import numpy as np
from scipy.ndimage import convolve
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "nca_demo.png"
rng_global = np.random.default_rng()


# ----------------------------------------------------------------------------
# 引擎 A：忠於藍圖的多通道 NCA（步驟 1~3）
# ----------------------------------------------------------------------------
NCA_SIZE = 110      # NCA panel 格點數（小一點跑比較快）

# --- 步驟 2：固定的感知卷積核 -------------------------------------------------
_IDENT = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
_SOBEL_Y = _SOBEL_X.T.copy()
_LAPL = np.array([[0.05, 0.2, 0.05],
                  [0.20, -1.0, 0.20],
                  [0.05, 0.2, 0.05]], dtype=np.float32)
_GAUSS = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
PERCEPTION = [_IDENT, _SOBEL_X, _SOBEL_Y, _LAPL, _GAUSS]   # 5 個過濾器


class NCA:
    """多通道神經細胞自動機。權重隨機初始化（不訓練、不搜尋）。"""

    def __init__(self, channels=8, hidden=64, dt=0.4, fire_rate=0.5, seed=0):
        self.C = channels
        self.dt = dt
        self.fire_rate = fire_rate
        rng = np.random.default_rng(seed)

        n_perc = self.C * len(PERCEPTION)
        # 步驟 3：兩層 1x1 卷積 = channel 維度上的小型 MLP
        self.w1 = rng.standard_normal((hidden, n_perc)).astype(np.float32) * (1.2 / np.sqrt(n_perc))
        self.b1 = rng.standard_normal(hidden).astype(np.float32) * 0.1
        self.w2 = rng.standard_normal((self.C, hidden)).astype(np.float32) * (1.2 / np.sqrt(hidden))
        self.b2 = np.zeros(self.C, dtype=np.float32)

        # 步驟 1：狀態矩陣，從一片低振幅噪聲長出來
        self.state = rng.standard_normal((self.C, NCA_SIZE, NCA_SIZE)).astype(np.float32) * 0.1
        self._rng = rng

    def _perceive(self):
        feats = []
        for f in PERCEPTION:
            for c in range(self.C):
                feats.append(convolve(self.state[c], f, mode="wrap"))
        return np.stack(feats, axis=0)            # (C*F, H, W)

    def step(self):
        perc = self._perceive()
        flat = perc.reshape(perc.shape[0], -1)
        h = np.tanh(self.w1 @ flat + self.b1[:, None])
        dx = (self.w2 @ h + self.b2[:, None]).reshape(self.C, NCA_SIZE, NCA_SIZE)
        dx -= dx.mean(axis=(1, 2), keepdims=True)          # 像 instance-norm
        # 隨機異步更新
        mask = (self._rng.random((1, NCA_SIZE, NCA_SIZE)) < self.fire_rate).astype(np.float32)
        self.state = np.tanh(self.state + self.dt * dx * mask)

    def run(self, steps):
        for _ in range(steps):
            self.step()
        return self.state

    def rgb(self):
        img = self.state[:3].transpose(1, 2, 0)
        lo, hi = img.min(), img.max()
        return (img - lo) / (hi - lo + 1e-8)


def _score(state):
    """挑規則用的指標。回傳 (ok, structure)。
    ok: 沒死、沒整片飽和。structure: 相鄰像素相關性（越高越平滑有結構）。"""
    a = state[0]
    if not np.isfinite(a).all():
        return False, 0.0
    if a.std() < 0.05:                          # 幾乎常數 -> 死了
        return False, 0.0
    if np.mean(np.abs(a) > 0.97) > 0.7:          # 大片貼在 ±1 -> 飽和噪聲
        return False, 0.0
    a = a - a.mean()
    denom = (a * a).sum() + 1e-8
    corr = float((a * np.roll(a, 1, axis=0)).sum() / denom)
    return True, corr


def find_nca_modes(n_modes=3, candidates=26, steps=100):
    """跑多組隨機規則，挑出最有結構的幾個（取代第 4 步的遺傳演算法）。"""
    found = []
    for _ in range(candidates):
        seed = int(rng_global.integers(0, 1_000_000))
        net = NCA(seed=seed)
        net.run(steps)
        ok, structure = _score(net.state)
        if ok and structure > 0.30:
            found.append((structure, net))
    found.sort(key=lambda t: -t[0])             # 結構越強排越前
    return [net for _, net in found[:n_modes]]


# ----------------------------------------------------------------------------
# 引擎 B：Gray-Scott 反應擴散（NCA 的確定性表親，可重現的命名模式）
# ----------------------------------------------------------------------------
GS_SIZE = 160
_GS_LAP = np.array([[0.05, 0.2, 0.05],
                    [0.20, -1.0, 0.20],
                    [0.05, 0.2, 0.05]], dtype=np.float32)


def gray_scott(f, k, steps, Du=1.0, Dv=0.5, seed=0):
    rng = np.random.default_rng(seed)
    U = np.ones((GS_SIZE, GS_SIZE), dtype=np.float32)
    V = np.zeros((GS_SIZE, GS_SIZE), dtype=np.float32)
    # 在中央放一小塊種子，讓「群落」自己往外長
    r, c = 12, GS_SIZE // 2
    U[c - r:c + r, c - r:c + r] = 0.50
    V[c - r:c + r, c - r:c + r] = 0.25
    V += rng.random((GS_SIZE, GS_SIZE)).astype(np.float32) * 0.02

    for _ in range(steps):
        uvv = U * V * V
        U += Du * convolve(U, _GS_LAP, mode="wrap") - uvv + f * (1.0 - U)
        V += Dv * convolve(V, _GS_LAP, mode="wrap") + uvv - (f + k) * V
        np.clip(U, 0.0, 1.0, out=U)
        np.clip(V, 0.0, 1.0, out=V)
    return V


# ----------------------------------------------------------------------------
# 組裝 montage
# ----------------------------------------------------------------------------
def main():
    panels = []   # (title, image, cmap)

    print("Running Gray-Scott reaction-diffusion (deterministic named modes)...")
    gs_presets = [
        ("Gray-Scott: mitosis", 0.0367, 0.0649, 14000),
        ("Gray-Scott: coral", 0.0545, 0.0620, 10000),
        ("Gray-Scott: waves", 0.0180, 0.0510, 10000),
    ]
    for title, f, k, steps in gs_presets:
        print(f"  - {title}")
        V = gray_scott(f, k, steps, seed=int(rng_global.integers(0, 1_000_000)))
        panels.append((title, V, "magma"))

    print("Searching random NCA rules with structure (replaces GA / step 4)...")
    nca_nets = find_nca_modes(n_modes=3, candidates=18, steps=90)
    for j, net in enumerate(nca_nets):
        print(f"  - NCA random rule #{j + 1}")
        panels.append((f"NCA: random rule #{j + 1} (Turing texture)", net.rgb(), None))

    if not panels:
        print("No panels produced; run again.")
        return

    cols = 3
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.6))
    axes = np.array(axes).reshape(-1)
    for ax, (title, img, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, interpolation="bilinear") if cmap else \
            ax.imshow(img, interpolation="bilinear")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"\nDone -> {OUT}")


if __name__ == "__main__":
    main()

"""
Experiment: embedding-grid cellular automaton.

Idea
----
1. Embed the text → high-dim vector (256-d from Gemini).
2. Lay every component onto a cell of a square grid (256 → 16×16).
3. Each cell's |component| = its *evolution intensity* (how strongly it reacts).
4. Evolve with a 4-neighbour (von Neumann: up/down/left/right) rule only.

This is a quick visual probe — render snapshots over time.

run: python grid_embed_test.py  "你的文字"
"""
import sys
import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the project's embedding (256-d). Falls back to a deterministic
# pseudo-embedding if the API/cache is unavailable, so the mechanics still run.
import lenia_text as L


def get_embedding(text):
    v = L._embed(text)
    if v is not None:
        print(f"[embed] real {len(v)}-d embedding")
        return np.asarray(v, dtype=np.float64)
    # fallback: deterministic random vector from the text hash
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
    v = np.random.default_rng(seed).standard_normal(L._EMBED_DIM)
    print(f"[embed] FALLBACK pseudo {len(v)}-d (no API/cache)")
    return v / (np.linalg.norm(v) + 1e-8)


def to_grid(v):
    """Reshape a 1-D vector to the largest square grid that fits."""
    side = int(np.floor(np.sqrt(len(v))))
    g = v[: side * side].reshape(side, side)
    return g, side


bell = lambda x, m, s: np.exp(-((x - m) / s) ** 2 / 2)


def evolve(seed, intensity, steps, rng, leak=0.012, gain=0.45):
    """
    Organic, slime-mould-like outward growth (4-neighbour, von Neumann only).

    Like the monotonic spread, a cell never decays — it only ever takes the
    MAX of itself and an incoming neighbour value (faded by `leak`).  But the
    spread is now STOCHASTIC and biased, so fronts branch and meander instead
    of forming perfect diamonds:

      • `crack`  — a persistent per-cell "ease of growth" field (random, fixed
                   for the whole run).  Growth keeps flowing through the same
                   easy cells → vein / tile-crack channels emerge.
      • per-step random acceptance → fuzzy, branching edges.

    Stronger seeds carry a larger value, survive more `leak` steps, and thus
    grow into bigger blobs (the high-intensity component = the central mass).
    """
    from scipy.ndimage import gaussian_filter
    A = seed.copy()
    # persistent channel field: smoothed noise → connected easy/hard CHANNELS
    # (so growth flows through veins, like slime mould in tile grout) then
    # sharpened so the contrast between "crack" and "tile" is strong.
    crack = gaussian_filter(rng.random(A.shape), sigma=1.6)
    crack = (crack - crack.min()) / (np.ptp(crack) + 1e-8)
    crack = crack ** 2.2                          # sharpen → narrow channels
    snaps = [A.copy()]
    snap_at = {0, steps // 8, steps // 4, steps // 2, steps - 1}
    for t in range(steps):
        nbr_max = np.maximum(
            np.maximum(np.roll(A, 1, 0), np.roll(A, -1, 0)),
            np.maximum(np.roll(A, 1, 1), np.roll(A, -1, 1)))
        incoming = nbr_max - leak
        # stochastic, channel-biased acceptance of the incoming front
        accept = rng.random(A.shape) < (gain * crack)
        grow   = accept & (incoming > A)
        A = np.where(grow, incoming, A)          # grow only; never shrinks
        if t in snap_at:
            snaps.append(A.copy())
    return snaps


def main():
    text  = sys.argv[1] if len(sys.argv) > 1 else "明月光"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    block = int(sys.argv[3]) if len(sys.argv) > 3 else 8       # cells per component
    leak  = float(sys.argv[4]) if len(sys.argv) > 4 else 0.006 # fade per cell
    floor = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05  # binarise threshold
    gain  = float(sys.argv[6]) if len(sys.argv) > 6 else 0.45  # growth randomness

    v = get_embedding(text)
    grid_raw, side = to_grid(v)
    print(f"[grid] {side}×{side} = {side*side} components "
          f"→ block {block}×{block} → field {side*block}×{side*block}")

    # intensity = |component| normalised to [0,1];
    # initial state = intensity itself → low-|z| cells start ~0 (dark/static),
    # high-|z| cells start bright and act as the active seeds.
    W  = np.abs(grid_raw)
    W  = W / (W.max() + 1e-8)
    A0 = W.copy()

    # ── upscale: each component → block×block region of the evolution field ───
    ones = np.ones((block, block))
    A0 = np.kron(A0, ones)
    W  = np.kron(W,  ones)

    rng = np.random.default_rng(
        int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32))
    snaps = evolve(A0, W, steps, rng, leak=leak, gain=gain)

    # ── binarise + colourise (same off-white-on-black look as the main app) ───
    def binar(field):
        """field in [0,1] → binary RGB: above `floor` → COLOR, else black."""
        mask = (field > floor).astype(float)
        rgb = np.zeros((*field.shape, 3))
        for i, c in enumerate(L.COLOR):
            rgb[:, :, i] = mask * c
        return rgb

    # ── render: intensity map (ref) + binarised init / mid / final ───────────
    #   snaps order: [init, t=0, t=steps//8, t=steps//4, t=steps//2, t=steps-1]
    panels = [("intensity |z|", W,          False),
              ("init",          snaps[0],   True),
              (f"t={steps//2}", snaps[4],   True),
              (f"t={steps-1}",  snaps[5],   True)]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6))
    fig.patch.set_facecolor("#000000")
    for ax, (title, img, do_bin) in zip(axes, panels):
        if do_bin:
            ax.imshow(binar(img), interpolation="nearest")
        else:
            ax.imshow(img, cmap="magma", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(title, color="#ddd", fontsize=9)
        ax.set_facecolor("#000000")
        ax.axis("off")
    plt.tight_layout()
    out = "grid_embed_test.png"
    plt.savefig(out, dpi=150, facecolor="#000000", bbox_inches="tight")
    print(f"saved → {out}  (leak={leak}, floor={floor})")


if __name__ == "__main__":
    main()

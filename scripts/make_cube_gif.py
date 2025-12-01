import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402


def make_cube_gif(
    cube_path: Path,
    out_path: Path,
    stride: int = 5,
    lead: int = 0,
    start_hold: int = 8,
    end_hold: int = 8,
    resize: int = 240,
    duration: int = 20,
    cmap_name: str = "magma",
    p_low: float = 0.5,
    p_high: float = 99.5,
) -> None:
    """
    Gera um GIF com frame do cube + curva de luz e ponteiro sincronizado.
    lead > 0 adianta o ponteiro; lead < 0 atrasa.
    """
    cube = np.load(cube_path)
    lc = cube.sum(axis=(1, 2)).astype(float)
    N = len(lc)

    # figura base da curva (estática)
    fig = plt.figure(figsize=(5.0, 3))
    plt.plot(lc, color="#4c78a8", lw=1.1)
    plt.xlabel("Frame")
    plt.ylabel("Flux (sum pixels)")
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.xlim(0, N - 1)
    plt.tight_layout()
    fig.canvas.draw()
    base_lc = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close(fig)

    base_img = Image.fromarray(base_lc)
    w_lc, h_lc = base_img.size
    lc_min, lc_max = float(lc.min()), float(lc.max())

    # normalização para os frames do cube
    p_lo_val, p_hi_val = np.percentile(cube, [p_low, p_high])
    cmap = plt.get_cmap(cmap_name)

    def make_combo(idx: int) -> Image.Image:
        frame = cube[idx]
        frame = np.clip(frame, p_lo_val, p_hi_val)
        norm = (frame - p_lo_val) / (p_hi_val - p_lo_val + 1e-9)
        rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
        img_frame = (
            Image.fromarray(rgb, mode="RGB")
            .resize((resize, resize), resample=Image.NEAREST)
        )
        draw_frame = ImageDraw.Draw(img_frame)
        draw_frame.rectangle(
            [0, 0, img_frame.width - 1, img_frame.height - 1],
            outline=(255, 130, 60),
            width=2,
        )

        lc_img = base_img.copy()
        draw = ImageDraw.Draw(lc_img)
        ptr = min(max(idx + lead, 0), N - 1)
        x = int(ptr / (N - 1) * (w_lc - 1))
        y_val = lc[ptr]
        y = int((1 - (y_val - lc_min) / (lc_max - lc_min + 1e-9)) * (h_lc - 1))
        draw.line([(x, 0), (x, h_lc)], fill=(228, 87, 86), width=2)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(228, 87, 86))

        combo = Image.new(
            "RGB",
            (img_frame.width + lc_img.width, max(img_frame.height, lc_img.height)),
            (10, 10, 10),
        )
        combo.paste(img_frame, (0, 0))
        combo.paste(lc_img, (img_frame.width, 0))
        return combo

    frames = []
    first = make_combo(0)
    frames.extend([first] * max(start_hold, 0))

    for idx in range(0, N, max(stride, 1)):
        frames.append(make_combo(idx))

    last = frames[-1]
    frames.extend([last] * max(end_hold, 0))

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    print(f"[OK] GIF salvo em: {out_path}")
    print(f"Frames: {len(frames)} | stride={stride} | lead={lead}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Gera GIF sincronizado (frame do cube + curva de luz).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--cube", required=True, help="caminho para cube.npy")
    ap.add_argument("--out", default="runs/cube_preview.gif", help="saida GIF")
    ap.add_argument("--stride", type=int, default=5, help="passo entre frames")
    ap.add_argument("--lead", type=int, default=0, help="adianta/atrasa ponteiro")
    ap.add_argument("--start-hold", type=int, default=8, help="frames repetidos no inicio")
    ap.add_argument("--end-hold", type=int, default=8, help="frames repetidos no final")
    ap.add_argument("--resize", type=int, default=240, help="tamanho do frame (px)")
    ap.add_argument("--duration", type=int, default=20, help="ms por frame no GIF")
    ap.add_argument("--cmap", default="magma", help="paleta matplotlib")
    ap.add_argument("--p-low", type=float, default=0.5, help="percentil minimo")
    ap.add_argument("--p-high", type=float, default=99.5, help="percentil maximo")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_cube_gif(
        cube_path=Path(args.cube),
        out_path=Path(args.out),
        stride=args.stride,
        lead=args.lead,
        start_hold=args.start_hold,
        end_hold=args.end_hold,
        resize=args.resize,
        duration=args.duration,
        cmap_name=args.cmap,
        p_low=args.p_low,
        p_high=args.p_high,
    )

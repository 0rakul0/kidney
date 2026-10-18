"""Figuras cientificas reproduziveis a partir de imagens e mascaras existentes.

Nao realiza inferencia, nao cria anotacoes e nao modifica arquivos de entrada.
Exige caminhos explicitos: nenhuma figura de revisao humana e simulada.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"rim": (1.0, .25, .28), "cortex": (.0, .75, .82),
          "medulla": (1.0, .82, .15), "cec": (1.0, .56, .0)}


def read_gray(path: str) -> np.ndarray:
    a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise FileNotFoundError(f"Imagem ou mascara nao encontrada: {path}")
    return a


def mask(path: str, shape: tuple[int, int]) -> np.ndarray:
    m = read_gray(path)
    if m.shape != shape:
        raise ValueError(f"Dimensoes incompatíveis em {path}: {m.shape} vs {shape}. "
                         "Alinhe as mascaras no mesmo espaco antes de exportar.")
    return m > 0


def overlay(image, masks):
    rgb = np.repeat((image / 255.0)[..., None], 3, axis=2)
    for name, data in masks.items():
        color = np.asarray(COLORS[name])
        rgb[data] = 0.55 * rgb[data] + 0.45 * color
    return np.clip(rgb, 0, 1)


def save(axes_data, output, dpi):
    fig, axes = plt.subplots(1, len(axes_data), figsize=(4.0 * len(axes_data), 4.2))
    if len(axes_data) == 1:
        axes = [axes]
    for ax, (title, array, cmap) in zip(axes, axes_data):
        ax.imshow(array, cmap=cmap, vmin=0 if cmap == "gray" else None,
                  vmax=255 if cmap == "gray" else None)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva: {output}")


def common(parser):
    parser.add_argument("--image", required=True, help="Ultrassom B-mode original (PNG/JPG).")
    parser.add_argument("--output", required=True, help="PNG de saida.")
    parser.add_argument("--dpi", type=int, default=300)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("revisao", help="Mascara automatica versus revisao REAL.")
    common(a)
    for name in ("automatic", "reviewed"):
        a.add_argument("--" + name, required=True)
    a = sub.add_parser("multiclasse", help="Mascaras anatomicas ja geradas.")
    common(a)
    for name in ("rim", "cortex", "medulla", "cec"):
        a.add_argument("--" + name, required=True)
    a = sub.add_parser("rois", help="ROIs e razao de intensidade cortex/CEC.")
    common(a)
    a.add_argument("--cortex", required=True)
    a.add_argument("--cec", required=True)
    a = sub.add_parser("casos", help="Dois casos comparaveis: favoravel e dificil.")
    a.add_argument("--good-image", required=True)
    a.add_argument("--good-reference", required=True)
    a.add_argument("--good-prediction", required=True)
    a.add_argument("--hard-image", required=True)
    a.add_argument("--hard-reference", required=True)
    a.add_argument("--hard-prediction", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.mode == "casos":
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for row, prefix, title in [(0, "good", "Caso favoravel"), (1, "hard", "Caso dificil")]:
            image = read_gray(getattr(args, prefix + "_image"))
            ref = mask(getattr(args, prefix + "_reference"), image.shape)
            pred = mask(getattr(args, prefix + "_prediction"), image.shape)
            panels = [(title + " - original", image, "gray"),
                      ("Referencia manual", overlay(image, {"rim": ref}), None),
                      ("Predicao", overlay(image, {"rim": pred}), None)]
            for ax, (label, data, cm) in zip(axes[row], panels):
                ax.imshow(data, cmap=cm)
                ax.set_title(label, fontsize=10)
                ax.axis("off")
        fig.tight_layout()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Figura salva: {args.output}")
        return
    image = read_gray(args.image)
    if args.mode == "revisao":
        predicted = mask(args.automatic, image.shape)
        reviewed = mask(args.reviewed, image.shape)
        delta = np.zeros((*image.shape, 3), dtype=float)
        delta[reviewed & ~predicted] = (0, .8, .4)
        delta[predicted & ~reviewed] = (1, .24, .24)
        save([("Original", image, "gray"),
              ("Mascara automatica", overlay(image, {"rim": predicted}), None),
              ("Mascara revisada", overlay(image, {"rim": reviewed}), None),
              ("Diferencas (verde=adicao)", delta, None)], args.output, args.dpi)
    elif args.mode == "multiclasse":
        masks = {name: mask(getattr(args, name), image.shape)
                 for name in ("rim", "cortex", "medulla", "cec")}
        labels = np.zeros((*image.shape, 3), dtype=float)
        for name, data in masks.items():
            labels[data] = COLORS[name]
        save([("Original", image, "gray"),
              ("Mascara anatomica", labels, None),
              ("Sobreposicao", overlay(image, masks), None)], args.output, args.dpi)
    elif args.mode == "rois":
        cortex, cec = mask(args.cortex, image.shape), mask(args.cec, image.shape)
        if not cortex.any() or not cec.any():
            parser.error("ROIs devem conter pixels validos de cortex e CEC.")
        mc, me = float(image[cortex].mean()), float(image[cec].mean())
        ratio = mc / me if me > 0 else float("nan")
        save([("Original B-mode", image, "gray"),
              (f"ROIs cortex/CEC | medias {mc:.1f}/{me:.1f}", 
               overlay(image, {"cortex": cortex, "cec": cec}), None),
              (f"Razao cortex/CEC = {ratio:.3f} (exploratoria)",
               overlay(image, {"cortex": cortex, "cec": cec}), None)], args.output, args.dpi)


if __name__ == "__main__":
    main()

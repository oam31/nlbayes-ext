#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparaison de nlbayes-ext avec l'implémentation IPOL de NL-Bayes
(Lebrun, 2013), appelée comme programme externe (``ipol_reference.py``).

Images de scikit-image (domaine public) réduites à 256 × 256 : astronaut
et camera en niveaux de gris, astronaut et coffee en couleur. Bruit blanc
gaussien σ ∈ {5, 10, 20, 40} sur l'échelle [0, 255], trois tirages ; image
bruitée **quantifiée sur 8 bits** et donnée telle quelle aux deux
implémentations (le programme IPOL ne lit que du PNG 8 bits). PSNR contre
l'image propre, estimée de l'étape 1 et estimée finale, écart entre les
deux sorties, temps.

Usage ::

    export NLBAYES_IPOL_EXE=/chemin/nl-bayes_20130617/NL_Bayes
    python examples/ipol_comparison.py --out comparaison_ipol/

Dépendances de l'exemple : scikit-image, imageio, pandas, et l'exécutable
IPOL compilé par l'utilisateur (GPL-3.0+, non fourni). Durée indicative :
2 min.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ipol_reference import find_executable, ipol_denoise, quantize  # noqa: E402

SIGMAS = (5, 10, 20, 40)
DRAWS = 3


def images() -> dict:
    from skimage import data
    from skimage.color import rgb2gray
    from skimage.transform import resize

    def r(x, c):
        return resize(x, (256, 256, 3) if c else (256, 256), anti_aliasing=True).astype(np.float64)

    return {
        "astronaut (gris)": r(rgb2gray(data.astronaut()), False),
        "camera (gris)": r(data.camera() / 255.0, False),
        "astronaut (RGB)": r(data.astronaut() / 255.0, True),
        "coffee (RGB)": r(data.coffee() / 255.0, True),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exe", help="exécutable IPOL NL_Bayes (sinon NLBAYES_IPOL_EXE)")
    p.add_argument("--out", type=Path, default=Path("comparaison_ipol"))
    args = p.parse_args()
    import pandas as pd
    from skimage.metrics import peak_signal_noise_ratio as psnr

    from nlbayes_ext import denoise_gray, denoise_rgb

    exe = find_executable(args.exe)
    if exe is None:
        p.error("exécutable IPOL introuvable : --exe ou NLBAYES_IPOL_EXE")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, x in images().items():
        f = denoise_rgb if x.ndim == 3 else denoise_gray
        for s255 in SIGMAS:
            s = s255 / 255.0
            for draw in range(DRAWS):
                rng = np.random.default_rng([s255, draw])
                n = quantize(x + rng.normal(0.0, s, x.shape), (0.0, 1.0))
                r = dict(image=name, sigma=s255, draw=draw, bruitee=psnr(x, n, data_range=1.0))
                for impl, run in (("ipol", lambda b: ipol_denoise(n, s, (0.0, 1.0), exe=exe, return_basic=b)),
                                  ("ext", lambda b: f(n, sigma=s, data_range=(0.0, 1.0), return_basic=b))):
                    t = time.time()
                    out = np.clip(run(False), 0, 1)
                    r[f"{impl}_s"] = time.time() - t
                    r[impl] = psnr(x, out, data_range=1.0)
                    r[f"{impl}_etape1"] = psnr(x, np.clip(run(True), 0, 1), data_range=1.0)
                    r[f"{impl}_out"] = out
                r["ecart_sorties"] = psnr(r.pop("ipol_out"), r.pop("ext_out"), data_range=1.0)
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "results_ipol.csv", index=False)
    t = df.groupby(["image", "sigma"])[["bruitee", "ipol_etape1", "ext_etape1", "ipol", "ext", "ecart_sorties",
                                        "ipol_s", "ext_s"]].mean()
    t["ext - ipol"] = t.ext - t.ipol
    md = ["| " + " | ".join(["image", "σ"] + list(t.columns)) + " |", "|" + "---|" * (len(t.columns) + 2)]
    for (img, s), row in t.round(2).iterrows():
        md.append("| " + " | ".join([img, str(s)] + [f"{v:.2f}" for v in row]) + " |")
    (args.out / "rapport.md").write_text(
        "# nlbayes-ext contre IPOL NL-Bayes (PSNR en dB, temps en s)\n\n" + "\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appel de l'implémentation IPOL de NL-Bayes (Lebrun, 2013) comme
**programme externe**, pour comparer ses sorties à celles de nlbayes-ext.

L'exécutable ``NL_Bayes`` (GPL-3.0-or-later) n'est ni fourni ni embarqué :
l'utilisateur le compile depuis https://www.ipol.im/pub/art/2013/16/ et
donne son chemin (``exe=`` ou variable ``NLBAYES_IPOL_EXE``). Les deux
programmes ne communiquent que par fichiers PNG ; aucun code IPOL n'est
lu, copié ni lié. Seuls le README de l'archive et sa ligne de commande
documentée ont servi.

Échange : l'image est ramenée sur [0, 255] et **quantifiée sur 8 bits**
(seul format lu par le programme IPOL) ; le programme est lancé sans
re-bruitage (``add_noise = 0``). Pour une comparaison juste, donner à
nlbayes-ext la même image quantifiée (voir :func:`quantize`).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ENV_VAR = "NLBAYES_IPOL_EXE"

# Ordre des sorties de la ligne de commande IPOL (README) :
# image sigma add_noise noisy denoised basic difference bias basic_bias diff_bias useArea1 useArea2 computeBias
_OUTPUTS = ("noisy", "denoised", "basic", "diff", "bias", "basic_bias", "diff_bias")


def find_executable(exe: str | os.PathLike | None = None) -> str | None:
    """Chemin de ``NL_Bayes`` : argument, variable ``NLBAYES_IPOL_EXE``, puis ``PATH`` ; None si absent."""
    path = exe or os.environ.get(ENV_VAR) or shutil.which("NL_Bayes")
    return str(path) if path and Path(path).is_file() else None


def quantize(image: np.ndarray, data_range: tuple[float, float]) -> np.ndarray:
    """Quantification 8 bits de l'échange, ramenée dans les unités de l'image."""
    lo, hi = data_range
    q = np.clip(np.rint(255.0 * (np.asarray(image, dtype=np.float64) - lo) / (hi - lo)), 0, 255)
    return q / 255.0 * (hi - lo) + lo


def ipol_denoise(
    image: np.ndarray,
    sigma: float,
    data_range: tuple[float, float],
    exe: str | os.PathLike | None = None,
    use_area1: bool = True,
    use_area2: bool = False,
    return_basic: bool = False,
    timeout: float | None = None,
) -> np.ndarray:
    """
    Débruite ``image`` (H, W) ou (H, W, 3) avec l'exécutable IPOL.

    ``sigma`` : écart-type du bruit dans les unités de l'image ;
    ``data_range`` : dynamique ramenée à [0, 255].
    """
    import imageio.v3 as iio

    path = find_executable(exe)
    if path is None:
        raise FileNotFoundError(f"Exécutable IPOL NL_Bayes introuvable : donner exe= ou {ENV_VAR}.")
    lo, hi = data_range
    scale = hi - lo
    sigma_255 = 255.0 * sigma / scale
    with tempfile.TemporaryDirectory(prefix="nlbayes_ipol_") as tmp:
        tmp = Path(tmp)
        src = tmp / "input.png"
        iio.imwrite(src, np.clip(np.rint(255.0 * (image - lo) / scale), 0, 255).astype(np.uint8))
        outs = {name: tmp / f"{name}.png" for name in _OUTPUTS}
        cmd = [path, str(src), f"{sigma_255:.6g}", "0", *(str(outs[n]) for n in _OUTPUTS),
               str(int(use_area1)), str(int(use_area2)), "0"]
        proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout, check=False)
        result = outs["basic" if return_basic else "denoised"]
        if proc.returncode != 0 or not result.is_file():
            raise RuntimeError(f"NL_Bayes a échoué (code {proc.returncode}) :\n{proc.stdout}\n{proc.stderr}")
        den = iio.imread(result).astype(np.float64)
    if den.ndim == 3 and den.shape[-1] == 4:
        den = den[..., :3]
    return den / 255.0 * scale + lo

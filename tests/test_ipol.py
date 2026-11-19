#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparaison avec l'implémentation IPOL de NL-Bayes (Lebrun, 2013), appelée
comme programme externe (``examples/ipol_reference.py``).

- Le protocole d'échange (ordre des 13 arguments, fichiers PNG, échelle)
  est toujours vérifié, avec un faux exécutable.
- La comparaison réelle n'est lancée que si l'exécutable IPOL est fourni
  (variable ``NLBAYES_IPOL_EXE`` ; GPL-3.0+, non embarqué). Tolérances tirées
  de ``examples/ipol_comparison.py`` (σ = 20 sur [0, 255], 256 × 256) :
  gris −0,18 à −0,10 dB, couleur +0,18 à +0,20 dB face à IPOL.
"""
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from _helpers import SEED, astronaut_gray, astronaut_rgb, psnr  # noqa: E402
from ipol_reference import find_executable, ipol_denoise, quantize  # noqa: E402
from nlbayes_ext import denoise_gray, denoise_rgb  # noqa: E402

pytest.importorskip("imageio")
IPOL = find_executable()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="script exécutable POSIX")
def test_command_line_protocol(tmp_path):
    """Faux NL_Bayes qui recopie l'entrée comme sortie : vérifie arguments, échelle et quantification."""
    fake = tmp_path / "NL_Bayes"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import shutil, sys\n"
        "a = sys.argv[1:]\n"
        "assert len(a) == 13, a\n"
        "assert a[2] == '0'  # pas de re-bruitage\n"
        "assert abs(float(a[1]) - 25.5) < 1e-6, a[1]\n"
        "shutil.copy(a[0], a[4])\n"
        "shutil.copy(a[0], a[5])\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    img = np.clip(astronaut_gray(32), 0, 1)
    out = ipol_denoise(img, 0.1, (0.0, 1.0), exe=fake)
    assert out.shape == img.shape
    assert np.max(np.abs(out - img)) <= 0.5 / 255 + 1e-9
    rgb = astronaut_rgb(32)
    assert ipol_denoise(rgb, 0.1, (0.0, 1.0), exe=fake).shape == rgb.shape


def test_missing_executable(monkeypatch, tmp_path):
    monkeypatch.delenv("NLBAYES_IPOL_EXE", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        ipol_denoise(np.zeros((8, 8)), 0.1, (0.0, 1.0))


def _case(rgb: bool):
    x = astronaut_rgb(256) if rgb else astronaut_gray(256)
    s = 20.0 / 255.0
    noisy = quantize(x + np.random.default_rng(SEED).normal(0.0, s, x.shape), (0.0, 1.0))
    return x, noisy, s


@pytest.mark.skipif(IPOL is None, reason="exécutable IPOL NL_Bayes non fourni (NLBAYES_IPOL_EXE)")
def test_gray_matches_ipol():
    x, noisy, s = _case(False)
    ref = ipol_denoise(noisy, s, (0.0, 1.0))
    out = denoise_gray(noisy, sigma=s, data_range=(0.0, 1.0))
    assert psnr(x, np.clip(out, 0, 1), data_range=1.0) > psnr(x, ref, data_range=1.0) - 0.4
    assert psnr(ref, np.clip(out, 0, 1), data_range=1.0) > 35.0


@pytest.mark.skipif(IPOL is None, reason="exécutable IPOL NL_Bayes non fourni (NLBAYES_IPOL_EXE)")
def test_rgb_matches_ipol():
    """Étape 2 couleur à canaux filtrés ensemble (tableau 1 de l'article) : au niveau d'IPOL ou mieux."""
    x, noisy, s = _case(True)
    ref = ipol_denoise(noisy, s, (0.0, 1.0))
    out = denoise_rgb(noisy, sigma=s, data_range=(0.0, 1.0))
    assert psnr(x, np.clip(out, 0, 1), data_range=1.0) > psnr(x, ref, data_range=1.0) - 0.1
    assert psnr(ref, np.clip(out, 0, 1), data_range=1.0) > 35.0

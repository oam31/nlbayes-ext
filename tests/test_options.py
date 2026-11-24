#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Options ``noise`` par canal et traitement par tuiles (``tile``, ``margin``).
"""
import numpy as np
import pytest

from _helpers import SEED, astronaut_gray, astronaut_rgb, psnr
from nlbayes_ext import channel_map, denoise_cfa, denoise_gray, denoise_rgb
from nlbayes_ext.vst import add_poisson_gaussian_noise

PEAK = 300.0


def _raw_rgb_per_channel(gains, reads, size=96):
    """Image RGB brute, gain et bruit de lecture propres à chaque canal (DN)."""
    x = astronaut_rgb(size) * PEAK
    raw = np.stack([add_poisson_gaussian_noise(x[..., c] * gains[c], gains[c], reads[c], seed=SEED + c)
                    for c in range(3)], -1)
    return x, raw


def _models(gains, reads):
    # var(z) = a + b·z en DN, z = e⁻ / K : a = lecture², b = 1/K
    return tuple((r ** 2, 1.0 / g) for g, r in zip(gains, reads))


def test_per_channel_identical_models_equal_common_model():
    gains, reads = (1.0, 1.0, 1.0), (3.0, 3.0, 3.0)
    _, raw = _raw_rgb_per_channel(gains, reads, 64)
    a = denoise_rgb(raw, noise=(9.0, 1.0))
    b = denoise_rgb(raw, noise=_models(gains, reads), offset=(0.0, 0.0, 0.0))
    assert np.allclose(a, b)


def test_per_channel_noise_helps_when_gains_differ():
    """Gains différents par canal : le modèle par canal bat un modèle commun (celui du vert)."""
    gains, reads = (0.5, 1.0, 2.0), (2.0, 3.0, 5.0)
    x, raw = _raw_rgb_per_channel(gains, reads)
    ref = x                                                    # signal moyen en DN
    per = denoise_rgb(raw, noise=_models(gains, reads))
    com = denoise_rgb(raw, noise=_models(gains, reads)[1])
    assert psnr(ref, per, data_range=PEAK) > psnr(ref, com, data_range=PEAK) + 0.3


def test_per_channel_noise_cfa():
    gains, reads = (0.5, 1.0, 2.0), (2.0, 3.0, 5.0)
    x = astronaut_rgb(64) * PEAK
    cm = channel_map(64, 64, "RGGB")
    mos = np.take_along_axis(x, cm[..., None], 2)[..., 0]
    raw = np.empty_like(mos)
    rng = np.random.default_rng(SEED)
    for c in range(3):
        m = cm == c
        raw[m] = rng.poisson(mos[m] * gains[c]) / gains[c] + rng.normal(0, reads[c], m.sum())
    per = denoise_cfa(raw, noise=_models(gains, reads), n_iter=2, n_threads=1)
    com = denoise_cfa(raw, noise=_models(gains, reads)[1], n_iter=2, n_threads=1)
    assert per.shape == x.shape and np.all(np.isfinite(per))
    # En CFA, pas de gain mesuré du modèle par canal (−0,08 à +0,04 dB, docs/technique/reference.md §2) :
    # on vérifie seulement qu'il ne dégrade pas.
    assert psnr(x, per, data_range=PEAK) > psnr(x, com, data_range=PEAK) - 0.3


@pytest.mark.parametrize("kwargs", [
    {"noise": ((1.0, 1.0), (1.0, 1.0))},
    {"noise": (1.0, 1.0), "offset": (0.0, 1.0)},
])
def test_per_channel_invalid(kwargs):
    with pytest.raises(ValueError):
        denoise_rgb(np.ones((32, 32, 3)) * 50, **kwargs)
    with pytest.raises(ValueError):
        denoise_gray(np.ones((32, 32)) * 50, noise=((1.0, 1.0),) * 3)


def test_tiles_close_to_whole_image():
    x = astronaut_gray(128)
    n = x + np.random.default_rng(SEED).normal(0, 0.1, x.shape)
    whole = denoise_gray(n, sigma=0.1, data_range=(n.min(), n.max()))
    tiled = denoise_gray(n, sigma=0.1, tile=48, margin=32)
    assert tiled.shape == x.shape
    assert abs(psnr(x, tiled, data_range=1) - psnr(x, whole, data_range=1)) < 0.15


def test_tiles_rgb_and_cfa_phase():
    x = astronaut_rgb(96)
    n = x + np.random.default_rng(SEED).normal(0, 0.05, x.shape)
    whole = denoise_rgb(n, sigma=0.05)
    tiled = denoise_rgb(n, sigma=0.05, tile=40, margin=24)
    assert abs(psnr(x, tiled, data_range=1) - psnr(x, whole, data_range=1)) < 0.15
    mos = np.take_along_axis(x, channel_map(96, 96, "GRBG")[..., None], 2)[..., 0]
    mos = mos + np.random.default_rng(SEED).normal(0, 0.03, mos.shape)
    w = denoise_cfa(mos, sigma=0.03, pattern="GRBG", n_iter=2, n_threads=1)
    t = denoise_cfa(mos, sigma=0.03, pattern="GRBG", n_iter=2, tile=40, margin=24, n_threads=1)
    # une phase de motif fausse dans une tuile coûterait plusieurs dB
    assert abs(psnr(x, np.clip(t, 0, 1), data_range=1) - psnr(x, np.clip(w, 0, 1), data_range=1)) < 0.3
    with pytest.raises(ValueError):
        denoise_cfa(mos, sigma=0.03, tile=41, margin=24)


def test_tiles_with_noise_model_use_global_settings():
    x = astronaut_gray(96) * PEAK
    raw = add_poisson_gaussian_noise(x, 1.0, 3.0, seed=SEED)
    whole = denoise_gray(raw, noise=(9.0, 1.0))
    tiled = denoise_gray(raw, noise=(9.0, 1.0), tile=48, margin=32)
    assert abs(psnr(x, tiled, data_range=PEAK) - psnr(x, whole, data_range=PEAK)) < 0.15

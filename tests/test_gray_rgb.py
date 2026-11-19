#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
denoise_gray et denoise_rgb : qualité, deux étapes, paramètres de l'article
et imbrications, moteurs vectorisé / boucle, modes couleur, modèle de bruit
noise=(a, b), estimation de σ, contrôles.
"""

import numpy as np
import pytest

from _helpers import MIN_GAIN_DB, SEED, SIGMA, astronaut_gray, psnr
from nlbayes_ext import YUV_MATRIX, article_parameters, denoise_gray, denoise_rgb, resolve_parameters
from nlbayes_ext.vst import add_poisson_gaussian_noise


def gain(clean, noisy, out):
    return psnr(clean, out, data_range=1.0) - psnr(clean, noisy, data_range=1.0)


def test_gray_denoises(clean, noisy):
    out = denoise_gray(noisy, sigma=SIGMA)
    assert out.shape == noisy.shape and np.all(np.isfinite(out))
    assert gain(clean, noisy, out) > 6.0          # mesuré : 7,0 dB (σ = 0,1, 128 × 128)


def test_second_step_improves(clean, noisy):
    basic = denoise_gray(noisy, sigma=SIGMA, return_basic=True)
    final = denoise_gray(noisy, sigma=SIGMA)
    assert psnr(clean, final, data_range=1.0) > psnr(clean, basic, data_range=1.0)


def test_estimated_sigma(clean, noisy):
    assert gain(clean, noisy, denoise_gray(noisy)) > MIN_GAIN_DB


def test_scale_invariance(clean):
    """Le résultat ne dépend pas des unités (image 12 bits)."""
    noisy12 = (clean + np.random.default_rng(SEED).normal(0, SIGMA, clean.shape)) * 4095.0
    a = denoise_gray(noisy12 / 4095.0, sigma=SIGMA)
    b = denoise_gray(noisy12, sigma=SIGMA * 4095.0)
    assert np.allclose(a * 4095.0, b, rtol=1e-4, atol=1e-2)


def test_article_parameters():
    p = article_parameters(25.0)
    assert (p["k1"], p["k2"], p["N1"], p["N2"], p["n1"], p["n2"]) == (5, 5, 60, 60, 35, 35)
    assert (p["gamma"], p["beta1"], p["beta2"], p["tau0"]) == (1.05, 1.0, 1.2, 4.0)
    p = article_parameters(80.0)
    assert (p["k1"], p["k2"], p["N1"], p["beta2"]) == (7, 7, 90, 1.0)
    assert article_parameters(15.0, color=True)["k1"] == 3


def test_parameter_nesting():
    """N suit k, n = α·k, N ≥ k² + 1."""
    p = resolve_parameters(25.0, False, {"k1": 7, "window_factor": 5})
    assert (p["N1"], p["n1"], p["n2"]) == (90, 35, 25)
    assert resolve_parameters(25.0, False, {"k1": 7, "N1": 20})["N1"] == 50


def test_unknown_parameter(noisy):
    with pytest.raises(TypeError):
        denoise_gray(noisy, sigma=SIGMA, not_a_param=1)


def test_engines_agree(clean, noisy):
    loop = denoise_gray(noisy, sigma=SIGMA, engine="loop")
    vec = denoise_gray(noisy, sigma=SIGMA)
    assert abs(psnr(clean, loop, data_range=1.0) - psnr(clean, vec, data_range=1.0)) < 0.15


def test_threads_do_not_change_result(noisy):
    assert np.allclose(denoise_gray(noisy, sigma=SIGMA, n_threads=1), denoise_gray(noisy, sigma=SIGMA, n_threads=4))


def test_yuv_matrix_is_orthonormal():
    assert np.allclose(YUV_MATRIX @ YUV_MATRIX.T, np.eye(3))


@pytest.mark.parametrize("mode", ["yuv", "rgb", "channel"])
def test_rgb_modes(rgb_pair, mode):
    clean_rgb, noisy_rgb = rgb_pair
    out = denoise_rgb(noisy_rgb, sigma=SIGMA, color_mode=mode)
    assert out.shape == noisy_rgb.shape and np.all(np.isfinite(out))
    assert gain(clean_rgb, noisy_rgb, out) > MIN_GAIN_DB


def test_rgb_estimated_sigma(rgb_pair):
    clean_rgb, noisy_rgb = rgb_pair
    assert gain(clean_rgb, noisy_rgb, denoise_rgb(noisy_rgb)) > MIN_GAIN_DB


def test_rgb_rejects_loop_engine_and_gray(rgb_pair, noisy):
    with pytest.raises(ValueError):
        denoise_rgb(rgb_pair[1], sigma=SIGMA, engine="loop", color_mode="yuv")
    with pytest.raises(ValueError):
        denoise_rgb(noisy, sigma=SIGMA)
    with pytest.raises(ValueError):
        denoise_gray(rgb_pair[1], sigma=SIGMA)


def test_noise_model_beats_global_sigma_at_low_flux():
    """
    Bruit poissonnien-gaussien à faible flux (pic 5 e⁻) : noise=(a, b), avec
    stabilisation interne, bat un σ global équivalent d'au moins 1 dB.
    """
    k, rn, off, peak = 2.0, 0.25, 100.0, 5.0
    signal = astronaut_gray(128) * peak
    raw = add_poisson_gaussian_noise(signal, k, rn, off, seed=SEED)
    ref = signal / k + off
    a, b = rn ** 2, 1.0 / k
    stab = denoise_gray(raw, noise=(a, b), offset=off)
    flat = denoise_gray(raw, sigma=float(np.sqrt(a + b * (ref - off).mean())))
    assert stab.shape == raw.shape
    assert psnr(ref, stab, data_range=peak / k) > psnr(ref, flat, data_range=peak / k) + 1.0


def test_noise_model_rgb_and_sigma_scale():
    k, rn, off, peak = 1.0, 2.0, 0.0, 300.0
    from _helpers import astronaut_rgb

    signal = astronaut_rgb(64) * peak
    raw = add_poisson_gaussian_noise(signal, k, rn, off, seed=SEED)
    a = denoise_rgb(raw, noise=(rn ** 2, 1.0 / k))
    b = denoise_rgb(raw, noise=(rn ** 2, 1.0 / k), sigma_scale=1.5)
    assert a.shape == raw.shape and not np.allclose(a, b)
    assert psnr(signal, a, data_range=peak) > psnr(signal, raw, data_range=peak) + MIN_GAIN_DB


@pytest.mark.parametrize("kwargs", [
    {"sigma": 1.0, "noise": (1.0, 1.0)}, {"noise": (1.0, 0.0)}, {"noise": (1.0, 1.0), "data_range": (0, 1)},
    {"sigma": 1.0, "offset": 2.0}, {"sigma": 1.0, "sigma_scale": -1.0}, {"noise": (1.0, 1.0), "inverse": "x"},
    {"sigma": -1.0},
])
def test_invalid_noise_arguments(noisy, kwargs):
    with pytest.raises(ValueError):
        denoise_gray(noisy, **kwargs)

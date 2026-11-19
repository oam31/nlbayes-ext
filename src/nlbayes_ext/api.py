#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API publique : trois débruiteurs nommés d'après leur entrée.

- :func:`denoise_gray` : image en niveaux de gris (H, W) ;
- :func:`denoise_rgb`  : image couleur (H, W, 3), par exemple démosaïquée ;
- :func:`denoise_cfa`  : mosaïque de Bayer (H, W) → image RGB (H, W, 3),
  débruitage et démosaïçage conjoints.

Le bruit se donne de deux façons, exclusives :

- ``sigma`` : bruit blanc gaussien d'écart-type connu (unités de l'image),
  estimé si ni ``sigma`` ni ``noise`` ne sont fournis ;
- ``noise=(a, b)`` : bruit poissonnien-gaussien du capteur,
  var(z) = a + b·(z − offset), a plancher (DN²), b pente (DN). La variance
  est alors stabilisée en interne (transformée d'Anscombe généralisée), le
  débruiteur travaille avec σ = 1, et l'image revient dans ses unités par
  l'inverse exacte non biaisée (:mod:`nlbayes_ext.vst`).

``sigma_scale`` multiplie le σ transmis au débruiteur (dans le domaine
stabilisé avec ``noise``) : 1 pour une image brute ; ≈ √κ_Y pour une image
démosaïquée, κ_Y étant le gain de variance du bruit en luminance apporté
par le démosaïqueur (≈ 1,35 pour Hamilton-Adams, d'où ≈ 1,1).
"""
from __future__ import annotations

import os
from typing import Callable

import numpy as np

from ._cfa import CFA_PATTERNS, _estimate_mosaic_sigma, nl_bayes_cfa_core
from ._gray import nl_bayes_core
from ._utils import check_image, estimate_noise_sigma
from .vst import INVERSES, generalized_anscombe, inverse_generalized_anscombe, noise_to_gat

__all__ = ["denoise_gray", "denoise_rgb", "denoise_cfa"]


def _check_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Image couleur (H, W, 3) attendue, reçu {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("L'image contient des valeurs non finies (NaN/inf).")
    return arr


def _noise_setup(img, sigma, noise, offset, sigma_scale, data_range, inverse, estimate: Callable):
    """
    Prépare l'entrée du débruiteur : renvoie (image de travail, σ, data_range, fonction de retour).
    """
    if sigma is not None and noise is not None:
        raise ValueError("Donner sigma (bruit gaussien) ou noise=(a, b) (bruit du capteur), pas les deux.")
    if not sigma_scale > 0:
        raise ValueError(f"sigma_scale doit être > 0, reçu {sigma_scale}.")
    if inverse not in INVERSES:
        raise ValueError(f"inverse doit valoir {INVERSES}, reçu '{inverse}'.")
    if noise is None:
        if offset != 0.0:
            raise ValueError("offset ne sert qu'avec noise=(a, b).")
        s = estimate(img) if sigma is None else float(sigma)
        if s < 0:
            raise ValueError(f"sigma doit être positif ou nul, reçu {s}.")
        return img, s * sigma_scale, data_range, (lambda x: x), None
    if data_range is not None:
        raise ValueError("data_range ne sert pas avec noise=(a, b) : l'image est d'abord stabilisée.")
    gain, read = noise_to_gat(noise)
    fwd = lambda x: generalized_anscombe(x, gain, read, offset)  # noqa: E731
    back = lambda d: inverse_generalized_anscombe(d, gain, read, offset, method=inverse)  # noqa: E731
    return fwd(img), float(sigma_scale), None, back, fwd


def denoise_gray(
    image: np.ndarray,
    sigma: float | None = None,
    noise: tuple[float, float] | None = None,
    *,
    offset: float = 0.0,
    sigma_scale: float = 1.0,
    inverse: str = "exact_unbiased",
    data_range: tuple[float, float] | None = None,
    use_area1: bool = True,
    use_area2: bool = False,
    return_basic: bool = False,
    engine: str = "vectorized",
    step: int = 3,
    chunk: int = 256,
    n_threads: int | None = None,
    **params,
) -> np.ndarray:
    """
    Débruite une image en niveaux de gris par NL-Bayes.

    Parameters
    ----------
    image : np.ndarray
        Image bruitée (H, W).
    sigma : float, optional
        Écart-type d'un bruit blanc gaussien (unités de l'image). Estimé si
        ni ``sigma`` ni ``noise`` ne sont fournis.
    noise : (float, float), optional
        Modèle du capteur var(z) = a + b·(z − ``offset``) : stabilisation de
        variance interne, puis inverse exacte non biaisée.
    offset : float
        Niveau d'obscurité (DN), avec ``noise`` seulement.
    sigma_scale : float
        Multiplie le σ transmis au débruiteur. Par défaut 1.
    inverse : {"exact_unbiased", "asymptotic", "algebraic"}
        Inverse de la stabilisation (avec ``noise``).
    data_range : tuple(float, float), optional
        Dynamique ramenée à [0, 255], échelle des paramètres de l'article
        (sans ``noise``). Par défaut, le min/max de l'image.
    use_area1, use_area2 : bool
        Critère des zones homogènes aux étapes 1 / 2 (article : oui / non).
    return_basic : bool
        Renvoyer l'estimée de l'étape 1.
    engine : {"vectorized", "loop"}
        ``"vectorized"`` : calcul par lots, références sur une grille de pas
        ``step`` ; ``"loop"`` : version fidèle à l'article (« paste trick »),
        plus lente.
    step, chunk, n_threads :
        Pas de la grille des références, taille des lots, fils d'exécution.
    **params
        Paramètres de l'article à remplacer : ``k1``, ``k2``, ``n1``, ``n2``,
        ``N1``, ``N2``, ``gamma``, ``beta1``, ``beta2``, ``tau0`` (niveaux² sur
        [0, 255]), ``window_factor`` (n = α·k).

    Returns
    -------
    np.ndarray
        Image débruitée (H, W), ``float64``, unités de l'entrée.
    """
    img = check_image(image)
    work, s, rng, back, _ = _noise_setup(img, sigma, noise, offset, sigma_scale, data_range, inverse,
                                         estimate_noise_sigma)
    out = nl_bayes_core(work, sigma=s, data_range=rng, use_area1=use_area1, use_area2=use_area2,
                        return_basic=return_basic, engine=engine, step=step, chunk=chunk,
                        n_threads=n_threads, **params)
    return back(out)


def denoise_rgb(
    image: np.ndarray,
    sigma: float | None = None,
    noise: tuple[float, float] | None = None,
    *,
    color_mode: str = "yuv",
    offset: float = 0.0,
    sigma_scale: float = 1.0,
    inverse: str = "exact_unbiased",
    data_range: tuple[float, float] | None = None,
    use_area1: bool = True,
    use_area2: bool = False,
    return_basic: bool = False,
    engine: str = "vectorized",
    step: int = 3,
    chunk: int = 256,
    n_threads: int | None = None,
    **params,
) -> np.ndarray:
    """
    Débruite une image couleur (H, W, 3) par NL-Bayes.

    Parameters
    ----------
    image : np.ndarray
        Image bruitée (H, W, 3), par exemple démosaïquée.
    color_mode : {"yuv", "rgb", "channel"}
        ``"yuv"`` (défaut, choix de l'article) : étape 1 dans l'espace
        orthonormé Y₀U₀V₀, groupes formés sur Y₀ ; étape 2 en RGB.
        ``"rgb"`` : les deux étapes en RGB, groupes communs. ``"channel"`` :
        chaque canal comme une image en niveaux de gris.
    sigma, noise, offset, sigma_scale, inverse, data_range, use_area1, use_area2,
    return_basic, engine, step, chunk, n_threads, **params :
        Comme :func:`denoise_gray`. ``noise`` s'applique aux trois canaux.
        Sur une image démosaïquée, ``sigma_scale`` ≈ √κ_Y (1,1 pour
        Hamilton-Adams). ``engine="loop"`` exige ``color_mode="channel"``.

    Returns
    -------
    np.ndarray
        Image débruitée (H, W, 3), ``float64``, unités de l'entrée.
    """
    img = _check_rgb(image)

    def est(x):
        return float(np.mean([estimate_noise_sigma(x[..., c]) for c in range(3)]))

    work, s, rng, back, _ = _noise_setup(img, sigma, noise, offset, sigma_scale, data_range, inverse, est)
    out = nl_bayes_core(work, sigma=s, data_range=rng, use_area1=use_area1, use_area2=use_area2,
                        return_basic=return_basic, color_mode=color_mode, engine=engine, step=step,
                        chunk=chunk, n_threads=n_threads, **params)
    return back(out)


def denoise_cfa(
    image: np.ndarray,
    sigma: float | None = None,
    noise: tuple[float, float] | None = None,
    *,
    pattern: str = "RGGB",
    offset: float = 0.0,
    sigma_scale: float = 1.0,
    inverse: str = "exact_unbiased",
    data_range: tuple[float, float] | None = None,
    k: int = 3,
    window_factor: float = 7.0,
    N: int = 60,
    beta: float = 1.0,
    tau0: float = 4.0,
    n_iter: int = 8,
    pilot: np.ndarray | str = "bilinear",
    pilot_kwargs: dict | None = None,
    use_area: bool = False,
    area_z: float = 2.0,
    area_gamma: float | None = None,
    area_mean: str = "mosaic",
    step: int = 3,
    chunk: int = 256,
    n_threads: int | None = None,
) -> np.ndarray:
    """
    Débruite et démosaïque une mosaïque de Bayer (NL-Bayes CFA).

    Estimateur des estimateurs linéaires par morceaux (Yu, Sapiro & Mallat,
    2012) appliqué aux groupes non locaux de NL-Bayes : pour un patch couleur
    x observé par y = U·x + n, x̂ = μ + C·Uᵀ(U·C·Uᵀ + βσ²I)⁻¹(y − U·μ), μ et C
    étant estimées sur une image pilote. Voir docs/REFERENCE_TECHNIQUE.md.

    Parameters
    ----------
    image : np.ndarray
        Mosaïque bruitée (H, W), une valeur mesurée par pixel.
    sigma, noise, offset, sigma_scale, inverse :
        Comme :func:`denoise_gray`. Avec ``noise=(a, b)``, la mosaïque est
        stabilisée (son bruit, indépendant d'un photosite à l'autre, devient
        blanc d'écart-type 1) : c'est l'usage recommandé sur données brutes.
        Sans ``noise`` ni ``sigma``, σ est estimé sur les quatre sous-plans.
    pattern : {"RGGB", "BGGR", "GRBG", "GBRG"}
        Motif 2 × 2, ligne par ligne, à partir du pixel (0, 0).
    data_range : tuple(float, float), optional
        Dynamique ramenée à [0, 255] (échelle de τ₀), sans ``noise``.
    k, window_factor, N, beta, tau0 :
        Côté des patchs (D = 3k²), fenêtre n = α·k, taille minimale des
        groupes, facteur de σ² dans U·C·Uᵀ + βσ²I, plancher de similarité
        (niveaux² sur [0, 255]).
    n_iter : int
        Nombre de passes ; chaque passe prend la sortie de la précédente
        comme pilote. Par défaut 8.
    pilot : "bilinear", "none" ou np.ndarray
        Pilote initial : démosaïçage bilinéaire puis NL-Bayes couleur
        (défaut), bilinéaire seul, ou image (H, W, 3) fournie dans les unités
        de la mosaïque. Le pilote ne sert qu'à former les groupes et estimer
        μ et C ; aucune de ses valeurs ne passe dans le résultat.
    pilot_kwargs : dict, optional
        Paramètres du NL-Bayes couleur du pilote ``"bilinear"``.
    use_area, area_z, area_gamma, area_mean :
        Critère des zones homogènes (désactivé par défaut ; utile à faible
        flux), seuil statistique z ou seuil fixe γ, valeur des groupes plats
        (``"mosaic"`` ou ``"pilot"``).
    step, chunk, n_threads :
        Pas de la grille des références, taille des lots, fils d'exécution.

    Returns
    -------
    np.ndarray
        Image RGB (H, W, 3), ``float64``, unités de la mosaïque.
    """
    mosaic = check_image(image)
    if pattern.upper() not in CFA_PATTERNS:
        raise ValueError(f"Motif inconnu '{pattern}' (attendu : {CFA_PATTERNS}).")
    work, s, rng, back, fwd = _noise_setup(mosaic, sigma, noise, offset, sigma_scale, data_range, inverse,
                                           _estimate_mosaic_sigma)
    if fwd is not None and not isinstance(pilot, str):
        pilot = fwd(np.asarray(pilot, dtype=np.float64))
    out = nl_bayes_cfa_core(
        work, sigma=s, pattern=pattern, data_range=rng, k=k, window_factor=window_factor, N=N, beta=beta,
        tau0=tau0, n_iter=n_iter, pilot=pilot, pilot_kwargs=pilot_kwargs, use_area=use_area, area_z=area_z,
        area_gamma=area_gamma, area_mean=area_mean, step=step, chunk=chunk,
        n_threads=n_threads if n_threads is not None else min(os.cpu_count() or 1, 8),
    )
    return back(out)

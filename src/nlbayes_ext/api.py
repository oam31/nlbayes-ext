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

``noise`` accepte aussi **un couple par canal** (R, G, B), avec ``offset``
scalaire ou à trois valeurs, pour ``denoise_rgb`` et ``denoise_cfa`` : chaque
canal est alors stabilisé avec son propre modèle.

``sigma_scale`` multiplie le σ transmis au débruiteur (dans le domaine
stabilisé avec ``noise``) : 1 pour une image brute ; √κ_Y + 0,2 pour une
image démosaïquée, κ_Y étant le gain de variance du bruit en luminance
apporté par le démosaïqueur (≈ 1,35 pour Hamilton-Adams, d'où ≈ 1,36).

**Tuiles** (``tile``, ``margin``) : l'image est traitée par tuiles de côté
``tile``, chacune élargie de ``margin`` pixels pris dans l'image voisine
(rabattus aux bords de l'image), seul l'intérieur étant gardé. σ et la
dynamique (échelle des paramètres de l'article) sont calculés une fois pour
toute l'image, de sorte que chaque tuile ait les mêmes réglages. Sert à
limiter la mémoire sur les grandes images ; le résultat diffère très peu du
traitement d'un seul bloc (positions des patchs de référence, bords des
tuiles).
"""
from __future__ import annotations

import os
from typing import Callable

import numpy as np

from ._cfa import CFA_PATTERNS, _estimate_mosaic_sigma, channel_map, nl_bayes_cfa_core
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


def _per_channel(noise, offset, n_channels: int):
    """Liste de (gain, bruit de lecture, offset) par canal ; un seul élément si le modèle est commun."""
    arr = np.asarray(noise, dtype=object)
    couples = [noise] if arr.ndim == 1 or (arr.ndim == 0) else list(noise)
    if len(couples) not in (1, n_channels) or (len(couples) > 1 and n_channels == 1):
        raise ValueError("noise : un couple (a, b), ou un couple par canal (R, G, B) pour une image couleur "
                         f"ou une mosaïque ; reçu {noise!r}.")
    offs = np.atleast_1d(np.asarray(offset, dtype=np.float64))
    if offs.size not in (1, len(couples)) and not (len(couples) == 1 and offs.size == n_channels):
        raise ValueError(f"offset : une valeur, ou une par canal ; reçu {offset!r}.")
    n = max(len(couples), offs.size)
    couples = couples * n if len(couples) == 1 else couples
    offs = np.repeat(offs, n) if offs.size == 1 else offs
    out = []
    for (cpl, off) in zip(couples, offs):
        gain, read = noise_to_gat(cpl)
        out.append((gain, read, float(off)))
    return out


def _by_channel(x: np.ndarray, chan: np.ndarray | None, models, func) -> np.ndarray:
    """Applique ``func(valeurs, gain, read, offset)`` canal par canal (``chan`` : canal de chaque valeur)."""
    if len(models) == 1:
        return func(x, *models[0])
    out = np.empty_like(x, dtype=np.float64)
    for c, model in enumerate(models):
        m = chan == c
        out[m] = func(x[m], *model)
    return out


def _rgb_channels(shape) -> np.ndarray:
    return np.broadcast_to(np.arange(3), shape)


def _noise_setup(img, sigma, noise, offset, sigma_scale, data_range, inverse, estimate: Callable,
                 chan_in=None, n_channels: int = 1):
    """
    Prépare l'entrée du débruiteur : renvoie (image de travail, σ, data_range, retour, aller pour une
    image RGB). ``chan_in`` : canal de chaque valeur de ``img`` (None : un seul canal).
    """
    if sigma is not None and noise is not None:
        raise ValueError("Donner sigma (bruit gaussien) ou noise=(a, b) (bruit du capteur), pas les deux.")
    if not sigma_scale > 0:
        raise ValueError(f"sigma_scale doit être > 0, reçu {sigma_scale}.")
    if inverse not in INVERSES:
        raise ValueError(f"inverse doit valoir {INVERSES}, reçu '{inverse}'.")
    if noise is None:
        if np.any(np.asarray(offset) != 0.0):
            raise ValueError("offset ne sert qu'avec noise=(a, b).")
        s = estimate(img) if sigma is None else float(sigma)
        if s < 0:
            raise ValueError(f"sigma doit être positif ou nul, reçu {s}.")
        return img, s * sigma_scale, data_range, (lambda x: x), None
    if data_range is not None:
        raise ValueError("data_range ne sert pas avec noise=(a, b) : l'image est d'abord stabilisée.")
    models = _per_channel(noise, offset, n_channels)

    def gat(v, g, r, o):
        return generalized_anscombe(v, g, r, o)

    def inv(v, g, r, o):
        return inverse_generalized_anscombe(v, g, r, o, method=inverse)

    work = _by_channel(img, chan_in, models, gat)

    def back(d):
        return _by_channel(d, _rgb_channels(d.shape) if d.ndim == 3 else None, models, inv)

    def fwd_rgb(x):
        return _by_channel(x, _rgb_channels(x.shape), models, gat)

    return work, float(sigma_scale), None, back, fwd_rgb


def _tiled(run, work: np.ndarray, tile: int | None, margin: int, extra=None, even: bool = False) -> np.ndarray:
    """
    ``run(sous_image, sous_extra)`` sur toute l'image ou par tuiles de côté ``tile`` élargies de
    ``margin`` (prises dans l'image, rabattues aux bords), dont on ne garde que l'intérieur. ``even`` :
    tuiles et marges paires (mosaïque : la phase du motif est conservée).
    """
    if tile is None:
        return run(work, extra)
    tile, margin = int(tile), int(margin)
    if tile < 8 or margin < 0:
        raise ValueError(f"tile ≥ 8 et margin ≥ 0 sont requis, reçu tile={tile}, margin={margin}.")
    if even and (tile % 2 or margin % 2):
        raise ValueError("Mosaïque : tile et margin doivent être pairs (phase du motif de Bayer).")
    h, w = work.shape[:2]

    def starts(n):
        if n <= tile:
            return [(0, n)]
        last = n - tile
        last -= last % 2 if even else 0
        st = list(range(0, last, tile)) + [last]
        return [(a, (st[k + 1] if k + 1 < len(st) else n)) for k, a in enumerate(st)]

    out = None
    for i0, i1 in starts(h):
        for j0, j1 in starts(w):
            a, b = max(0, i0 - margin), min(h, i1 + margin)
            c, d = max(0, j0 - margin), min(w, j1 + margin)
            sub_extra = extra[a:b, c:d] if isinstance(extra, np.ndarray) else extra
            res = run(work[a:b, c:d], sub_extra)
            if out is None:
                out = np.empty((h, w) + res.shape[2:], dtype=np.float64)
            out[i0:i1, j0:j1] = res[i0 - a:i1 - a, j0 - c:j1 - c]
    return out


def _global_range(work: np.ndarray, rng, tile):
    """Dynamique commune à toutes les tuiles (sinon chacune prendrait son propre min/max)."""
    if tile is None or rng is not None:
        return rng
    return float(work.min()), float(work.max())

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
    tile: int | None = None,
    margin: int = 32,
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
    tile, margin : int, optional
        Traitement par tuiles de côté ``tile`` (None : un seul bloc),
        élargies de ``margin`` pixels (32 par défaut, au moins la moitié de
        la fenêtre de recherche plus le côté d'un patch).
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
    rng = _global_range(work, rng, tile)
    out = _tiled(lambda x, _: nl_bayes_core(x, sigma=s, data_range=rng, use_area1=use_area1, use_area2=use_area2,
                                            return_basic=return_basic, engine=engine, step=step, chunk=chunk,
                                            n_threads=n_threads, **params),
                 work, tile, margin)
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
    tile: int | None = None,
    margin: int = 32,
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
        Comme :func:`denoise_gray`. ``noise`` : un couple (a, b) commun aux
        trois canaux, ou un couple par canal ``((a_R, b_R), (a_G, b_G),
        (a_B, b_B))`` (``offset`` : une valeur ou trois). Sur une image
        démosaïquée, ``sigma_scale`` ≈ √κ_Y + 0,2 (1,36 pour Hamilton-Adams).
        ``engine="loop"`` exige ``color_mode="channel"``.
    tile, margin :
        Comme :func:`denoise_gray`.

    Returns
    -------
    np.ndarray
        Image débruitée (H, W, 3), ``float64``, unités de l'entrée.
    """
    img = _check_rgb(image)

    def est(x):
        return float(np.mean([estimate_noise_sigma(x[..., c]) for c in range(3)]))

    work, s, rng, back, _ = _noise_setup(img, sigma, noise, offset, sigma_scale, data_range, inverse, est,
                                         chan_in=_rgb_channels(img.shape), n_channels=3)
    rng = _global_range(work, rng, tile)
    out = _tiled(lambda x, _: nl_bayes_core(x, sigma=s, data_range=rng, use_area1=use_area1, use_area2=use_area2,
                                            return_basic=return_basic, color_mode=color_mode, engine=engine,
                                            step=step, chunk=chunk, n_threads=n_threads, **params),
                 work, tile, margin)
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
    tile: int | None = None,
    margin: int = 32,
) -> np.ndarray:
    """
    Débruite et démosaïque une mosaïque de Bayer (NL-Bayes CFA).

    Estimateur des estimateurs linéaires par morceaux (Yu, Sapiro & Mallat,
    2012) appliqué aux groupes non locaux de NL-Bayes : pour un patch couleur
    x observé par y = U·x + n, x̂ = μ + C·Uᵀ(U·C·Uᵀ + βσ²I)⁻¹(y − U·μ), μ et C
    étant estimées sur une image pilote. Voir docs/technique/reference.md.

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
    tile, margin : int, optional
        Comme :func:`denoise_gray`, valeurs **paires** (phase du motif
        conservée). Chaque tuile fait ses propres passes : la marge doit
        couvrir la fenêtre de recherche (32 par défaut pour k = 3).

    Returns
    -------
    np.ndarray
        Image RGB (H, W, 3), ``float64``, unités de la mosaïque.
    """
    mosaic = check_image(image)
    if pattern.upper() not in CFA_PATTERNS:
        raise ValueError(f"Motif inconnu '{pattern}' (attendu : {CFA_PATTERNS}).")
    work, s, rng, back, fwd = _noise_setup(mosaic, sigma, noise, offset, sigma_scale, data_range, inverse,
                                           _estimate_mosaic_sigma, chan_in=channel_map(*mosaic.shape, pattern),
                                           n_channels=3)
    if not isinstance(pilot, str):
        pilot = np.asarray(pilot, dtype=np.float64)
        if pilot.shape != mosaic.shape + (3,):
            raise ValueError(f"Pilote (H, W, 3) attendu, de forme {mosaic.shape + (3,)}, reçu {pilot.shape}.")
        if fwd is not None:
            pilot = fwd(pilot)
    rng = _global_range(work, rng, tile)
    threads = n_threads if n_threads is not None else min(os.cpu_count() or 1, 8)
    out = _tiled(lambda x, pil: nl_bayes_cfa_core(
        x, sigma=s, pattern=pattern, data_range=rng, k=k, window_factor=window_factor, N=N, beta=beta,
        tau0=tau0, n_iter=n_iter, pilot=pil, pilot_kwargs=pilot_kwargs, use_area=use_area, area_z=area_z,
        area_gamma=area_gamma, area_mean=area_mean, step=step, chunk=chunk, n_threads=threads),
        work, tile, margin, extra=pilot, even=True)
    return back(out)

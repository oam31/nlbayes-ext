#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilitaires internes : validation des images, estimation de l'écart-type
d'un bruit blanc, passage à l'échelle [0, 1].

Convention : ``sigma`` est l'écart-type d'un bruit blanc gaussien additif,
dans les unités de l'image d'entrée.
"""
from __future__ import annotations

import numpy as np

__all__ = ["check_image", "estimate_noise_sigma", "resolve_sigma", "to_unit_range", "from_unit_range"]


def check_image(image: np.ndarray) -> np.ndarray:
    """
    Valide une image 2D et la renvoie en ``float64``.

    Raises
    ------
    ValueError
        Si l'image n'est pas 2D ou contient des valeurs non finies.
    """
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Image 2D attendue, reçu ndim={arr.ndim}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("L'image contient des valeurs non finies (NaN/inf).")
    return arr


def estimate_noise_sigma(image: np.ndarray) -> float:
    """
    Estime l'écart-type d'un bruit blanc gaussien additif.

    Estimateur robuste de Donoho & Johnstone (1994) : médiane des valeurs
    absolues des coefficients de Haar diagonaux du premier niveau, divisée
    par 0,6745.

    L'estimateur ne voit que les hautes fréquences du bruit : sur une image
    démosaïquée, où elles sont atténuées, il sous-estime σ. Fournir alors le
    modèle de bruit du capteur (``noise=(a, b)``).
    """
    x = check_image(image)
    h, w = (x.shape[0] // 2) * 2, (x.shape[1] // 2) * 2
    if min(h, w) < 2:
        raise ValueError(f"Image trop petite pour estimer σ : {x.shape}.")
    x = x[:h, :w]
    d = (x[0::2, 0::2] - x[0::2, 1::2] - x[1::2, 0::2] + x[1::2, 1::2]) / 2.0
    return float(np.median(np.abs(d)) / 0.6745)


def resolve_sigma(image: np.ndarray, sigma: float | None) -> float:
    """Renvoie ``sigma`` s'il est fourni, sinon son estimation sur ``image``."""
    if sigma is None:
        return estimate_noise_sigma(image)
    if sigma < 0:
        raise ValueError(f"sigma doit être positif ou nul, reçu {sigma}.")
    return float(sigma)


def to_unit_range(
    image: np.ndarray, data_range: tuple[float, float] | None = None
) -> tuple[np.ndarray, float, float]:
    """
    Ramène linéairement une image vers [0, 1].

    Renvoie l'image normalisée, le décalage ``lo`` et l'échelle ``scale``
    tels que ``image = normalisée * scale + lo``. ``data_range`` : bornes
    ``(min, max)`` de la dynamique ; par défaut, le min/max de l'image.
    """
    lo, hi = data_range if data_range is not None else (float(image.min()), float(image.max()))
    scale = hi - lo
    if scale <= 0:
        scale = 1.0
    return (image - lo) / scale, lo, scale


def from_unit_range(image: np.ndarray, lo: float, scale: float) -> np.ndarray:
    """Inverse de :func:`to_unit_range`."""
    return image * scale + lo

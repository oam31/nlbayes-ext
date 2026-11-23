"""
nlbayes-ext — prolongement de NL-Bayes aux images de capteur : NL-Bayes en
niveaux de gris et couleur RGB, et extension aux mosaïques de Bayer brutes
(débruitage et démosaïçage conjoints depuis les mesures du capteur), avec
modèle de bruit poissonnien-gaussien.

Implémentation écrite à partir des publications (Buades, Lebrun & Morel,
2013 ; Yu, Sapiro & Mallat, 2012), sans code tiers. Portage libre qui
n'implique pas les auteurs des articles. Licence MIT.

>>> from nlbayes_ext import denoise_gray, denoise_rgb, denoise_cfa
>>> rgb = denoise_cfa(mosaique, noise=(a, b), pattern="RGGB")   # doctest: +SKIP
"""
from ._cfa import CFA_PATTERNS, bilinear_demosaic, cfa_channels, channel_map
from ._gray import article_parameters, resolve_parameters
from ._vectorized import COLOR_MODES, YUV_MATRIX
from .api import denoise_cfa, denoise_gray, denoise_rgb

__version__ = "0.2.0"

__all__ = [
    "denoise_gray",
    "denoise_rgb",
    "denoise_cfa",
    "article_parameters",
    "resolve_parameters",
    "bilinear_demosaic",
    "cfa_channels",
    "channel_map",
    "CFA_PATTERNS",
    "COLOR_MODES",
    "YUV_MATRIX",
    "__version__",
]

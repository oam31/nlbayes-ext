"""Constantes et assistants des tests."""
import os

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr  # noqa: F401  (réexporté)

#: Écart-type du bruit des images de test (dynamique [0, 1]).
SIGMA = 0.1
#: Gain de PSNR minimal exigé (test de non-régression grossier).
MIN_GAIN_DB = 3.0
#: Graine des tirages, modifiable par NLBAYES_TEST_SEED.
SEED = int(os.environ.get("NLBAYES_TEST_SEED", "0"))


def astronaut_gray(size: int) -> np.ndarray:
    from skimage import data
    from skimage.color import rgb2gray
    from skimage.transform import resize

    return resize(rgb2gray(data.astronaut()), (size, size), anti_aliasing=True).astype(np.float64)


def astronaut_rgb(size: int) -> np.ndarray:
    from skimage import data
    from skimage.transform import resize

    return resize(data.astronaut().astype(np.float64) / 255.0, (size, size, 3), anti_aliasing=True)

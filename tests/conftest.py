"""
Fixtures communes. Image de test « astronaut » (NASA, domaine public,
fournie par scikit-image, extra ``test``), réduite ; bruit blanc gaussien
de σ connu. Graine modifiable par ``NLBAYES_TEST_SEED``.
"""
import numpy as np
import pytest

from _helpers import SEED, SIGMA, astronaut_gray, astronaut_rgb


@pytest.fixture(scope="session")
def clean() -> np.ndarray:
    return astronaut_gray(128)


@pytest.fixture(scope="session")
def noisy(clean) -> np.ndarray:
    return clean + np.random.default_rng(SEED).normal(0.0, SIGMA, clean.shape)


@pytest.fixture(scope="session")
def rgb_pair():
    clean_rgb = astronaut_rgb(96)
    return clean_rgb, clean_rgb + np.random.default_rng(SEED).normal(0.0, SIGMA, clean_rgb.shape)

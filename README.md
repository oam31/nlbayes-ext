# nlbayes-ext

> **En développement** (version 0.x) : l'API et les valeurs par défaut
> peuvent encore changer, en particulier pour `denoise_cfa`
> ([feuille de route](docs/technique/feuille-de-route-cfa.md)).

Débruitage d'images **de capteur**, dans le prolongement de NL-Bayes
(Buades, Lebrun & Morel, 2013), en deux parties :

- **NL-Bayes** pour les images en niveaux de gris et couleur RGB
  (`denoise_gray`, `denoise_rgb`) : implémentation de la méthode publiée,
  au niveau de l'implémentation de référence d'IPOL, complétée par un
  modèle de bruit de capteur poissonnien-gaussien ;
- **l'extension aux mosaïques de Bayer** (`denoise_cfa`), qui suit une voie
  en partie différente : au lieu de débruiter une image déjà démosaïquée,
  dont le démosaïçage a corrélé le bruit, elle part des **mesures brutes du
  capteur**, où le bruit est blanc, et estime directement l'image couleur
  complète. Elle **débruite et démosaïque en une seule opération**.

Python, numpy et scipy seulement, licence MIT.

## L'extension CFA en bref

| | NL-Bayes (`denoise_rgb`) | Extension CFA (`denoise_cfa`) |
|---|---|---|
| Entrée | image RGB, déjà démosaïquée | mosaïque de Bayer brute, une mesure par pixel |
| Bruit vu par l'estimateur | corrélé par le démosaïçage | blanc (après stabilisation) |
| Modèle d'un patch | x̃ = x + n | y = U·x + n, U sélectionne les mesures du motif |
| Estimateur | filtre de Wiener du groupe (deux étapes) | espérance a posteriori d'un problème inverse linéaire, x̂ = μ + C·Uᵀ(U·C·Uᵀ + βσ²I)⁻¹(y − U·μ) |
| Ce qui est repris de NL-Bayes | — | groupes de patchs semblables, seuil τ, moyenne et covariance non locales |
| Démosaïçage | externe, avant | fait par l'estimateur ; le démosaïçage initial ne sert que de pilote |
| Schéma | deux étapes | passes successives, la sortie devenant le pilote de la suivante |

L'estimateur est celui des estimateurs linéaires par morceaux de Yu,
Sapiro et Mallat (2012) ; le prolongement consiste à lui donner, pour
chaque patch, la gaussienne de son groupe NL-Bayes, et à itérer. Sur
mosaïques simulées, l'extension dépasse nettement la chaîne
« démosaïçage puis NL-Bayes », en particulier sur le bruit de couleur
(`docs/technique/cfa.md` §4.5) ; elle est en cours de validation
([feuille de route](docs/technique/feuille-de-route-cfa.md)).

> **Portage libre.** nlbayes-ext est écrit à partir des seules publications
> citées ci-dessous, sans reprendre de code existant. Il n'implique pas les
> auteurs de ces articles et n'est pas approuvé par eux. L'implémentation de
> référence de NL-Bayes est celle d'IPOL (Lebrun, 2013, GPL-3.0+), à
> laquelle la partie NL-Bayes est comparée (`docs/technique/implementation.md`
> §6).

## Installation

```bash
pip install "git+https://github.com/oam31/nlbayes-ext.git@v0.2.0"
pip install -e ".[test]"                    # développement, depuis le dépôt local
```

## Usage

```python
from nlbayes_ext import denoise_gray, denoise_rgb, denoise_cfa

# bruit blanc gaussien d'écart-type connu (unités de l'image)
propre = denoise_gray(image, sigma=12.0)

# bruit du capteur var(z) = a + b·(z − offset) : stabilisation de variance interne
propre = denoise_gray(brute, noise=(a, b), offset=64.0)

# image couleur démosaïquée : σ du domaine stabilisé ≈ √κ_Y du démosaïqueur
rgb = denoise_rgb(demosaiquee, noise=(a, b), sigma_scale=1.1)

# mosaïque de Bayer brute → image RGB débruitée et démosaïquée
rgb = denoise_cfa(mosaique, noise=(a, b), pattern="RGGB")

# options : modèle de bruit par canal (R, G, B), traitement par tuiles
rgb = denoise_rgb(image, noise=((aR, bR), (aG, bG), (aB, bB)), tile=512, margin=32)
```

| Fonction | Entrée | Sortie | Méthode |
|---|---|---|---|
| `denoise_gray` | (H, W) | (H, W) | NL-Bayes, deux étapes (article IPOL) |
| `denoise_rgb` | (H, W, 3) | (H, W, 3) | NL-Bayes couleur : étape 1 en Y₀U₀V₀, étape 2 en RGB à canaux filtrés ensemble |
| `denoise_cfa` | mosaïque (H, W) | (H, W, 3) | estimateur gaussien d'un problème inverse (Yu, Sapiro & Mallat) sur les groupes NL-Bayes |

Le bruit se donne par `sigma` (bruit blanc gaussien) ou par
`noise=(a, b)` (a plancher en DN², b pente en DN ; un bruit stationnaire
s'écrit `sigma=√a`). Sans l'un ni l'autre, σ est estimé.

## Documentation

- [documents techniques](docs/technique/index.md) : méthodes,
  paramètres, modèle de bruit, choix d'implémentation, comparaison avec IPOL ;
- [feuille de route](docs/technique/feuille-de-route-cfa.md) : ce qui reste à
  faire avant une version 1.0 de `denoise_cfa` ;
- [CHANGELOG.md](CHANGELOG.md) : ce qui change d'une version à l'autre.

Ces documents sont publiés avec la référence de l'API par Sphinx et MyST
(`docs/`, `make html` dans ce répertoire ; extra `docs`).

## Tests

```bash
pip install -e ".[test]"
pytest
NLBAYES_IPOL_EXE=/chemin/NL_Bayes pytest tests/test_ipol.py   # comparaison avec IPOL, si compilé
```

## Références

- A. Buades, M. Lebrun, J.-M. Morel, « A Nonlocal Bayesian Image Denoising
  Algorithm », *SIAM J. Imaging Sciences* 6(3), 2013, p. 1665-1688.
- M. Lebrun, A. Buades, J.-M. Morel, « Implementation of the "Non-Local
  Bayes" (NL-Bayes) Image Denoising Algorithm », *Image Processing On Line*
  3, 2013, p. 1-42, https://doi.org/10.5201/ipol.2013.16
- G. Yu, G. Sapiro, S. Mallat, « Solving Inverse Problems With Piecewise
  Linear Estimators: From Gaussian Mixture Models to Structured Sparsity »,
  *IEEE Trans. Image Processing* 21(5), 2012, p. 2481-2499,
  https://doi.org/10.1109/TIP.2011.2176743
- M. Mäkitalo, A. Foi, « Optimal Inversion of the Generalized Anscombe
  Transformation for Poisson-Gaussian Noise », *IEEE Trans. Image
  Processing* 22(1), 2013, p. 91-103.

## Licence

MIT, voir [`LICENSE`](LICENSE). L'exécutable IPOL utilisé pour la
comparaison (GPL-3.0+) n'est ni fourni ni lié : il est appelé comme
programme séparé, s'il est installé par l'utilisateur.

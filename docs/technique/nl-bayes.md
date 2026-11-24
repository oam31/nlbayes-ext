# Modèle de bruit et NL-Bayes gris et couleur

## 1. Vue d'ensemble

| Fonction | Entrée → sortie | Méthode | Source |
|---|---|---|---|
| `denoise_gray` | (H, W) → (H, W) | NL-Bayes, deux étapes | Lebrun, Buades & Morel, IPOL 2013 |
| `denoise_rgb` | (H, W, 3) → (H, W, 3) | NL-Bayes couleur | idem, §3.3, §3.4, tableau 1 |
| `denoise_cfa` | mosaïque (H, W) → (H, W, 3) | débruitage et démosaïçage conjoints | Yu, Sapiro & Mallat, IEEE TIP 2012, sur les groupes NL-Bayes |

Les deux premières fonctions implémentent NL-Bayes tel que publié.
`denoise_cfa` en est un prolongement par une voie en partie différente :
il part de la mosaïque brute (bruit blanc) au lieu d'une image démosaïquée
(bruit corrélé), remplace le filtre de Wiener par l'estimateur d'un
problème inverse linéaire, et itère ([§4](cfa.md#4-nl-bayes-cfa-denoise_cfa)). Implémentation écrite à partir de
ces seules publications : aucun code tiers consulté ni repris. C'est un
portage libre, qui n'implique pas les auteurs des articles.

Organisation du paquet `nlbayes_ext` :

| Module | Rôle |
|---|---|
| `api.py` | les trois fonctions publiques, modèle de bruit |
| `_gray.py` | paramètres de l'article, moteur « boucle » (fidèle à l'article) |
| `_vectorized.py` | moteur vectorisé (défaut), niveaux de gris et couleur |
| `_cfa.py` | NL-Bayes CFA, démosaïçage bilinéaire du pilote, motifs de Bayer |
| `vst.py` | stabilisation de variance (Anscombe généralisée) et inverses |
| `_utils.py` | validation, estimation de σ, mise à l'échelle |

## 2. Modèle de bruit et stabilisation

Deux façons, exclusives, de décrire le bruit :

- `sigma` : bruit blanc gaussien d'écart-type σ (unités de l'image). Estimé
  si absent : médiane des coefficients de Haar diagonaux / 0,6745 (Donoho
  & Johnstone, 1994) ; sur une mosaïque, moyenne sur les quatre sous-plans.
- `noise=(a, b)` : bruit poissonnien-gaussien du capteur,

  ```
  var(z) = a + b·(z − offset)      a : plancher (DN²)      b : pente (DN) = 1/K
  ```

  Un bruit stationnaire s'écrit a = σ², b = 0 : donner alors `sigma = √a`.

**Modèle par canal** (`denoise_rgb`, `denoise_cfa`) : `noise` accepte
aussi trois couples `((a_R, b_R), (a_G, b_G), (a_B, b_B))`, et `offset`
une ou trois valeurs ; chaque canal (ou chaque photosite de la mosaïque,
selon le motif) est stabilisé et inversé avec son propre modèle. Utile si
les gains diffèrent entre canaux, par exemple après une balance des blancs
appliquée avant le débruitage. Mesuré (astronaut 128 × 128, crête 300,
gains 0,5 / 1 / 2 et lectures 2 / 3 / 5 par canal) :

| | modèle par canal | modèle commun (vert) | modèle commun (moyenne) |
|---|---|---|---|
| `denoise_rgb` | **34,33 dB** | 33,46 dB | 34,14 dB |
| `denoise_cfa` (3 tirages) | 29,64 à 29,70 dB | 29,65 à 29,77 dB | 29,61 à 29,71 dB |

En RGB, le modèle par canal gagne 0,9 dB sur un modèle commun mal choisi
et 0,2 dB sur la moyenne. En CFA, il n'apporte rien : la stabilisation,
non linéaire et différente d'un canal à l'autre, déforme les relations
linéaires entre canaux que l'estimateur apprend dans la covariance du
groupe, ce qui compense le gain d'une variance exacte (explication
probable, non démontrée). L'option reste disponible.

Avec `noise`, la fonction applique la transformée d'Anscombe généralisée
(Murtagh, Starck & Bijaoui, 1995), avec gain K = 1/b et bruit de lecture
√a : le bruit devient de variance ≈ 1. Le débruiteur travaille avec
σ = `sigma_scale` (1 par défaut), puis l'image revient dans ses unités par
l'**inverse exacte non biaisée** (Mäkitalo & Foi, 2013 ; option `inverse` :
`"exact_unbiased"`, `"asymptotic"`, `"algebraic"`).

- **Image brute** (gris ou mosaïque) : le bruit stabilisé est blanc
  d'écart-type 1, `sigma_scale = 1`. Pour `denoise_cfa`, c'est l'usage
  recommandé : le bruit d'une mosaïque est indépendant d'un photosite à
  l'autre.
- **Image démosaïquée** (`denoise_rgb`) : le démosaïçage multiplie la
  variance du bruit en luminance par κ_Y et le corrèle ; donner
  `sigma_scale ≈ √κ_Y` (≈ 1,1 pour Hamilton-Adams, κ_Y ≈ 1,35).

## 3. NL-Bayes, niveaux de gris et couleur

### 3.1 Les deux étapes

Pour chaque patch de référence P̃ (k × k), un groupe de patchs semblables
dans une fenêtre n × n (n = 7k) ; moyenne P̄ et covariance C du groupe.

- **Étape 1** (formules 10 à 12 de l'article) : N₁ plus proches voisins
  dans l'image bruitée. Si la variance du groupe est ≤ (γσ)², le groupe
  est jugé homogène et remplacé par sa moyenne ; sinon chaque patch Q̃ est
  estimé par Q̂ = P̄ + (C − β₁σ²I)·C⁻¹·(Q̃ − P̄). Agrégation non pondérée →
  estimée de base.
- **Étape 2** (formules 13 à 15) : groupes formés sur l'estimée de base,
  avec le seuil τ = max(τ₀, d_N₂) ; covariance C_b des patchs de l'estimée
  de base, P̄ moyenne des patchs bruités ; Q̂ = P̄ + C_b·(C_b + β₂σ²I)⁻¹·(Q̃ − P̄).

### 3.2 Couleur (`denoise_rgb`)

`color_mode` :

- `"yuv"` (défaut, article) : étape 1 dans l'espace orthonormé Y₀U₀V₀,
  groupes formés sur Y₀, chaque canal filtré séparément avec ces groupes ;
  étape 2 en RGB, distance moyenne sur les trois canaux de l'estimée de
  base (formule 13), **canaux filtrés ensemble** : patchs de 3k² valeurs et
  covariance commune (tableau 1 de l'article : deux groupes 3D en tout à
  l'étape 2, contre deux par canal pour BM3D) ;
- `"rgb"` : les deux étapes en RGB, groupes communs ; étape 2 à canaux
  filtrés ensemble ;
- `"channel"` : chaque canal comme une image en niveaux de gris.

### 3.3 Paramètres

Valeurs de l'article (tableau §4.8, σ sur l'échelle [0, 255] ;
`article_parameters`) ; patchs de 5 en niveaux de gris (§6.1.2) :

| σ (sur [0, 255]) | k₁, k₂ couleur | k₁, k₂ gris | N (k = 3, 5, 7) | n | γ | β₁ | β₂ | τ₀ |
|---|---|---|---|---|---|---|---|---|
| < 20 | 3, 3 | 5, 5 | 30, 60, 90 | 7k | 1,05 | 1 | 1,2 | 4 |
| 20 à 50 | 5, 3 | 5, 5 | | | | | 1,2 | |
| 50 à 70 | 7, 5 | 7, 5 | | | | | 1,0 | |
| ≥ 70 | 7, 7 | 7, 7 | | | | | 1,0 | |

Chaque valeur peut être remplacée (`k1`, `k2`, `n1`, `n2`, `N1`, `N2`,
`gamma`, `beta1`, `beta2`, `tau0`, `window_factor`). Imbrications : changer
k sans N applique N = 30 / 60 / 90 ; la fenêtre suit n = α·k ; N est porté
à k² + 1 au moins (covariance inversible à l'étape 1).

Redondance : σ n'intervient que par β₁σ², β₂σ² et γσ. Multiplier σ par s
équivaut à multiplier β₁, β₂ par s² et γ par s : un réglage est de trop
parmi `sigma_scale`, β₁, β₂, γ.

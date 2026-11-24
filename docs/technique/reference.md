# nlbayes-ext — référence technique

**Plan.** §1 vue d'ensemble ; §2 modèle de bruit et stabilisation ; §3
NL-Bayes (niveaux de gris et couleur) ; §4 NL-Bayes CFA ; §5 choix
d'implémentation ; §6 comparaison avec l'implémentation IPOL ; §7 tests ;
§8 variantes écartées. Ce qui change d'une version à l'autre est dans
`CHANGELOG.md` ; ce qui reste à faire avant une version 1.0 de
`denoise_cfa`, dans la [feuille de route](feuille-de-route-cfa.md).

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
problème inverse linéaire, et itère (§4). Implémentation écrite à partir de
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

## 4. NL-Bayes CFA (`denoise_cfa`)

### 4.1 Modèle et estimateur

Un patch couleur k × k est un vecteur x de dimension D = 3k² ; la mosaïque
en mesure d = k² valeurs :

```
y = U·x + n        U : sélection (d × D) des valeurs mesurées (motif et position du patch)
                   n ~ N(0, σ²·I)
x ~ N(μ, C)        a priori du groupe de patchs semblables

x̂ = μ + C·Uᵀ·(U·C·Uᵀ + β·σ²·I)⁻¹·(y − U·μ)
```

C'est l'espérance a posteriori des estimateurs linéaires par morceaux de
Yu, Sapiro et Mallat (2012), chaque patch étant estimé avec la gaussienne
de son groupe NL-Bayes. Toutes les composantes sont estimées, mesurées
comme manquantes : la méthode débruite et démosaïque en une opération.
Seule U·C·Uᵀ + βσ²I (d × d) est inversée, définie positive même si C est
singulière.

### 4.2 Algorithme

1. **Pilote** : démosaïçage bilinéaire puis `denoise_rgb` (défaut), ou
   bilinéaire seul (`pilot="none"`), ou image fournie. Il sert à former les
   groupes et à estimer μ et C ; aucune de ses valeurs ne passe dans le
   résultat.
2. **Groupes** : pour chaque référence (grille de pas `step`), patchs à
   distance ≤ max(τ₀, d_N) dans le pilote ; μ et C sur les patchs pilotes.
3. **Phases** : U dépend de la position du patch dans le motif 2 × 2. Une
   résolution par phase présente dans le groupe.
4. **Agrégation** non pondérée des estimées.
5. **Passes** : la sortie devient le pilote ; `n_iter` passes (8).

### 4.3 Paramètres

| Paramètre | Défaut | Rôle |
|---|---|---|
| `pattern` | `"RGGB"` | motif (`"RGGB"`, `"BGGR"`, `"GRBG"`, `"GBRG"`) |
| `k` | 3 | côté des patchs (D = 27) |
| `window_factor` | 7 | fenêtre n = 7k |
| `N` | 60 | taille minimale des groupes ; N = 30 estime mal une covariance de dimension 27 |
| `beta` | 1 | facteur de σ² ; 0,7 à 1,4 changent peu le résultat |
| `tau0` | 4 | plancher de similarité, niveaux² sur [0, 255] |
| `n_iter` | 8 | passes ; plus de passes favorisent les grandes zones uniformes |
| `pilot` | `"bilinear"` | pilote initial |
| `use_area`, `area_z`, `area_gamma`, `area_mean` | non, 2, —, `"mosaic"` | critère des zones homogènes (§4.4) |

### 4.4 Critère des zones homogènes

Variance des **mesures de la mosaïque** du groupe, canal par canal autour
de la moyenne de chaque canal ; groupe plat si
s²/σ² ≤ 1 + z·√(2/(n − 3)), n le nombre de mesures (ou s² ≤ (γσ)² avec
`area_gamma`). Un groupe plat prend la moyenne de chaque canal (mesures ou
pilote). Désactivé par défaut : utile à faible flux (luminance des zones
sombres), mais coûteux sur une zone parfaitement uniforme, où la moyenne
du groupe repose sur trop peu de mesures indépendantes.

### 4.5 Performances

Mesurées en dehors de ce dépôt (chaîne capteur simulée : orthophotos
texturées et lisses, scènes synthétiques à grandes zones uniformes ;
mosaïque RGGB, bruit de Poisson + lecture 3 e⁻, 300 à 7 500 e⁻ en crête),
contre la vraie image RGB, réglages par défaut. Référence : Hamilton-Adams
puis `denoise_rgb` (étape 2 commune, σ = C = √κ_Y + 0,2, β₂ = 1,2).

| Jeu | PSNR, 300 / 1 500 / 7 500 e⁻ : référence → `denoise_cfa` | Gain apparié moyen (minimum) |
|---|---|---|
| orthophotos texturées | 30,30 / 31,17 / 31,25 → 31,28 / 34,18 / 35,10 | +1,0 / +3,0 / +3,9 (+0,10) |
| orthophotos lisses | 31,41 / 32,02 / 32,28 → 32,99 / 35,96 / 37,49 | +1,6 / +3,9 / +5,2 (+0,81) |
| scènes synthétiques | 31,69 / 31,32 / 31,30 → 36,52 / 37,36 / 37,28 | +4,8 / +6,0 / +6,0 (+4,26) |

- Erreur de couleur des zones sombres : −11 à −60 %. Erreur de luminance
  des zones sombres : −2 à −57 %, sauf à 300 e⁻ sur les orthophotos
  (+5 à +7 %), que le critère des zones homogènes ramène au niveau de la
  référence.
- Sans bruit, `denoise_cfa` démosaïque mieux qu'Hamilton-Adams (+4,3 à
  +7 dB sur ces scènes).
- Coût : 7 à 12 fois celui de la référence (8 passes).

### 4.6 Traitement par tuiles

Option `tile` (côté, None par défaut : un seul bloc) et `margin` (32
pixels par défaut) des trois fonctions. L'image est découpée en tuiles ;
chacune est élargie de `margin` pixels pris dans l'image voisine (rabattus
aux bords de l'image), traitée, et seul son intérieur est gardé. σ (estimé
ou stabilisé) et la dynamique ramenée à [0, 255] sont calculés **une fois
pour toute l'image** : chaque tuile a exactement les mêmes réglages. En
CFA, `tile` et `margin` sont pairs, pour que chaque tuile commence sur la
même phase du motif de Bayer ; chaque tuile fait ses propres passes,
pilote compris.

La marge doit couvrir la fenêtre de recherche : n/2 + k, soit 22 pour
k = 3 (n = 21) et 30 pour k = 5 (n = 35) ; 32 convient aux réglages par
défaut, 56 est prudent pour k = 7 (n = 49).

Mesuré (niveaux de gris, σ = 0,1, 8 fils) :

| Image | Tuiles | PSNR | Temps | Mémoire de pointe (numpy) |
|---|---|---|---|---|
| 512 × 512 | aucune | 31,024 dB | 3,8 s | 987 Mo |
| | 256 | 31,021 dB | 4,8 s | 706 Mo |
| 1 024 × 1 024 | aucune | 35,204 dB | 17,3 s | 2 578 Mo |
| | 256 | 35,196 dB | 24,4 s | 1 622 Mo |
| 2 048 × 2 048 | 256 | 35,126 dB | 102 s | 1 734 Mo |

Sans tuiles, la mémoire croît avec la surface de l'image (environ 2,6 Go
à 1 024², une dizaine de gigaoctets attendus à 2 048²). Avec des tuiles de
256, elle plafonne vers 1,7 Go, fixée par les tampons de calcul de chaque
fil, quelle que soit la taille de l'image. Le prix : 25 à 40 % de temps en
plus (les marges sont calculées deux fois), un écart de PSNR inférieur à
0,01 dB. En CFA (256², tuiles de 128) : PSNR identique, mémoire −22 %,
temps +50 %.

## 5. Choix d'implémentation

Points que l'article ne fixe pas, ou dont l'implémentation s'écarte :

| Point | Choix |
|---|---|
| Patchs de référence | moteur vectorisé (défaut) : grille de pas `step` = 3, calcul par lots, au lieu du « paste trick » séquentiel ; moteur `engine="loop"` fidèle à l'article (gris et `"channel"`) |
| Bords | fenêtre de recherche rabattue à l'intérieur de l'image |
| Covariance de l'étape 2 | centrée sur la moyenne des patchs de l'estimée de base |
| Matrice singulière (étape 1) | patchs laissés inchangés (remarque §3.3 de l'article) |
| Échelle | image ramenée sur [0, 255] (`data_range` ou min/max) pour appliquer les paramètres de l'article ; calcul en flottants, sans quantification |
| Étape 2 couleur | canaux filtrés ensemble (tableau 1) |

## 6. Comparaison avec l'implémentation IPOL

**Cadre.** L'implémentation de référence d'IPOL (archive
`nl-bayes_20130617.tar.gz`, Lebrun 2013, **GPL-3.0-or-later**) est appelée
comme **programme séparé** (`examples/ipol_reference.py`), sur des fichiers
PNG. La GPL n'interdit pas d'exécuter un programme ni de comparer ses
sorties ; nlbayes-ext ne contient, ne lie ni ne distribue aucun code ou
binaire IPOL. L'exécutable est compilé et fourni par l'utilisateur
(`NLBAYES_IPOL_EXE`). Seuls le README de l'archive et sa ligne de commande
documentée ont été lus.

**Protocole** (`examples/ipol_comparison.py`) : images de scikit-image
réduites à 256 × 256 ; bruit blanc gaussien σ ∈ {5, 10, 20, 40} sur
[0, 255], 3 tirages ; image bruitée **quantifiée sur 8 bits** (seul format
lu par IPOL) et donnée telle quelle aux deux implémentations ; réglages par
défaut des deux côtés ; PSNR (dB) contre l'image propre.

| Image | σ | Bruitée | Étape 1 IPOL | Étape 1 ext | IPOL | ext | ext − IPOL | Écart des sorties |
|---|---|---|---|---|---|---|---|---|
| astronaut (gris) | 5 | 34,47 | 37,35 | 38,80 | 39,07 | 39,05 | −0,02 | 47,93 |
| | 10 | 28,50 | 33,31 | 34,10 | 34,63 | 34,50 | −0,13 | 43,65 |
| | 20 | 22,61 | 28,68 | 29,35 | 30,16 | 29,98 | −0,18 | 39,48 |
| | 40 | 16,90 | 23,76 | 24,39 | 25,52 | 25,27 | −0,26 | 35,36 |
| camera (gris) | 5 | 34,19 | 36,90 | 38,89 | 38,89 | 38,93 | +0,05 | 47,41 |
| | 10 | 28,23 | 33,98 | 34,66 | 34,90 | 34,79 | −0,11 | 43,81 |
| | 20 | 22,41 | 30,63 | 31,09 | 31,57 | 31,47 | −0,10 | 41,34 |
| | 40 | 16,87 | 26,81 | 27,46 | 27,99 | 27,89 | −0,10 | 38,55 |
| astronaut (RGB) | 5 | 34,49 | 39,19 | 39,90 | 40,42 | 40,53 | +0,11 | 50,57 |
| | 10 | 28,55 | 34,42 | 35,38 | 35,92 | 36,07 | +0,15 | 45,25 |
| | 20 | 22,71 | 30,07 | 30,84 | 31,53 | 31,71 | +0,18 | 41,22 |
| | 40 | 17,05 | 24,69 | 25,66 | 26,35 | 26,65 | +0,30 | 35,97 |
| coffee (RGB) | 5 | 34,28 | 38,81 | 39,39 | 39,87 | 40,05 | +0,18 | 50,38 |
| | 10 | 28,38 | 34,32 | 35,19 | 35,74 | 35,95 | +0,21 | 45,03 |
| | 20 | 22,61 | 30,63 | 31,20 | 31,91 | 32,11 | +0,20 | 41,36 |
| | 40 | 17,01 | 25,88 | 26,50 | 27,17 | 27,49 | +0,32 | 36,75 |

« Écart des sorties » : PSNR d'une sortie par rapport à l'autre.

Constats :

1. **Niveaux de gris** : nlbayes-ext est au niveau d'IPOL (−0,26 à
   +0,05 dB). Son estimée de base est meilleure (+0,5 à +2 dB) ; l'étape 2
   d'IPOL rattrape cet écart. Différences connues : grille de pas 3 au lieu
   du « paste trick », patchs de 5 en gris (§6.1.2 de l'article),
   quantification 8 bits côté IPOL.
2. **Couleur** : nlbayes-ext dépasse IPOL de 0,11 à 0,32 dB, une fois
   l'étape 2 à canaux filtrés ensemble (§8).
3. **Sorties proches** : 35 à 51 dB de PSNR entre les deux sorties.
4. **Temps** (256 × 256) : IPOL (C++, OpenMP) 0,07 à 0,18 s ; nlbayes-ext
   (numpy) 0,4 à 0,8 s.

## 7. Tests

`pytest` (extra `test`) : 85 tests, dont 2 sautés sans l'exécutable IPOL.

| Fichier | Contenu |
|---|---|
| `test_gray_rgb.py` | gain, deux étapes, σ estimé, invariance d'échelle, paramètres et imbrications, moteurs, modes couleur, `noise=(a, b)`, contrôles |
| `test_cfa.py` | motifs, gain sur le bilinéaire, motif faux, sans bruit, `noise=(a, b)`, pilote, passes, critère des zones homogènes, contrôles |
| `test_vst.py` | transformée, inverse exacte non biaisée (Monte-Carlo, 5 incertitudes-types), stabilisation de la variance, simulateur, `noise_to_gat` |
| `test_options.py` | `noise` par canal (identique au modèle commun si les canaux sont égaux, gain en RGB, CFA sans dégradation, contrôles) ; tuiles (gris, RGB, CFA avec phase du motif, modèle de bruit) proches du traitement d'un bloc |
| `test_ipol.py` | protocole d'appel (faux exécutable) ; si `NLBAYES_IPOL_EXE` : gris à 0,4 dB d'IPOL au plus, couleur au moins au niveau d'IPOL (à 0,1 dB) |

Graine modifiable par `NLBAYES_TEST_SEED`.

## 8. Variantes écartées

- **Étape 2 couleur canal par canal**, avec les mêmes groupes pour les
  trois canaux : le tableau 1 de l'article indique deux groupes 3D en tout
  à l'étape 2 (contre deux par canal pour BM3D), donc des patchs RGB de
  3k² valeurs et une covariance commune (§5). Canal par canal, l'écart à
  IPOL en couleur est de −0,15 à −1,16 dB (l'étape 2 dégrade parfois
  l'estimée de base) ; canaux ensemble, de +0,11 à +0,32 dB. Les niveaux
  de gris ne sont pas concernés.

# Implémentation, comparaison avec IPOL et tests

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

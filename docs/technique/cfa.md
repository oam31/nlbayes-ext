# NL-Bayes CFA

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

# Documents techniques

Ces documents décrivent l'état présent du code. Les chapitres gardent une
numérotation unique d'un document à l'autre : un renvoi « §n » désigne
toujours le même chapitre. Ce qui change d'une version à l'autre est dans
les [notes de version](../changelog.md).

| Document | Chapitres | Contenu |
|---|---|---|
| [Modèle de bruit et NL-Bayes gris et couleur](nl-bayes.md) | 1, 2, 3 | Vue d'ensemble, modèle de bruit et stabilisation, les deux étapes, couleur, paramètres |
| [NL-Bayes CFA](cfa.md) | 4 | `denoise_cfa` : modèle et estimateur, algorithme, paramètres, zones homogènes, performances, tuiles |
| [Implémentation, comparaison avec IPOL et tests](implementation.md) | 5, 6, 7, 8 | Choix d'implémentation, comparaison avec l'exécutable IPOL, tests, variantes écartées |
| [Feuille de route de `denoise_cfa`](feuille-de-route-cfa.md) | | Ce qui reste à faire avant une version 1.0 |

```{toctree}
:hidden:
:maxdepth: 2

nl-bayes
cfa
implementation
feuille-de-route-cfa
```

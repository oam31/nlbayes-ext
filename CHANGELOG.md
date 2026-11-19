# Change Log

## Version History

0.1.0 (2026-11-19)

:   -   Première version : `denoise_gray`, `denoise_rgb` (NL-Bayes, Lebrun,
        Buades & Morel 2013) et `denoise_cfa` (débruitage et démosaïçage
        conjoints d'une mosaïque de Bayer, estimateur de Yu, Sapiro &
        Mallat 2012 sur les groupes NL-Bayes).
    -   Modèle de bruit `noise=(a, b)` (var = a + b·(z − offset)) avec
        stabilisation de variance interne et inverse exacte non biaisée.
    -   Étape 2 couleur à canaux filtrés ensemble (tableau 1 de l'article) :
        au niveau de l'implémentation IPOL en niveaux de gris, au-dessus en
        couleur (`docs/REFERENCE_TECHNIQUE.md` §6).
    -   Comparaison avec l'exécutable IPOL (programme externe, non fourni) :
        `examples/ipol_reference.py`, `examples/ipol_comparison.py`,
        `tests/test_ipol.py`.

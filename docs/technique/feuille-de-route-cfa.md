# Feuille de route : `denoise_cfa` vers une version 1.0

`denoise_cfa` (NL-Bayes CFA) est fonctionnel et testé, mais il n'a été
évalué que sur des mosaïques simulées. Ce document liste ce qui reste à
faire avant de figer son API et ses valeurs par défaut, par ordre
d'importance. Les points 1 et 2 peuvent changer la méthode ; les autres
sont du travail d'ingénierie. Les réglages par défaut (N = 60, 8 passes)
sont mesurés contre la vraie image : une référence « démosaïçage sans
bruit » pénalise tout ce qui corrige aussi la couleur
([référence technique](reference.md) §4.5).

## 1. Validation sur données réelles

Tout est simulé (bruit poissonnien-gaussien idéal, une seule orthophoto et
des scènes synthétiques). À faire :

- une vraie mosaïque brute, avec son modèle (a, b) mesuré (courbe de
  transfert photonique) ;
- les jeux de référence du démosaïçage et du débruitage conjoints (Kodak,
  McMaster, jeux de vraies images brutes) ;
- vérifier les écarts au modèle : gains différents par canal, bruit de
  ligne, non-linéarités, pixels défectueux.

## 2. Critère des zones homogènes

Aujourd'hui optionnel (`use_area`). Il ramène la luminance des zones
sombres à faible flux au niveau de la chaîne démosaïçage + NL-Bayes, mais
coûte jusqu'à 10 dB sur une zone parfaitement uniforme : la moyenne d'un
groupe plat repose sur trop peu de mesures indépendantes (une soixantaine
de patchs 3 × 3 qui se chevauchent). Piste : calculer cette moyenne sur un
support plus large que le groupe (toutes les mesures de la fenêtre qui
passent le même test), puis l'activer par défaut si le test sur zone
uniforme ne perd plus rien.

## 3. Nombre de passes

Fixe (8). L'optimum dépend du contenu : 12 passes et plus pour les grandes
zones uniformes, 4 suffisent à faible flux sur des scènes texturées. Pistes :
critère d'arrêt (variation de l'estimée d'une passe à l'autre), ou règle
selon le flux et la texture.

## 4. Modèle de bruit intégré à l'estimateur

Aujourd'hui, `noise=(a, b)` passe par la stabilisation de variance. Piste :
σ² = a + b·μ par groupe (μ moyenne du groupe) directement dans
U·C·Uᵀ + βσ²I, sans transformée ni inverse. Évite le biais résiduel de la
transformée aux très faibles flux. À comparer à la stabilisation.

## 5. Vitesse

8 passes coûtent 6 à 11 fois la chaîne démosaïçage + NL-Bayes. Le
traitement par tuiles (`tile`, `margin`) borne la mémoire des grandes
images ([référence technique](reference.md) §4.6). Pistes : réutiliser les
groupes d'une passe à l'autre quand le pilote change peu, arrêt anticipé
(point 3), tuiles en parallèle, éventuellement un moteur GPU.

## 6. Modèle de bruit par canal

`noise` accepte un couple (a, b) par canal ; en CFA, sans gain mesuré
([référence technique](reference.md) §2). À revoir avec le point 4
(modèle de bruit dans l'estimateur, sans transformée non linéaire par
canal).

## 7. API à figer

Noms et valeurs par défaut des paramètres ; motifs acceptés (les quatre
motifs de Bayer ; Quad Bayer et X-Trans hors périmètre) ; comportement aux
bords et pour des tailles impaires (testé, à documenter) ; sorties
facultatives (pilote, nombre de groupes plats).

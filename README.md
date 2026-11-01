# EuroSAT : classification d'images satellites

Classification de vignettes Sentinel-2 (64×64 px, RGB) en 10 types d'occupation du sol, avec le jeu de données [EuroSAT](https://github.com/phelber/EuroSAT)
(27 000 images).

## Classes

AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture,PermanentCrop, Residential, River, SeaLake.

## Étapes réalisées

1. Exploration : image par classe, couleur moyenne par classe
2. Baselines scikit-learn avec des variables construites à la main
3. Comparaison avec les pixels bruts
4. Elaboration d'un CNN pour classifier 

## Résultats (20 % des images en test)

| Méthode               | Entrée                | Précision     |
|-----------------------|-----------------------|---------------|
| Plus proche centroïde | 8 variables           | 0,489         |
| Forêt aléatoire       | pixels bruts (12 288) | 0,675         |
| Forêt aléatoire       | couleur moyenne (3)   | 0,605 ± 0,005 |
| Forêt aléatoire       | + écart-type (6)      | 0,774 ± 0,006 |
| Forêt aléatoire       | + texture (8)         | 0,823 ± 0,005 |

## Ce que j'ai observé

- Le plus gros gain provient de la prise en compte de l'écart-type entre chaque pixel. Cependant il existe un nombre de paramètre incalculable qui permettent de s'approcher d'une superbe prédiction. Avoir ajouté la question de la texture (regarder si les pixels de même couleur se suivent ou sont éparpillés) à permis de bien renforcer mon modèle par exemple
- Avec ce jeu de données, les classes qui sont beaucoup confondues sont les rivières avec les autoroutes. Très proche l'un de l'autre si l'on se base sur les variables avec lesquelles je nourri mon modèle.
- On en vient à comprendre pouquoi les pixels brutes font moins bien que les variables construites. Ils ne prennent pas en compte leur environnement ni qui est positionné à côté de qui. Cela apporte des déchets.

## Lancer le projet

Les notebooks sont exécutés sur Google Colab (GPU gratuit). Le jeu de données se télécharge automatiquement avec `torchvision.datasets.EuroSAT(download=True)`.

## Suite prévue

- CNN en PyTorch, puis transfer learning
- Analyse des erreurs (matrice de confusion)
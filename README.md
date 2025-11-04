# Analyse de paniers avec MLxtend

Ce projet analyse un jeu de données de transactions d'une boutique de cadeaux britannique (`dataset_baskets_dated.csv`). L'objectif est de nettoyer les données, d'effectuer une analyse statistique descriptive et de découvrir des règles d'association (itemsets) à l'aide de `pandas`, `seaborn` et `mlxtend`.

L'analyse complète est contenue dans le script `analyser_paniers.py`.

---

## Fonctionnalités

Le script `analyser_paniers.py` exécute les tâches suivantes :

1.  **Chargement et nettoyage :**
    * Chargement du CSV et gestion des types de données (dates, ID clients).
    * **Nettoyage syntaxique** : Conversion des noms de produits en minuscules et suppression des espaces superflus.
    * **Nettoyage sémantique** : Identification et filtrage des articles "non-produits" (ex: 'postage', 'manual', 'bank charges', 'discount') pour ne conserver que les articles pertinents pour l'analyse.
    * Suppression des paniers vides après nettoyage.

2.  **Analyse statistique et descriptive :**
    * Calcul des statistiques globales (nombre de paniers, clients, articles uniques).
    * Simulation de "pruning" (élagage) pour estimer l'impact de différents seuils de support.
    * Analyse des clients les plus actifs (par nombre de paniers et d'articles).
    * Classement des 20 articles les plus populaires.

3.  **Visualisation de données :**
    * Génération et sauvegarde automatique de 4 graphiques dans le dossier `/graphiques` :
        * Distribution des paniers par mois.
        * Distribution des paniers par jour de la semaine et par heure.
        * Distribution de la taille des paniers (après filtrage).
        * Top 20 des articles les plus vendus.

4.  **Analyse des règles d'association (MLxtend) :**
    * Préparation et encodage One-Hot des transactions valides.
    * Application de l'algorithme **Apriori** pour extraire les itemsets fréquents (support min: 2%, taille max: 5).
    * Génération des **règles d'association** (confiance min: 70%).
    * Affichage des 20 meilleures règles, triées par **lift** pour identifier les associations les plus fortes et les moins évidentes.

---

## Prérequis et installation

Pour exécuter ce projet, vous aurez besoin de Python3 et des bibliothèques listées dans `requirements.txt`.

1.  **Clonez ce dépôt :**
    ```bash
    git clone [https://github.com/sftss/TP-Data-MLxtend](https://github.com/sftss/TP-Data-MLxtend)
    cd VOTRE_REPO
    ```

2.  **Créez un environnement virtuel (recommandé) :**
    ```bash
    python -m venv venv
    ```

3.  **Activez l'environnement :**
    * Sur macOS/Linux :
        ```bash
        source venv/bin/activate
        ```
    * Sur Windows :
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Installez les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

---

## Lancement de l'analyse

Pour lancer l'analyse complète, exécutez le script principal :

```bash
python analyser_paniers.py
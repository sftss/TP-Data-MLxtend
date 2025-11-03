# ==============================================================================
# SCRIPT D'ANALYSE STATISTIQUE
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, ast

# Configuration
FILEPATH = "dataset_baskets_dated.csv"
sns.set_theme(style="whitegrid")

def load_and_clean_data(filepath: str) -> pd.DataFrame | None:
    """
    Charge, nettoie et formate les données du CSV
    """
    print(f"--- 1. Chargement et Nettoyage de {filepath} ---")
    
    # --- 1a. Chargement ---
    try:
        # L'encodage par défaut (UTF-8) semble correct, sinon 'latin1'
        df = pd.read_csv(filepath)
        print(f"Succès : Fichier chargé. {df.shape[0]} lignes initiales.")
        print(f"Colonnes trouvées: {df.columns.to_list()}")
    except FileNotFoundError:
        print(f"ERREUR : Fichier non trouvé à l'emplacement : {filepath}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"ERREUR : Impossible de lire le fichier. {e}", file=sys.stderr)
        return None

    # --- 1b. Vérification et Nettoyage ---
    
    # Vérifier les colonnes de l'extrait
    required_cols = ['basket_id', 'date_trans', 'customer_id', 'products']
    if not all(col in df.columns for col in required_cols):
        print(f"ERREUR: Colonnes requises manquantes. {required_cols} sont attendues.", file=sys.stderr)
        return None

    # Nettoyer 'customer_id' (ex: 17850.0) et supprimer les lignes sans client
    df.dropna(subset=['customer_id'], inplace=True)
    df['customer_id'] = df['customer_id'].astype(int)

    # Conversion de la date (timestamp)
    try:
        # 'date_trans' est un timestamp, nous le convertissons
        df['date'] = pd.to_datetime(df['date_trans'])
        print("Conversion de 'date_trans' (timestamp) en datetime: OK")
    except Exception as e:
        print(f"AVERTISSEMENT : Échec de la conversion de 'date_trans'. {e}")
        df['date'] = pd.NaT # Mettre à "Not a Time" si échec

    # Conversion de la colonne 'products' (string-list vers VRAIE liste)
    # C'est l'étape la plus importante
    print("Conversion de la colonne 'products' (string -> list)...")
    clean_products = []
    for item_str in df['products']:
        try:
            # ast.literal_eval convertit "['A', 'B']" en ['A', 'B']
            # C'est la seule méthode SÛRE pour le faire.
            evaluated = ast.literal_eval(str(item_str))
            
            # On s'assure que c'est bien une liste
            if isinstance(evaluated, list):
                clean_products.append(evaluated)
            else:
                clean_products.append([]) # Ex: la ligne [nan]
        except (ValueError, SyntaxError, TypeError):
            # En cas d'erreur de formatage, on met une liste vide
            clean_products.append([]) 
    
    df['products_list'] = clean_products

    # Supprimer les paniers vides (ceux qui étaient [nan] ou mal formés)
    initial_rows = df.shape[0]
    df = df[df['products_list'].str.len() > 0]
    print(f"{initial_rows - df.shape[0]} paniers vides ou invalides supprimés.")
    
    print(f"Nettoyage terminé. {df.shape[0]} paniers valides restants.")
    print("-" * 50)
    return df

def print_global_stats(df: pd.DataFrame):
    """
    Affiche les statistiques descriptives globales.
    """
    print("--- 2. Statistiques Descriptives Globales ---")
    
    date_debut = df['date'].min()
    date_fin = df['date'].max()
    nb_paniers = df['basket_id'].nunique()
    nb_clients = df['customer_id'].nunique()

    print(f"Période analysée  : du {date_debut.date()} au {date_fin.date()}")
    print(f"Paniers (transactions) uniques : {nb_paniers:,}")
    print(f"Clients uniques       : {nb_clients:,}")
    # Le nombre d'articles uniques sera calculé à l'étape 4
    print("-" * 50)

def analyze_temporal_distribution(df: pd.DataFrame):
    """
    Crée et affiche un graphique du nombre de paniers par mois.
    """
    print("--- 3. Analyse Temporelle (Génération du graphique) ---")
    
    # Nous n'avons pas de 'Revenue', nous comptons donc les paniers
    paniers_mensuels = df.set_index('date').resample('M')['basket_id'].count()

    plt.figure(figsize=(14, 7))
    ax = paniers_mensuels.plot(kind='line', marker='o', color='royalblue')
    ax.set_title('Distribution des Paniers par Mois', fontsize=16)
    ax.set_xlabel('Mois')
    ax.set_ylabel('Nombre de Paniers')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    print("Graphique du nombre de paniers mensuels généré.")
    print("-" * 50)

def analyze_popular_items(df: pd.DataFrame):
    """
    Affiche les articles les plus populaires en "explosant" les listes de paniers.
    """
    print("--- 4. Analyse des Articles Populaires ---")
    
    # 'df' a une colonne 'products_list' qui contient des listes.
    # 'explode' va créer une nouvelle ligne pour chaque item dans chaque liste.
    # C'est l'opération clé pour analyser les articles.
    all_items_series = df.explode('products_list')['products_list']

    print(f"Nombre total d'articles (non uniques) dans tous les paniers : {len(all_items_series):,}")

    # value_counts() fait le décompte de chaque article
    articles_populaires_brut = all_items_series.value_counts()
    
    print(f"Nombre total d'articles uniques trouvés : {len(articles_populaires_brut):,}")
    
    print("\n--- TOP 20 ARTICLES (BRUT) ---")
    print(articles_populaires_brut.head(20).to_string())
    
    # --- Filtrage ---
    # 'POSTAGE' a été vu dans l'extrait
    articles_a_exclure = ['POSTAGE', 'nan'] 
    
    # L'index de la Series contient les noms des articles
    articles_populaires_filtres = articles_populaires_brut[
        ~articles_populaires_brut.index.astype(str).str.contains('|'.join(articles_a_exclure), case=False, na=True)
    ]
    
    print("\n--- TOP 20 ARTICLES (FILTRÉS : hors frais, etc.) ---")
    print(articles_populaires_filtres.head(20).to_string())
    print("-" * 50)

# ==============================================================================
# SCRIPT PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    
    # Chargement
    df_clean = load_and_clean_data(FILEPATH)
    
    if df_clean is not None:
        # Statistiques Globales
        print_global_stats(df_clean)
        
        # Analyse Temporelle
        analyze_temporal_distribution(df_clean)
        
        # Étape 4: Articles Populaires
        # (L'analyse géographique n'est pas possible, pas de colonne 'Country')
        analyze_popular_items(df_clean)
        
        print("\n✅ Analyse statistique de base terminée.")
        print("Les graphiques se sont affichés dans des fenêtres séparées.")
    else:
        print("❌ Échec => impossible de charger ou de nettoyer les données.", file=sys.stderr)
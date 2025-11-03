# ==============================================================================
# SCRIPT D'ANALYSE STATISTIQUE AVANCÉE
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, ast

# Configuration
FILEPATH = "dataset_baskets_dated.csv"
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (14, 7) # taille graphiques

def load_and_clean_data(filepath: str):
    """
    Charge, nettoie et formate les données du CSV
    """
    print(f"\nChargement et nettoyage de {filepath} ")
    
    df = pd.read_csv(filepath)
    print(f"Fichier chargé : {df.shape[0]} lignes")
    
    df.dropna(subset=['customer_id'], inplace=True)
    df['customer_id'] = df['customer_id'].astype(int)

    try:
        df['date'] = pd.to_datetime(df['date_trans'])
        print("Conversion de 'date_trans' (timestamp) en datetime: OK")
    except Exception:
        df['date'] = pd.NaT 

    # conversion de la colonne 'products' en liste
    print("Conversion de 'products...")
    
    def safe_literal_eval(item_str):
        try:
            return ast.literal_eval(str(item_str))
        except:
            return []


    df['products_list'] = df['products'].apply(safe_literal_eval)
    df['basket_size'] = df['products_list'].str.len() # taille du panier

    # supprimer les paniers vides (nan ou mal formés)
    initial_rows = df.shape[0]
    df = df[df['basket_size'] > 0]
    print(f"{initial_rows - df.shape[0]} paniers vides et invalides supprimés")
    print(f"Il y a {df.shape[0]} paniers valides restants.")
    print("-" * 50)
    return df

def get_all_items_series(df: pd.DataFrame) -> pd.Series:
    """
    Explose le dataframe pour obtenir une Series de tous les articles vendus
    """
    print("Préparation de la liste de tous les articles...")
    all_items_series = df.explode('products_list')['products_list']
    
    # Nettoyage : supprimer valeurs vides et articles exclus
    all_items_series = all_items_series.dropna()
    all_items_series = all_items_series[all_items_series.astype(str).str.strip() != '']
    articles_a_exclure = ['POSTAGE']
    all_items_series = all_items_series[~all_items_series.isin(articles_a_exclure)]
    return all_items_series

def print_global_stats(df: pd.DataFrame, all_items_series: pd.Series):
    """
    Affiche les statistiques descriptives + simulation de pruning
    """
    print("Statistiques Descriptives + Analyse de Pruning")
    
    nb_paniers = df['basket_id'].nunique()
    nb_clients = df['customer_id'].nunique()
    item_counts = all_items_series.value_counts()

    print(f"Paniers uniques : {nb_paniers:,}")
    print(f"Clients uniques : {nb_clients:,}")
    print(f"Articles uniques (filtrés) : {len(item_counts):,}")
    print(f"Articles vendus (total) : {len(all_items_series):,}")

    print("\nSIMULATION DE PRUNING")

    pruning_stats = []
    for support_pct in [5, 2, 1, 0.5, 0.2, 0.1]:
        min_support = support_pct / 100.0
        min_count = int(min_support * nb_paniers)
        
        # combien d'articles dépassent ce seuil
        items_restants = (item_counts >= min_count).sum()
        
        pruning_stats.append({
            "Seuil (min_support)": f"{support_pct}%",
            "Nb Paniers Min.": f"{min_count} paniers",
            "Articles Restants": f"{items_restants} / {len(item_counts)}",
            "Impact": f"{items_restants / len(item_counts):.1%}"
        })
    
    print(pd.DataFrame(pruning_stats).to_string(index=False))
    print("-" * 50)

def analyze_distributions(df: pd.DataFrame):
    """
    Crée 4 graphiques :
    1. Paniers par mois
    2. Paniers par jour de la semaine
    3. Paniers par heure
    4. Taille des paniers
    """
    print("Analyse des Distributions")

    #  paniers par mois
    paniers_mensuels = df.set_index('date').resample('M')['basket_id'].count()
    plt.figure(figsize=(14, 7))
    ax1 = paniers_mensuels.plot(kind='line', marker='o', color='royalblue')
    ax1.set_title('Distribution des Paniers par Mois', fontsize=16)
    ax1.set_xlabel('Mois')
    ax1.set_ylabel('Nombre de Paniers')
    plt.tight_layout()
    plt.show()

    #  paniers par jour et heure
    df['weekday'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    
    weekdays_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(14, 12))
    
    sns.countplot(data=df, x='weekday', ax=ax2, order=weekdays_order, palette="Blues_d")
    ax2.set_title('Paniers par Jour de la Semaine', fontsize=16)
    ax2.set_xlabel('Jour')
    ax2.set_ylabel('Nombre de Paniers')
    
    sns.countplot(data=df, x='hour', ax=ax3, palette="Oranges_d")
    ax3.set_title('Paniers par Heure de la Journée', fontsize=16)
    ax3.set_xlabel('Heure (0-23)')
    ax3.set_ylabel('Nombre de Paniers')
    
    plt.tight_layout()
    plt.show()

    #  taille des paniers
    plt.figure(figsize=(14, 7))
    # limiter l'axe des X
    max_size = int(df['basket_size'].quantile(0.99))

    ax4 = sns.histplot(data=df, x='basket_size', bins=range(1, max_size + 2), kde=False, color='green')
    ax4.set_title(f'Distribution de la Taille des Paniers (jusqu\'à {max_size} articles)', fontsize=16)
    ax4.set_xlabel('Nombre d\'articles dans le panier')
    ax4.set_ylabel('Nombre de Paniers')
    ax4.set_yscale('log') # échelle log pour voir les paniers plus grands

    # stats clés
    mean_size = df['basket_size'].mean()
    median_size = df['basket_size'].median()
    print(f"Stats - Taille des paniers: Moyenne={mean_size:.2f}, Médiane={median_size:.0f}")
    print("Info: L'échelle du graphique 'Taille des Paniers' est logarithmique.")
    
    plt.tight_layout()
    plt.show()
    print("Graphiques de distribution générés")
    print("-" * 50)

def analyze_popular_items(all_items_series: pd.Series):
    """
    Affiche le Top 20 et génère un Word Cloud
    """
    # print("Analyse des Articles Populaires")
    
    # item_counts = all_items_series.value_counts()

    # print("\nTOP 20 ARTICLES")
    # print(item_counts.head(20).to_string())
    
    # print("\nGénération du nuage de mots (Word Cloud)...")
    # # Transformer les comptes en dictionnaire pour le word cloud
    
    # plt.figure(figsize=(15, 8))
    # plt.axis('off')
    # plt.title('Top 100 des Articles les Plus Fréquents', fontsize=16)
    # plt.show()

    # print("-" * 50)
    
    # return item_counts.head(100)

def analyze_customer_activity(df: pd.DataFrame, all_items_series: pd.Series):
    """
    Affiche le Top 10 des clients par paniers et par articles.
    """
    print("Analyse de l'Activité Client")
    
    # top 10 par nombre de paniers
    top_clients_baskets = df['customer_id'].value_counts().head(10).reset_index()
    top_clients_baskets.columns = ['ID Client (Paniers)', 'Nb de Paniers']

    # top 10 par nombre d'articles
    df_exploded = df.explode('products_list')
    top_clients_items = df_exploded['customer_id'].value_counts().head(10).reset_index()
    top_clients_items.columns = ['ID Client (Articles)', 'Nb d\'Articles']

    # affichage côte à côte
    top_clients_df = pd.concat([top_clients_baskets, top_clients_items], axis=1)
    
    print("\nTOP 10 CLIENTS (par Nb de Paniers vs Nb d'Articles) ")
    print(top_clients_df.to_string(index=False))
    print("-" * 50)


# ==============================================================================
# SCRIPT PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    
    df_clean = load_and_clean_data(FILEPATH)
    if df_clean is not None:
        # calculer la liste de tous les articles
        all_items_series = get_all_items_series(df_clean)

        # statistiques Globales & Pruning
        print_global_stats(df_clean, all_items_series)

        # distributions (temps, jour, heure, taille)
        analyze_distributions(df_clean)

        # articles populaires
        analyze_popular_items(all_items_series)

        # activité client
        analyze_customer_activity(df_clean, all_items_series)

        print("\nAnalyse terminée")
    else:
        print("Échec : impossible de charger ou de nettoyer les données", file=sys.stderr)
# analyser_paniers_pro.py
# ==============================================================================
# SCRIPT D'ANALYSE STATISTIQUE AVANCÉE
#
# Objectif :
# 1. Charger et nettoyer les données (structure 1 panier/ligne).
# 2. Calculer les statistiques descriptives et de "pruning".
# 3. Analyser les distributions (temporelle, horaire, taille des paniers).
# 4. Analyser les articles populaires (Top N et Word Cloud).
# 5. Analyser les clients (Top 10).
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import ast  # Pour convertir en toute sécurité le string "['item1', ...]" en liste

# --- Dépendances optionnelles ---
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False
    print("AVERTISSEMENT: 'wordcloud' non installé. Le nuage de mots sera désactivé.")
    print("Pour l'installer : pip install wordcloud")

try:
    from tqdm import tqdm
    # Appliquer tqdm à toutes les opérations 'apply' de pandas
    tqdm.pandas(desc="Progression")
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("AVERTISSEMENT: 'tqdm' non installé. Pas de barre de progression.")
    print("Pour l'installer : pip install tqdm")

# --- 0. Configuration ---
FILEPATH = "dataset_baskets_dated.csv"
sns.set(style="whitegrid", palette="muted") # Style global
plt.rcParams['figure.figsize'] = (14, 7) # Taille des graphiques

def load_and_clean_data(filepath: str) -> pd.DataFrame | None:
    """
    Charge, nettoie et formate les données du CSV (structure 1 panier/ligne).
    """
    print(f"\n--- 1. Chargement et Nettoyage de {filepath} ---")
    
    try:
        df = pd.read_csv(filepath)
        print(f"Succès : Fichier chargé. {df.shape[0]} lignes initiales.")
    except FileNotFoundError:
        print(f"ERREUR : Fichier non trouvé à : {filepath}", file=sys.stderr)
        return None
    
    required_cols = ['basket_id', 'date_trans', 'customer_id', 'products']
    if not all(col in df.columns for col in required_cols):
        print(f"ERREUR: Colonnes requises manquantes. {required_cols} sont attendues.", file=sys.stderr)
        return None

    df.dropna(subset=['customer_id'], inplace=True)
    df['customer_id'] = df['customer_id'].astype(int)

    try:
        df['date'] = pd.to_datetime(df['date_trans'])
        print("Conversion de 'date_trans' (timestamp) en datetime: OK")
    except Exception:
        df['date'] = pd.NaT 

    # --- Conversion de la colonne 'products' (string-list vers VRAIE liste) ---
    print("Conversion de la colonne 'products' (string -> list)...")
    
    def safe_literal_eval(item_str):
        try:
            evaluated = ast.literal_eval(str(item_str))
            return evaluated if isinstance(evaluated, list) else []
        except (ValueError, SyntaxError, TypeError):
            return []

    # Utilise 'progress_apply' si tqdm est dispo, sinon 'apply'
    if HAS_TQDM:
        df['products_list'] = df['products'].progress_apply(safe_literal_eval)
    else:
        df['products_list'] = df['products'].apply(safe_literal_eval)

    df['basket_size'] = df['products_list'].str.len()
    
    # Supprimer les paniers vides (ceux qui étaient [nan] ou mal formés)
    initial_rows = df.shape[0]
    df = df[df['basket_size'] > 0]
    print(f"{initial_rows - df.shape[0]} paniers vides ou invalides supprimés.")
    
    print(f"Nettoyage terminé. {df.shape[0]} paniers valides restants.")
    print("-" * 50)
    return df

def get_all_items_series(df: pd.DataFrame) -> pd.Series:
    """
    "Explose" le dataframe pour obtenir une Series de tous les articles vendus.
    """
    print("Préparation de la liste de tous les articles (explosion)...")
    all_items_series = df.explode('products_list')['products_list']
    
    # Nettoyage des articles (ex: 'POSTAGE' ou 'nan' en string)
    articles_a_exclure = ['POSTAGE', 'nan', ''] 
    all_items_series = all_items_series[
        ~all_items_series.astype(str).str.contains('|'.join(articles_a_exclure), case=False, na=True)
    ]
    return all_items_series

def print_global_stats(df: pd.DataFrame, all_items_series: pd.Series):
    """
    Affiche les statistiques descriptives et la simulation de pruning.
    """
    print("--- 2. Statistiques Descriptives et Analyse de Pruning ---")
    
    nb_paniers = df['basket_id'].nunique()
    nb_clients = df['customer_id'].nunique()
    item_counts = all_items_series.value_counts()
    
    print(f"Paniers (transactions) uniques : {nb_paniers:,}")
    print(f"Clients uniques       : {nb_clients:,}")
    print(f"Articles uniques (filtrés) : {len(item_counts):,}")
    print(f"Articles vendus (total) : {len(all_items_series):,}")
    
    print("\n--- 💡 SIMULATION DE PRUNING (ÉLAGAGE) ---")
    print("Info: Cette table vous aide à choisir le 'min_support' pour Apriori.")
    
    pruning_stats = []
    for support_pct in [5, 2, 1, 0.5, 0.2, 0.1]:
        min_support = support_pct / 100.0
        min_count = int(min_support * nb_paniers)
        
        # Compter combien d'articles dépassent ce seuil
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
    Crée 4 graphiques de distribution :
    1. Paniers par mois
    2. Paniers par jour de la semaine
    3. Paniers par heure
    4. Taille des paniers (Histogramme)
    """
    print("--- 3. Analyse des Distributions (Génération des graphiques) ---")
    
    # --- 3a. Paniers par Mois ---
    paniers_mensuels = df.set_index('date').resample('M')['basket_id'].count()
    plt.figure(figsize=(14, 7))
    ax1 = paniers_mensuels.plot(kind='line', marker='o', color='royalblue')
    ax1.set_title('Distribution des Paniers par Mois', fontsize=16)
    ax1.set_xlabel('Mois')
    ax1.set_ylabel('Nombre de Paniers')
    plt.tight_layout()
    plt.show()

    # --- 3b & 3c. Jour de la Semaine et Heure ---
    df['weekday'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    
    # Ordonner les jours
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

    # --- 3d. Taille des Paniers ---
    plt.figure(figsize=(14, 7))
    # Limiter l'axe des X pour une meilleure lisibilité (ex: 99e percentile)
    max_size = int(df['basket_size'].quantile(0.99))
    
    ax4 = sns.histplot(data=df, x='basket_size', bins=range(1, max_size + 2), kde=False, color='green')
    ax4.set_title(f'Distribution de la Taille des Paniers (jusqu\'à {max_size} articles)', fontsize=16)
    ax4.set_xlabel('Nombre d\'articles dans le panier')
    ax4.set_ylabel('Nombre de Paniers')
    ax4.set_yscale('log') # Échelle log pour voir les paniers plus grands
    
    # Afficher les stats clés
    mean_size = df['basket_size'].mean()
    median_size = df['basket_size'].median()
    print(f"Stats - Taille des paniers: Moyenne={mean_size:.2f}, Médiane={median_size:.0f}")
    print("Info: L'échelle du graphique 'Taille des Paniers' est logarithmique.")
    
    plt.tight_layout()
    plt.show()
    print("Graphiques de distribution générés.")
    print("-" * 50)

def analyze_popular_items(all_items_series: pd.Series):
    """
    Affiche le Top 20 et génère un Word Cloud (si possible).
    """
    print("--- 4. Analyse des Articles Populaires ---")
    
    item_counts = all_items_series.value_counts()
    
    print("\n--- TOP 20 ARTICLES (FILTRÉS) ---")
    print(item_counts.head(20).to_string())
    
    if HAS_WORDCLOUD:
        print("\nGénération du nuage de mots (Word Cloud)...")
        # Transformer les comptes en dictionnaire pour le word cloud
        wordcloud_data = item_counts.head(100).to_dict()
        
        wc = WordCloud(width=1000, 
                       height=500, 
                       background_color='white', 
                       colormap='viridis').generate_from_frequencies(wordcloud_data)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title('Top 100 des Articles les Plus Fréquents', fontsize=16)
        plt.show()
    
    print("-" * 50)

def analyze_customer_activity(df: pd.DataFrame, all_items_series: pd.Series):
    """
    Affiche le Top 10 des clients par paniers et par articles.
    """
    print("--- 5. Analyse de l'Activité Client ---")
    
    # Top 10 par nombre de paniers
    top_clients_baskets = df['customer_id'].value_counts().head(10).reset_index()
    top_clients_baskets.columns = ['ID Client (Paniers)', 'Nb de Paniers']
    
    # Top 10 par nombre d'articles
    # Nous avons besoin de 'df_exploded' (mais 'all_items_series' est basé dessus)
    # Recréons-le avec l'ID client
    df_exploded = df.explode('products_list')
    top_clients_items = df_exploded['customer_id'].value_counts().head(10).reset_index()
    top_clients_items.columns = ['ID Client (Articles)', 'Nb d\'Articles']
    
    # Concaténer pour un affichage côte à côte
    top_clients_df = pd.concat([top_clients_baskets, top_clients_items], axis=1)
    
    print("\n--- TOP 10 CLIENTS (par Nb de Paniers vs Nb d'Articles) ---")
    print(top_clients_df.to_string(index=False))
    print("-" * 50)


# ==============================================================================
# SCRIPT PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    
    # Étape 1: Chargement et Nettoyage
    df_clean = load_and_clean_data(FILEPATH)
    
    if df_clean is not None:
        # Pré-calculer la liste de tous les articles (utilisée par 3 fonctions)
        all_items_series = get_all_items_series(df_clean)
        
        # Étape 2: Statistiques Globales & Pruning
        print_global_stats(df_clean, all_items_series)
        
        # Étape 3: Distributions (Temps, Jour, Heure, Taille)
        analyze_distributions(df_clean)
        
        # Étape 4: Articles Populaires & Word Cloud
        analyze_popular_items(all_items_series)
        
        # Étape 5: Activité Client
        analyze_customer_activity(df_clean, all_items_series)
        
        print("\n✅ Analyse terminée.")
        print("Tous les graphiques se sont affichés dans des fenêtres séparées.")
    else:
        print("❌ Échec de l'analyse : impossible de charger ou de nettoyer les données.", file=sys.stderr)
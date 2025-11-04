import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, ast, os
from mlxtend.preprocessing import TransactionEncoder  # NOUVEAU
from mlxtend.frequent_patterns import apriori, association_rules  # NOUVEAU

FILEPATH = 'dataset_baskets_dated.csv'
OUTPUT_DIR = 'graphiques'
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (14, 7)

def clean_item_name(item_name):
    """Nettoie un nom d'article"""
    if not isinstance(item_name, str):
        return None
    name = item_name.lower()
    name = name.strip()
    name = name.strip('\'"')
    return name

def load_and_clean_data(filepath: str):
    """Charge, nettoie et formate les données"""
    print(f"\nChargement et nettoyage de {filepath}...")
    df = pd.read_csv(filepath)
    print(f"{df.shape[0]} lignes chargées")
    
    # nettoyage clients
    df.dropna(subset=['customer_id'], inplace=True)
    df['customer_id'] = df['customer_id'].astype(int)

    # nettoyage dates
    try:
        df['date'] = pd.to_datetime(df['date_trans'])
        print('Conversion date_trans timestamp en datetime')
    except Exception:
        df['date'] = pd.NaT

    # conversion de products en liste avec nettoyage
    print('Conversion de products en liste...')
    def safe_literal_eval_and_clean(item_str):
        try:
            items = ast.literal_eval(str(item_str))
            if isinstance(items, list):
                cleaned_items = [clean_item_name(item) for item in items]
                return [item for item in cleaned_items if item and item.strip()]
            else:
                return []
        except:
            return []

    df['products_list'] = df['products'].apply(safe_literal_eval_and_clean)
    df['basket_size'] = df['products_list'].str.len()
    
    initial_rows = df.shape[0]
    df = df[df['basket_size'] > 0]
    print(f"{initial_rows - df.shape[0]} paniers vides supprimés")
    print(f"{df.shape[0]} paniers restants")
    print('-' * 50)
    return df

# filtrage sémantique
def filter_and_get_all_items(df: pd.DataFrame):
    """Filtre les produits des listes et retourne la Series de tous les articles valides"""
    print("Filtrage sémantique...")
    
    all_items_series_full = df.explode('products_list')['products_list'].dropna()
    all_items_series_full = all_items_series_full[all_items_series_full.astype(str).str.strip() != '']
    
    junk_keywords = [
        'postage', 'manual', 'bank charges', 'cruk', 'samples', 
        'adjustment', 'return', 'amazon fee', 'discount', 
        'dotcom', 'shipping', 'carrier', 'matrix', 'faulty', 'check',
        'bad debt', 'write off'
    ]
    pattern = '|'.join(junk_keywords)

    # trouver + exclure les produits uniques qui sont mauvais
    junk_items_set = set(all_items_series_full[all_items_series_full.str.contains(pattern, case=False, na=False)].unique())
    print(f"{len(junk_items_set)} produits uniques pour exclusion")

    # Serie des produits VALIDES
    all_items_series_valid = all_items_series_full[~all_items_series_full.isin(junk_items_set)]
    print(f"{len(all_items_series_full) - len(all_items_series_valid)} de produits mauvais supprimées de la Series")

    # MAJ le DataFrame
    def filter_junk_from_list(item_list):
        return [item for item in item_list if item not in junk_items_set]
    
    df['products_list_filtered'] = df['products_list'].apply(filter_junk_from_list)

    # MAJ taille du panier basée sur la liste filtrée
    df['basket_size_filtered'] = df['products_list_filtered'].str.len()

    # filtrer paniers vides
    initial_rows = df.shape[0]
    df = df[df['basket_size_filtered'] > 0].copy() # éviter SettingWithCopyWarning
    print(f"{initial_rows - df.shape[0]} paniers sont devenus vides après filtrage 'junk' et ont été supprimés.")
    print(f"Il y a {df.shape[0]} paniers valides restants pour l'analyse.")
    print('-' * 50)
    
    return df, all_items_series_valid

def print_global_stats(df: pd.DataFrame, all_items_series: pd.Series):
    """Affiche les statistiques descriptives (basées sur les données filtrées)."""
    print('Statistiques Descriptives (après nettoyage complet)')
    # utilise les données filtrées
    nb_paniers = df['basket_id'].nunique()
    nb_clients = df['customer_id'].nunique()
    item_counts = all_items_series.value_counts()

    print(f"Paniers uniques (nets) : {nb_paniers:,}")
    print(f"Clients uniques (nets) : {nb_clients:,}")
    print(f"Articles uniques (nets) : {len(item_counts):,}")
    print(f"Articles vendus (nets) : {len(all_items_series):,}")

    print('\nSIMULATION DE PRUNING (sur paniers nets)')
    pruning_stats = []
    for support_pct in [5, 2, 1, .5, .2, .1]:
        min_support = support_pct / 100
        min_count = int(min_support * nb_paniers)
        items_restants = (item_counts >= min_count).sum()
        pruning_stats.append({
            'Seuil (min_support)': f"{support_pct}%",
            'Nb Paniers Min.': f"{min_count} paniers",
            'Articles Restants': f"{items_restants} / {len(item_counts)}",
            'Impact': f"{items_restants / len(item_counts):.1%}"
        })
    print(pd.DataFrame(pruning_stats).to_string(index=False))
    print('-' * 50)

def analyze_distributions(df: pd.DataFrame):
    """Crée 4 graphiques"""
    print('Analyse des Distributions')
    
    # Paniers par mois
    paniers_mensuels = df.set_index('date').resample('ME')['basket_id'].count()
    plt.figure(figsize=(14, 7))
    ax1 = paniers_mensuels.plot(kind='line', marker='o', color='royalblue')
    ax1.set_title('Distribution des Paniers par Mois', fontsize=16)
    ax1.set_xlabel('Mois')
    ax1.set_ylabel('Nombre de Paniers')
    plt.tight_layout()
    filepath_mois = os.path.join(OUTPUT_DIR, '1_paniers_par_mois.png')
    plt.savefig(filepath_mois, dpi=150, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé : {filepath_mois}")
    plt.close()

    # Jour et Heure
    df['weekday'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    weekdays_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(14, 12))
    sns.countplot(data=df, x='weekday', hue='weekday', ax=ax2, order=weekdays_order, palette='Blues_d', legend=False)
    ax2.set_title('Paniers par Jour de la Semaine', fontsize=16)
    sns.countplot(data=df, x='hour', hue='hour', ax=ax3, palette='Oranges_d', legend=False)
    ax3.set_title('Paniers par Heure de la Journée', fontsize=16)
    plt.tight_layout()
    filepath_jour_heure = os.path.join(OUTPUT_DIR, '2_paniers_par_jour_et_heure.png')
    plt.savefig(filepath_jour_heure, dpi=150, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé : {filepath_jour_heure}")
    plt.close()

    # Taille des paniers
    plt.figure(figsize=(14, 7))
    max_size = int(df['basket_size_filtered'].quantile(.99))
    ax4 = sns.histplot(data=df, x='basket_size_filtered', bins=range(1, max_size + 2), kde=False, color='green')
    ax4.set_title(f"Distribution de la Taille des Paniers (filtrés, jusqu'à {max_size} articles)", fontsize=16)
    ax4.set_xlabel("Nombre d'articles dans le panier (net)")
    ax4.set_ylabel('Nombre de Paniers')
    ax4.set_yscale('log')
    mean_size = df['basket_size_filtered'].mean()
    median_size = df['basket_size_filtered'].median()
    print(f"Stats - Taille des paniers (nets): Moyenne={mean_size:.2f}, Médiane={median_size:.0f}")
    plt.tight_layout()
    filepath_taille = os.path.join(OUTPUT_DIR, '3_taille_des_paniers_filtres.png')
    plt.savefig(filepath_taille, dpi=150, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé : {filepath_taille}")
    plt.close()
    print('Tous les graphiques de distribution ont été générés et sauvegardés')
    print('-' * 50)

def analyze_popular_items(all_items_series: pd.Series):
    """Affiche top 20 + graphique à barres"""
    print("Analyse des Articles Populaires (après nettoyage complet)")
    item_counts = all_items_series.value_counts()
    
    print("\nTOP 20 DES ARTICLES LES PLUS VENDUS (nets) :")
    print(item_counts.head(20).to_string())
    print('-' * 50)

    plt.figure(figsize=(14, 10))
    top_20_items = item_counts.head(20)
    ax = sns.barplot(x=top_20_items.values, y=top_20_items.index, palette='viridis', orient='h')
    ax.set_title('Top 20 des Articles les Plus Populaires (nets)', fontsize=16)
    ax.set_xlabel('Nombre de Ventes')
    ax.set_ylabel('Article')
    
    plt.tight_layout()
    filepath_top_items = os.path.join(OUTPUT_DIR, '4_top_20_articles.png')
    plt.savefig(filepath_top_items, dpi=150, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé : {filepath_top_items}")
    plt.close()
    print('-' * 50)

def analyze_customer_activity(df: pd.DataFrame, all_items_series: pd.Series):
    """Affiche le Top 10 des clients"""
    print("Analyse de l'Activité Client")
    top_clients_baskets = df['customer_id'].value_counts().head(10).reset_index()
    top_clients_baskets.columns = ['ID Client (Paniers)', 'Nb de Paniers (nets)']

    df_exploded = df.explode('products_list_filtered')
    top_clients_items = df_exploded['customer_id'].value_counts().head(10).reset_index()
    top_clients_items.columns = ['ID Client (Articles)', "Nb d'Articles (nets)"]

    top_clients_df = pd.concat([top_clients_baskets, top_clients_items], axis=1)
    print("\nTOP 10 CLIENTS (par Nb de Paniers vs Nb d'Articles) ")
    print(top_clients_df.to_string(index=False))
    print('-' * 50)

def analyze_association_rules(df: pd.DataFrame, min_support=0.02, max_k=5, min_confidence=0.7):
    """exécute Apriori + règles d'association"""
    print("analyse des règles d'association (Apriori)")

    # utilise products_list_filtered
    transactions_list = df['products_list_filtered'].tolist()
    print(f"{len(transactions_list)} paniers valides pour l'analyse Apriori.")

    # 2. Encoder les transactions
    print("Encodage des transactions (one-hot)...")
    te = TransactionEncoder()
    te_ary = te.fit(transactions_list).transform(transactions_list)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # 3. Appliquer Apriori (Itemsets fréquents)
    print(f"Recherche des itemsets fréquents (Support >= {min_support}, Max_K = {max_k})...")
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True, max_len=max_k)
    
    if frequent_itemsets.empty:
        print(f"AUCUN ITEMSET FRÉQUENT TROUVÉ avec un support >= {min_support}. Essayez un support plus bas.")
        print('-' * 50)
        return

    # triez les 20 meilleurs itemsets
    frequent_itemsets_sorted = frequent_itemsets.sort_values(by='support', ascending=False)
    print("\nTOP 20 DES ITEMSETS FRÉQUENTS :")
    print(frequent_itemsets_sorted.head(20).to_string(index=False))
    print('-' * 50)

    # générer les règles d'association
    print(f"Génération des règles d'association (Confiance >= {min_confidence})...")
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
    
    if rules.empty:
        print(f"AUCUNE RÈGLE D'ASSOCIATION TROUVÉE avec une confiance >= {min_confidence}.")
        print('-' * 50)
        return

    # trier par lift
    rules_sorted = rules.sort_values(by='lift', ascending=False)
    
    print("\nTOP RÈGLES D'ASSOCIATION (Triées par Lift) :")

    cols_to_show = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
    print(rules_sorted[cols_to_show].head(20).to_string(index=False))
    print('-' * 50)


if __name__ == "__main__":
    df_clean = load_and_clean_data(FILEPATH)
    
    if df_clean is not None:
        df_filtered, all_items_valid = filter_and_get_all_items(df_clean)
        
        print_global_stats(df_filtered, all_items_valid)
        analyze_distributions(df_filtered) 
        analyze_popular_items(all_items_valid)
        analyze_customer_activity(df_filtered, all_items_valid)        

        # support 2%, k=5, confiance 70%
        analyze_association_rules(
            df_filtered, 
            min_support=0.02, 
            max_k=5, 
            min_confidence=0.7
        )
        
        print('\nFIN')
    else:
        print('impossible de charger ou de nettoyer les données', file=sys.stderr)
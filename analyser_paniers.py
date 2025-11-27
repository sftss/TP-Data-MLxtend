import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, sys, ast, os
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from datetime import timedelta
from matplotlib.ticker import ScalarFormatter

# changer les valeurs si besoin
FILEPATH = "dataset_baskets_dated.csv"
OUTPUT_DIR = "graphiques"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (14, 7)

def clean_item_name(item_name):
    """Nettoie les articles"""
    if not isinstance(item_name, str):
        return None
    name = item_name.lower()
    name = name.strip()
    name = name.strip('\'"')
    return name

def load_and_clean_data(filepath: str):
    """Charge, nettoie et formate les données"""
    print(f"\nChargement et nettoyage de {filepath}")
    # Petit check de sécurité si le fichier n'est pas là
    if not os.path.exists(filepath):
        print(f"ERREUR: Fichier {filepath} introuvable.")
        return None

    df = pd.read_csv(filepath)
    print(f"{df.shape[0]} lignes chargées")
    
    # nettoyage clients
    df.dropna(subset=["customer_id"], inplace=True)
    df["customer_id"] = df["customer_id"].astype(int)

    # nettoyage dates (ns pour les timestamps)
    df["date"] = pd.to_datetime(df["date_trans"], unit="ns")
    print("Conversion date_trans timestamp en datetime")

    # conversion de products en liste avec nettoyage
    print("Conversion de products en liste")
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

    df["products_list"] = df["products"].apply(safe_literal_eval_and_clean)
    df["basket_size"] = df["products_list"].str.len()

    initial_rows = df.shape[0]
    df = df[df["basket_size"] > 0]
    print(f"{initial_rows - df.shape[0]} paniers vides supprimés")
    print(f"{df.shape[0]} paniers")
    print("-" * 50)
    return df

def filter_and_get_all_items(df: pd.DataFrame):
    """Filtre les produits des listes et retourne la Series de tous les articles"""
    print("Filtrage début")

    all_items_series_full = df.explode("products_list")["products_list"].dropna()
    all_items_series_full = all_items_series_full[all_items_series_full.astype(str).str.strip() != ""]

    # mots-clés à bannir (ajouter les noms suspects)
    junk_keywords = [
        "postage", "manual", "bank charges", "cruk", "samples",
        "adjustment", "return", "amazon fee", "discount",
        "dotcom", "shipping", "carrier", "matrix", "faulty", "check",
        "bad debt", "write off"
    ]
    pattern = "|".join(junk_keywords)

    # trouver + exclure les produits uniques qui sont mauvais
    junk_items_set = set(all_items_series_full[all_items_series_full.str.contains(pattern, case=False, na=False)].unique())
    print(f"{len(junk_items_set)} produits uniques pour exclusion")

    # Serie des produits VALIDES
    all_items_series_valid = all_items_series_full[~all_items_series_full.isin(junk_items_set)]
    print(f"{len(all_items_series_full) - len(all_items_series_valid)} de produits mauvais bannis de la Series")

    # MAJ de la DataFrame
    def filter_junk_from_list(item_list):
        return [item for item in item_list if item not in junk_items_set]

    df["products_list_filtered"] = df["products_list"].apply(filter_junk_from_list)

    # MAJ taille du panier basée sur la liste filtrée
    df["basket_size_filtered"] = df["products_list_filtered"].str.len()

    # filtrer paniers vides
    initial_rows = df.shape[0]
    df = df[df["basket_size_filtered"] > 0].copy()
    print(f"{initial_rows - df.shape[0]} paniers sont devenus vides après filtrage (supprimés)")
    print(f"{df.shape[0]} paniers valides restants")
    print("-" * 50)
    
    return df, all_items_series_valid

def analyze_distributions(df: pd.DataFrame):
    """Crée graphiques taille paniers, jour/heure, paniers par mois"""
    print("Analyse des distributions pour les graphiques")
    
    # 1. paniers par mois
    paniers_mensuels = df.set_index("date").resample("ME")["basket_id"].count()
    plt.figure(figsize=(14, 7))
    ax1 = paniers_mensuels.plot(kind="line", marker="o", color="royalblue")
    ax1.set_title("Distribution des paniers par mois", fontsize=16)
    ax1.set_xlabel("Mois")
    ax1.set_ylabel("Nombre de paniers")
    plt.tight_layout()
    filepath_mois = os.path.join(OUTPUT_DIR, "1_paniers_par_mois.png")
    plt.savefig(filepath_mois, dpi=150, bbox_inches="tight")
    plt.close()

    # 2. jour et heure
    df["weekday"] = df["date"].dt.day_name()
    df["hour"] = df["date"].dt.hour
    weekdays_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(14, 12))
    sns.countplot(data=df, x="weekday", hue="weekday", ax=ax2, order=weekdays_order, palette="Blues_d", legend=False)
    ax2.set_title("Paniers par jour de la semaine", fontsize=16)
    sns.countplot(data=df, x="hour", hue="hour", ax=ax3, palette="Oranges_d", legend=False)
    ax3.set_title("Paniers par heure de la journée", fontsize=16)
    plt.tight_layout()
    filepath_jour_heure = os.path.join(OUTPUT_DIR, "2_paniers_par_jour_et_heure.png")
    plt.savefig(filepath_jour_heure, dpi=150, bbox_inches="tight")
    plt.close()

    # 3. taille des paniers
    plt.figure(figsize=(14, 7))
    max_size = int(df["basket_size_filtered"].quantile(.99))
    ax4 = sns.histplot(data=df, x="basket_size_filtered", bins=range(1, max_size + 2), kde=False, color="green")
    ax4.set_title(f"Distribution de la taille des paniers ({max_size} articles)", fontsize=16)
    ax4.set_xlabel("Nombre d'articles dans le panier")
    ax4.set_ylabel("Nombre de paniers")
    ax4.set_yscale("log")

    # valeurs entières sur l'axe Y
    ax4.yaxis.set_major_formatter(ScalarFormatter())
    ax4.ticklabel_format(style="plain", axis="y")

    mean_size = df["basket_size_filtered"].mean()
    median_size = df["basket_size_filtered"].median()
    plt.tight_layout()
    filepath_taille = os.path.join(OUTPUT_DIR, "3_taille_des_paniers_filtres.png")
    plt.savefig(filepath_taille, dpi=150, bbox_inches="tight")
    plt.close()
    print("-" * 50)

def analyze_popular_items(all_items_series: pd.Series):
    """Affiche top 20 + graphique à barres"""
    item_counts = all_items_series.value_counts()
    
    print("\nTop 20 des articles les plus vendus")

    plt.figure(figsize=(14, 10))
    top_20_items = item_counts.head(20)
    ax = sns.barplot(x=top_20_items.values, y=top_20_items.index, hue=top_20_items.index, palette="viridis", orient="h", legend=False)
    ax.set_title("Top 20 des articles les plus populaires", fontsize=16)
    ax.set_xlabel("Nombre de ventes")
    ax.set_ylabel("Article")

    plt.tight_layout()
    filepath_top_items = os.path.join(OUTPUT_DIR, "4_top_20_articles.png")
    plt.savefig(filepath_top_items, dpi=150, bbox_inches="tight")
    plt.close()
    print("-" * 50)

### --- FONCTION POUR GENERER LE CODE LATEX  --- ###
def generate_latex_tikz(rules_df):
    """Génère le code LaTeX pour visualiser les 3 meilleures règles"""
    print("\n" + "="*50)
    print("CODE LATEX (A copier-coller dans Overleaf)")
    print("="*50)
    
    # On prend les 3 meilleures règles selon le Lift
    top_3 = rules_df.head(3).copy()
    
    # Début du document LaTeX
    latex_code = r"""
\documentclass{article}
\usepackage{tikz}
\usetikzlibrary{shapes, arrows.meta, positioning}
\begin{document}
\begin{figure}[h]
\centering
\begin{tikzpicture}[
    node distance=2cm and 3cm,
    item/.style={rectangle, draw=blue!60, fill=blue!5, very thick, minimum size=7mm},
    rule/.style={circle, draw=red!60, fill=red!5, very thick, minimum size=7mm},
    arrow/.style={->, -{Latex[width=3mm]}, thick}
]
"""
    y_pos = 0
    for idx, row in top_3.iterrows():
        ant = "\\newline ".join(list(row['antecedents']))
        cons = "\\newline ".join(list(row['consequents']))
        lift = row['lift']
        conf = row['confidence']
        
        # Création des noeuds
        block = f"""
    % Regle {idx+1}
    \\node[item, align=center] (ant{idx}) at (0, {y_pos}) {{{ant}}};
    \\node[rule] (r{idx}) [right=of ant{idx}] {{R{idx+1}}};
    \\node[item, align=center] (cons{idx}) [right=of r{idx}] {{{cons}}};
    \\draw[arrow] (ant{idx}) -- (r{idx});
    \\draw[arrow] (r{idx}) -- node[above, font=\\small] {{Lift: {lift:.2f}}} node[below, font=\\small] {{Conf: {conf:.0%}}} (cons{idx});
"""
        latex_code += block
        y_pos -= 3.5 

    latex_code += r"""
\end{tikzpicture}
\caption{Visualisation des 3 meilleures règles d'association}
\end{figure}
\end{document}
"""
    print(latex_code)
    print("="*50 + "\n")


def analyze_association_rules(df: pd.DataFrame, min_support=0.02, max_k=5, min_confidence=0.7):
    """Apriori + règles d'association"""
    transactions_list = df["products_list_filtered"].tolist()
    print(f"{len(transactions_list)} paniers pour Apriori")

    # encoder transactions en matrice SPARSE
    te = TransactionEncoder()
    try:
        # matrice sparse
        te_ary = te.fit(transactions_list).transform(transactions_list, sparse=True)
        df_encoded = pd.DataFrame(te_ary.toarray(), columns=te.columns_).astype(bool)
    except (TypeError, AttributeError):
        # fallback (pb de version)
        te_ary = te.fit(transactions_list).transform(transactions_list)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # appliquer Apriori
    print(f"Recherche itemsets avec support >= {min_support} et max_k = {max_k}")
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True, max_len=max_k)
    
    if frequent_itemsets.empty:
        print(f"Aucun itemset avec un support >= {min_support}, (pt support plus bas)")
        print("-" * 50)
        return None

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    if rules.empty:
        print(f"Aucune règle trouvée avec une confiance >= {min_confidence}")
        print("-" * 50)
        return None

    # trier par lift
    rules_sorted = rules.sort_values(by="lift", ascending=False)
    cols_to_show = ["antecedents", "consequents", "support", "confidence", "lift"]
    print(rules_sorted[cols_to_show].head(20).to_string(index=False))
    print("-" * 50)

    # graphique des règles d'association
    top_rules = rules_sorted.head(15).copy()
    if not top_rules.empty:
        
        top_rules["rule"] = top_rules.apply(
            lambda row: f"{', '.join(row['antecedents'])} → {', '.join(row['consequents'])}", 
            axis=1
        )
        
        plt.figure(figsize=(14, 10))
        ax = sns.barplot(data=top_rules,  x="lift",  y="rule",  hue="confidence", palette="RdYlGn", orient="h", legend=True)
        ax.set_title("Top 15 des règles d'association (par Lift)", fontsize=16)
        ax.set_xlabel("Lift")
        ax.set_ylabel("Règle")
        plt.legend(title="Confiance", loc="lower right")
        plt.tight_layout()
        filepath_rules = os.path.join(OUTPUT_DIR, "5_regles_association.png")
        plt.savefig(filepath_rules, dpi=150, bbox_inches="tight")
        plt.close()
        
    ### --- On retourne les règles pour pouvoir générer le LaTeX ensuite --- ###
    return rules_sorted

if __name__ == "__main__":
    df_clean = load_and_clean_data(FILEPATH)
    if df_clean is not None:
        df_filtered, all_items_valid = filter_and_get_all_items(df_clean)

        # tt les données pour les graphiques
        analyze_distributions(df_filtered) 
        analyze_popular_items(all_items_valid)

        # Filtrage pour les règles d'association
        print("\nFiltrage pour règles d'association")
        if not df_filtered.empty:
            max_date = df_filtered["date"].max()
            periode_avant = max_date - timedelta(days=30) # changer la période ici
            print(f"Période : {periode_avant} à {max_date}")

            df_periode = df_filtered[df_filtered["date"] >= periode_avant].copy()
            print(f"{df_periode.shape[0]} paniers pour MLxtend")

            
            df_rules = analyze_association_rules(
                df_periode,
                min_support=0.02,   # 2% demandé
                max_k=3,            # taille 3 max demandée
                min_confidence=0.8  # 80% demandé
            )
            
            ### --- APPEL DE LA FONCTION LATEX --- ###
            if df_rules is not None and not df_rules.empty:
                generate_latex_tikz(df_rules)

        print("\nFIN")
    else:
        print("impossible de charger / nettoyer les données", file=sys.stderr)
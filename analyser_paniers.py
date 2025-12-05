import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, sys, ast, os
import networkx as nx # LaTeX
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
    def clean_et_parse_item_liste(item_str):
        try:
            items = ast.literal_eval(str(item_str))
            if isinstance(items, list):
                cleaned_items = [clean_item_name(item) for item in items]
                return [item for item in cleaned_items if item and item.strip()]
            else:
                return []
        except:
            return []

    df["products_list"] = df["products"].apply(clean_et_parse_item_liste)
    df["basket_size"] = df["products_list"].str.len()

    initial_rows = df.shape[0]
    df = df[df["basket_size"] > 0]
    print(f"{initial_rows - df.shape[0]} paniers vides supprimés")
    print(f"{df.shape[0]} paniers")
    print("-" * 50)
    return df

def filtrer_et_extraire_all_items(df: pd.DataFrame):
    """Filtre les produits des listes et retourne la Series de tous les articles"""
    print("Filtrage début")

    all_items_series_full = df.explode("products_list")["products_list"].dropna()
    all_items_series_full = all_items_series_full[all_items_series_full.astype(str).str.strip() != ""]

    # mots-clés à bannir (ajouter les noms suspects)
    poubelle_keywords = [
        "postage", "manual", "bank charges", "cruk", "samples",
        "adjustment", "return", "amazon fee", "discount",
        "dotcom", "shipping", "carrier", "matrix", "faulty", "check",
        "bad debt", "write off"
    ]
    pattern = "|".join(poubelle_keywords)

    # trouver + exclure les produits uniques qui sont mauvais
    poubelle_items_set = set(all_items_series_full[all_items_series_full.str.contains(pattern, case=False, na=False)].unique())
    print(f"{len(poubelle_items_set)} produits uniques pour exclusion")

    # Serie des produits VALIDES
    items_series_valide = all_items_series_full[~all_items_series_full.isin(poubelle_items_set)]
    print(f"{len(all_items_series_full) - len(items_series_valide)} de produits mauvais bannis de la Series")

    # MAJ de la DataFrame
    def filter_junk_from_list(item_list):
        return [item for item in item_list if item not in poubelle_items_set]

    df["products_list_filtered"] = df["products_list"].apply(filter_junk_from_list)

    # MAJ taille du panier basée sur la liste filtrée
    df["basket_size_filtered"] = df["products_list_filtered"].str.len()

    # filtrer paniers vides
    initiale_rows = df.shape[0]
    df = df[df["basket_size_filtered"] > 0].copy()
    print(f"{initiale_rows - df.shape[0]} paniers sont devenus vides après filtrage (supprimés)")
    print(f"{df.shape[0]} paniers valides restants")
    print("-" * 50)
    
    return df, items_series_valide

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
    semaine_ordre = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(14, 12))
    sns.countplot(data=df, x="weekday", hue="weekday", ax=ax2, order=semaine_ordre, palette="Blues_d", legend=False)
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

    # mean_size = df["basket_size_filtered"].mean()
    # median_size = df["basket_size_filtered"].median()
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

# ============================================================
# ANALYSE CLIENTS
# ============================================================

def analyse_rfm_clients(df: pd.DataFrame):
    """Analyse RFM (Récence, Fréquence, Valeur) des clients (avec la taille moyenne des paniers comme valeur)"""
    print("\nAnalyse RFM")
    
    date_reference = df["date"].max() + timedelta(days=1)
    
    rfm = df.groupby("customer_id").agg(
        recence=("date", lambda x: (date_reference - x.max()).days),
        frequence=("basket_id", "nunique"),
        valeur_moy=("basket_size_filtered", "mean"),
        nb_articles_total=("basket_size_filtered", "sum")
    ).reset_index()
    
    # score RFM
    rfm["R_score"] = pd.qcut(rfm["recence"], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop")
    rfm["F_score"] = pd.qcut(rfm["frequence"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    rfm["M_score"] = pd.qcut(rfm["valeur_moy"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    
    rfm["RFM_score"] = rfm["R_score"].astype(int) + rfm["F_score"].astype(int) + rfm["M_score"].astype(int)
    
    # segmentation
    def segmenter_client(row):
        r, f, m = int(row["R_score"]), int(row["F_score"]), int(row["M_score"])
        if r >= 4 and f >= 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Clients fidèles"
        elif r >= 4 and f <= 2:
            return "Nouveaux clients"
        elif r <= 2 and f >= 3:
            return "À risque"
        elif r <= 2 and f <= 2:
            return "Perdus"
        else:
            return "Occasionnels"
    
    rfm["segment"] = rfm.apply(segmenter_client, axis=1)
    
    # graphique segments
    plt.figure(figsize=(12, 6))
    segment_counts = rfm["segment"].value_counts()
    couleurs = {"Champions": "#2ecc71", "Clients fidèles": "#3498db", 
              "Nouveaux clients": "#9b59b6", "À risque": "#e74c3c",
              "Perdus": "#95a5a6", "Occasionnels": "#f39c12"}
    ax = sns.barplot(x=segment_counts.index, y=segment_counts.values, 
                     hue=segment_counts.index, palette=couleurs, legend=False)
    ax.set_title("Segmentation RFM des clients", fontsize=16)
    ax.set_xlabel("Segment")
    ax.set_ylabel("Nombre de clients")
    for i, v in enumerate(segment_counts.values):
        ax.text(i, v + 10, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "6_segmentation_rfm.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Segments clients :\n{segment_counts.to_string()}")
    print("-" * 50)
    return rfm


def analyse_fidelite_clients(df: pd.DataFrame):
    """Analyse de la fidélité, distribution des visites, clients uniques vs récurrents"""
    print("\nAnalyse de la fidélité clients")
    
    visites_par_client = df.groupby("customer_id")["basket_id"].nunique()
    
    # stats
    clients_uniques = (visites_par_client == 1).sum()
    clients_recurrents = (visites_par_client > 1).sum()
    clients_fideles = (visites_par_client >= 5).sum()
    
    print(f"Clients uniques (1 visite) : {clients_uniques} ({clients_uniques/len(visites_par_client)*100:.1f}%)")
    print(f"Clients récurrents (2+ visites) : {clients_recurrents} ({clients_recurrents/len(visites_par_client)*100:.1f}%)")
    print(f"Clients fidèles (5+ visites) : {clients_fideles} ({clients_fideles/len(visites_par_client)*100:.1f}%)")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # distribution des visites
    max_visites = int(visites_par_client.quantile(0.95))
    sns.histplot(visites_par_client[visites_par_client <= max_visites], bins=range(1, max_visites + 2), 
                 ax=axes[0], color="steelblue", kde=False)
    axes[0].set_title("Distribution du nombre de visites par client", fontsize=14)
    axes[0].set_xlabel("Nombre de visites")
    axes[0].set_ylabel("Nombre de clients")
    axes[0].set_yscale("log")
    
    # valeurs entières sur l'axe Y
    axes[0].yaxis.set_major_formatter(ScalarFormatter())
    axes[0].ticklabel_format(style="plain", axis="y")
    
    # camembert fidélité
    labels = ["1 visite", "2-4 visites", "5+ visites"]
    tailles = [
        (visites_par_client == 1).sum(),
        ((visites_par_client >= 2) & (visites_par_client < 5)).sum(),
        (visites_par_client >= 5).sum()
    ]
    couleur_partie = ["#e74c3c", "#f39c12", "#2ecc71"]
    axes[1].pie(tailles, labels=labels, autopct="%1.1f%%", colors=couleur_partie, startangle=90)
    axes[1].set_title("Répartition des clients par fidélité", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "7_fidelite_clients.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("-" * 50)


def analyse_evolution_clients(df: pd.DataFrame):
    """Analyse l'évolution des nouveaux clients vs récurrents par mois"""
    print("\nAnalyse évolution nouveaux/récurrents par mois")
    
    df_sorted = df.sort_values("date")
    premiere_visite = df_sorted.groupby("customer_id")["date"].min().reset_index()
    premiere_visite.columns = ["customer_id", "premiere_visite"]
    premiere_visite["mois_premiere_visite"] = premiere_visite["premiere_visite"].dt.to_period("M")
    
    df_merged = df.merge(premiere_visite[["customer_id", "mois_premiere_visite"]], on="customer_id")
    df_merged["mois_achat"] = df_merged["date"].dt.to_period("M")
    df_merged["est_nouveau"] = df_merged["mois_achat"] == df_merged["mois_premiere_visite"]
    
    evolution = df_merged.groupby(["mois_achat", "est_nouveau"])["customer_id"].nunique().unstack(fill_value=0)
    evolution.columns = ["Récurrents", "Nouveaux"]
    evolution.index = evolution.index.astype(str)
    
    plt.figure(figsize=(14, 7))
    evolution.plot(kind="bar", stacked=True, color=["#3498db", "#2ecc71"], ax=plt.gca())
    plt.title("Évolution des clients nouveaux vs récurrents par mois", fontsize=16)
    plt.xlabel("Mois")
    plt.ylabel("Nombre de clients uniques")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Type de client")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "8_evolution_clients.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("-" * 50)

def generee_latex_regles(regles_df: pd.DataFrame):
    """Génère un code LaTeX  pour visualiser les 3 meilleures règles. Utilise NetworkX pour placer les items autour de chaque règle."""
    TOP_K_VISUALIZE = 3 # changer si besoin selon le prof
    print(f"\nGénération LaTeX pour les {TOP_K_VISUALIZE} meilleures règles")
    
    latex = [
        r"\documentclass[tikz,border=2pt,png]{standalone}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usetikzlibrary{arrows.meta, positioning, calc}",
        "",
        r"% --- Début du document ---",
        r"\begin{document}",
        r"\begin{tikzpicture}[",
        r"  % Style des noeuds 'Item' (Produit)",
        r"  item/.style={",
        r"    circle, fill=green!40, draw=green!80!black, thick,",
        r"    minimum size=1.5cm, align=center, font=\sffamily\scriptsize",
        r"  },",
        r"  % Style des noeuds 'Rule' (Règle)",
        r"  rule/.style={",
        r"    circle, fill=red!40, draw=red!80!black, thick,",
        r"    minimum size=0.8cm, align=center, font=\sffamily\bfseries\small",
        r"  },",
        r"  % Style des flèches",
        r"  every edge/.style={draw, -{Stealth[length=3mm, width=2mm]}}",
        r"]",
        ""
    ]

    # centres verticalement espacés
    regles_centers = [(0, 15), (0, 0), (0, -15)]
    radius = 6
    node_id_counter = 1

    top_regles = regles_df.head(TOP_K_VISUALIZE) # 3 premières règles

    for i, (idx, row) in enumerate(top_regles.iterrows()):
        if i >= 3: break
        
        regles_nom = f"R{i+1}"
        regles_id = f"r{i+1}"
        center_x, center_y = regles_centers[i]
        latex.append(f"% --- Règle {i+1} (Lift: {row['lift']:.2f}) ---")
        latex.append(
            f"\\node[rule] ({regles_id}) at ({center_x:.2f}, {center_y:.2f}) {{{regles_nom}}};"
        )
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        items = antecedents + consequents
        
        if not items:
            continue

        # NetworkX pour placer les items en cercle
        G_items = nx.Graph()
        G_items.add_nodes_from(items)
        # centre le layout circulaire sur la règle
        pos_items = nx.circular_layout(G_items, scale=radius, center=(center_x, center_y))
        item_node_ids = {} 

        for item in items:
            item_tikz_id = f"n{node_id_counter}"
            node_id_counter += 1
            item_node_ids[item] = item_tikz_id
            (x, y) = pos_items[item]
            # échappement des caractères pour LaTeX
            label = str(item).replace("-", "-\\\\").replace("_", "\\_").replace(" ", "\\\\")
            latex.append(
                f"\\node[item] ({item_tikz_id}) at ({x:.2f}, {y:.2f}) {{{label}}};"
            )

        # flèches antécédents -> règle en vert
        for ant in antecedents:
            ant_id = item_node_ids[ant]
            latex.append(
                f"\\draw[green!60!black, thick, ->] ({ant_id}) -- ({regles_id});"
            )

        # flèches règle -> conséquents en rouge
        for cons in consequents:
            cons_id = item_node_ids[cons]
            lift_label = f"{row['lift']:.2f}"
            latex.append(
                f"\\draw[red!80!black, thick, ->] ({regles_id}) -- "
                f"node[pos=0.6, above, sloped, font=\\tiny, fill=white, inner sep=1pt] {{{lift_label}}} "
                f"({cons_id});"
            )
        latex.append("") 

    latex.append(r"\end{tikzpicture}")
    latex.append(r"\end{document}")
    
    print("-" * 50)
    print("CODE LATEX (TIKZ) A COPIER DANS OVERLEAF :")
    print("-" * 50)
    print("\n".join(latex))
    print("-" * 50)


def analtyse_regles_association(df: pd.DataFrame, min_support=0.02, max_k=5, min_confidence=0.7):
    """Apriori + règles d'association"""
    transactions_liste = df["products_list_filtered"].tolist()
    print(f"{len(transactions_liste)} paniers pour Apriori")

    # encoder transactions en matrice SPARSE
    te = TransactionEncoder()
    try:
        # matrice sparse
        te_ary = te.fit(transactions_liste).transform(transactions_liste, sparse=True)
        df_encodee = pd.DataFrame(te_ary.toarray(), columns=te.columns_).astype(bool)
    except (TypeError, AttributeError):
        # fallback (pb de version)
        te_ary = te.fit(transactions_liste).transform(transactions_liste)
        df_encodee = pd.DataFrame(te_ary, columns=te.columns_)

    # appliquer Apriori
    print(f"Recherche itemsets avec support >= {min_support} et max_k = {max_k}")
    frequent_itemsets = apriori(df_encodee, min_support=min_support, use_colnames=True, max_len=max_k)
    
    if frequent_itemsets.empty:
        print(f"Aucun itemset avec un support >= {min_support}, (pt support plus bas)")
        print("-" * 50)
        return None

    regles = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    if regles.empty:
        print(f"Aucune règle trouvée avec une confiance >= {min_confidence}")
        print("-" * 50)
        return None

    # trier par lift
    regles_triees = regles.sort_values(by="lift", ascending=False)
    cols_affichees = ["antecedents", "consequents", "support", "confidence", "lift"]
    # print(regles_triees[cols_affichees].head(20).to_string(index=False)) #DEBUG
    print("-" * 50)

    # graphique des règles d'association
    top_regles = regles_triees.head(15).copy()
    if not top_regles.empty:
        # formattage (guillemets simples à l'intérieur)
        top_regles["rule"] = top_regles.apply(
            lambda row: f"{', '.join(row['antecedents'])} → {', '.join(row['consequents'])}",
            axis=1
        )
        # arrondir la confiance pour la légende
        top_regles["confidence"] = top_regles["confidence"].round(2)
        plt.figure(figsize=(14, 10))
        ax = sns.barplot(data=top_regles,  x="lift",  y="rule",  hue="confidence", palette="RdYlGn", orient="h", legend=True)
        ax.set_title("Top 5 des règles d'association (par Lift)", fontsize=16)
        ax.set_xlabel("Lift")
        ax.set_ylabel("Règle")
        plt.legend(title="Confiance", loc="lower right")
        plt.tight_layout()
        filepath_rules = os.path.join(OUTPUT_DIR, "5_regles_association.png")
        plt.savefig(filepath_rules, dpi=150, bbox_inches="tight")
        plt.close()

    # IMPORTANT retourne le dataframe pour l'utiliser ensuite
    return regles_triees

if __name__ == "__main__":
    df_clean = load_and_clean_data(FILEPATH)
    if df_clean is not None:
        df_filtree, all_items_valide = filtrer_et_extraire_all_items(df_clean)

        # tt les données pour les graphiques
        analyze_distributions(df_filtree)
        analyze_popular_items(all_items_valide)

        print("ANALYSE CLIENTS") 
        rfm_df = analyse_rfm_clients(df_filtree)
        analyse_fidelite_clients(df_filtree)
        analyse_evolution_clients(df_filtree)

        print("ANALYSE RÈGLES D'ASSOCIATION")
        # Filtrage pour les règles d'association
        print("\nFiltrage pour règles d'association")
        if not df_filtree.empty:
            max_date = df_filtree["date"].max()
            periode_avant = max_date - timedelta(days=30) # changer la période ici
            print(f"Période : {periode_avant} à {max_date}")

            df_periode = df_filtree[df_filtree["date"] >= periode_avant].copy()
            print(f"{df_periode.shape[0]} paniers pour MLxtend")

            # support 2%, k=3, confiance 0.8
            df_regles = analtyse_regles_association(
                df_periode,
                min_support=0.02,
                max_k=3,
                min_confidence=0.8
            )
            
            # APPEL DU LATEX SI RÈGLES TROUVÉES
            if df_regles is not None and not df_regles.empty:
                generee_latex_regles(df_regles)

        print("\nFIN")
    else:
        print("impossible de charger / nettoyer les données", file=sys.stderr)
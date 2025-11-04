import csv
from collections import Counter
import matplotlib.pyplot as plt

with open('dataset_baskets_dated.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    data = list(reader)

print(f"Total lignes: {len(data)}")
print(f"Colonnes: {list(data[0].keys())}\n")

print("=== APERÇU ===")
for i, row in enumerate(data[:3]):
    print(f"\nLigne {i+1}: {row}\n")

print("=== STATISTIQUES ===")
for column in data[0].keys():
    values = [row[column] for row in data if row[column]]
    unique = len(set(values))
    print(f"\n{column}:")
    print(f"  Uniques: {unique}")
    print(f"  Non vides: {len(values)}")
    print(f"  Vides: {len(data) - len(values)}")

    # graphique distribution
    if unique <= 10:
        counter = Counter(values)
        for value, count in counter.most_common(5):
            print(f"    {value}: {count}")
    
    # Créer un graphique pour les top 10 valeurs (si plus de 0)
    counter = Counter(values)
    top_values = counter.most_common(20)
    if top_values:
        labels, counts = zip(*top_values)
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(labels)), counts)
        plt.xlabel('Valeurs')
        plt.ylabel('Fréquence')
        plt.title(f'Distribution de {column} (Top 10)')
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'distribution_{column}.png')
        plt.close()
        print(f"  ✓ Graphique sauvegardé: distribution_{column}.png")

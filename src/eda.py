from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Dummy Data HSS.csv"
OUTPUT_DIR = ROOT / "reports" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# -------------------------
# 1. Distribution des ventes
# -------------------------
plt.figure()
sns.histplot(df["Sales"], bins=30, kde=True)
plt.title("Distribution des ventes")
plt.savefig(OUTPUT_DIR / "sales_distribution.png")
plt.close()

# -------------------------
# 2. Distribution des budgets
# -------------------------
for col in ["TV", "Radio", "Social Media"]:
    plt.figure()
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f"Distribution {col}")
    plt.savefig(OUTPUT_DIR / f"{col}_distribution.png")
    plt.close()

# -------------------------
# 3. Relation budget vs ventes
# -------------------------
for col in ["TV", "Radio", "Social Media"]:
    plt.figure()
    sns.scatterplot(x=df[col], y=df["Sales"])
    plt.title(f"{col} vs Sales")
    plt.savefig(OUTPUT_DIR / f"{col}_vs_sales.png")
    plt.close()

# -------------------------
# 4. Corrélation
# -------------------------
plt.figure(figsize=(6,5))
corr = df[["TV", "Radio", "Social Media", "Sales"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Matrice de corrélation")
plt.savefig(OUTPUT_DIR / "correlation_matrix.png")
plt.close()

# -------------------------
# 5. Influenceur vs ventes
# -------------------------
plt.figure()
sns.boxplot(x=df["Influencer"], y=df["Sales"])
plt.title("Ventes selon l'influenceur")
plt.savefig(OUTPUT_DIR / "sales_by_influencer.png")
plt.close()

# -------------------------
# 6. Outliers
# -------------------------
plt.figure()
sns.boxplot(data=df[["TV", "Radio", "Social Media", "Sales"]])
plt.title("Détection des outliers")
plt.savefig(OUTPUT_DIR / "outliers.png")
plt.close()

print("EDA terminée. Graphiques générés dans reports/eda/")
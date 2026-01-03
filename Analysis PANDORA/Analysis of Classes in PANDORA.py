import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path_profiles = r"../Files/Top70%oceanprofiles.csv"

df = pd.read_csv(path_profiles)

cat_cols_raw = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism"
]

for col in cat_cols_raw:
    df[col + "_cat"] = pd.cut(
        df[col],
        bins=[0, 33, 66, np.inf],
        labels=["niedrig", "mittel", "hoch"]
    )

cat_cols = [
    "openness_cat",
    "conscientiousness_cat",
    "extraversion_cat",
    "agreeableness_cat",
    "neuroticism_cat"
]

df = df.dropna(subset=cat_cols).reset_index(drop=True)
print("Nutzer mit vollständigem OCEAN Profil:", len(df))

verteilung_pro_dim = {}

for dim in cat_cols_raw:
    counts = df[dim + "_cat"].value_counts(normalize=True)
    verteilung_pro_dim[dim] = counts

verteilung_pro_dim_df = pd.DataFrame(verteilung_pro_dim).T
verteilung_pro_dim_df = verteilung_pro_dim_df[["niedrig", "mittel", "hoch"]]


alle_kategorien = pd.concat([df[i] for i in cat_cols], axis=0)

gesamt_verteilung = alle_kategorien.value_counts(normalize=True).sort_index()

gesamt_verteilung_df = gesamt_verteilung.to_frame(name="Anteil")

verteilung_pro_dim_df.loc["Gesamt"] = gesamt_verteilung_df["Anteil"]

print(gesamt_verteilung_df)
print(verteilung_pro_dim_df)

fig, ax = plt.subplots(figsize=(19, 10))

ax.axis("off")

verteilung_pro_dim_df = verteilung_pro_dim_df.apply(lambda col: col.apply(lambda x: f"{x:.2%}"))

table = ax.table(
    cellText=verteilung_pro_dim_df[["niedrig", "mittel", "hoch"]],
    cellLoc="center",
    loc="upper left",
)

table.auto_set_font_size(False)
table.set_fontsize(15)

table.scale(1, 7)

for (reihe, col), zelle in table.get_celld().items():
    if reihe == 0:
        zelle.set_text_props(weight="bold")
    if reihe == len(verteilung_pro_dim_df):
        zelle.set_text_props(weight="bold")

plt.title("Verteilung der OCEAN Ausprägungen pro Dimension und insgesamt", pad=20, fontsize=20, fontweight="bold")

plt.savefig(
    "verteilung_ocean_tabelle.png"
)

plt.close()
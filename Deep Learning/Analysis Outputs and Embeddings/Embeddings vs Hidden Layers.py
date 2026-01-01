from tensorflow.keras.models import load_model, Model
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from matplotlib.colors import ListedColormap

path_profiles = r"../../Files/Top70%oceanprofiles.csv"
path_comments = r"../../Files/commentsfromoceanpeople.csv"

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

comments = pd.read_csv(path_comments)
print(type(comments))
print(comments.columns)

comments["body"] = comments["body"].fillna("").astype(str)
print("Geladene Kommentare:", len(comments))

tqdm.pandas()

length_comments = []

def collect_texts(group, menge=20):
    texts = group["body"].sample(min(len(group), menge)).tolist()
    length_comments.append(min(len(group), menge))
    return " ".join(texts)

user_texts = comments.groupby("author").progress_apply(collect_texts)
print(sum(length_comments)/len(length_comments))
user_texts = user_texts.reset_index()
user_texts.columns = ["author", "text"]

df_users = df.merge(user_texts, on="author", how="inner")
print("Finale Anzahl Nutzer:", len(df_users))

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda")

def embed(text):
    return model.encode(text, normalize_embeddings=True)


embeddings = np.vstack(df_users["text"].progress_apply(embed))

pca2 = PCA(n_components=2, random_state=42)

model = load_model("../../Trained Models/model_version1_qwen+mlp.keras")

geteilter_relu = Model(inputs=model.input, outputs=model.get_layer("geteilter_relu").output)
hidden_geteilt = geteilter_relu.predict(embeddings)

pca_test = {
    "embeddings": pca2.fit_transform(embeddings).astype(np.float64),
    "shared": pca2.fit_transform(hidden_geteilt).astype(np.float64)
}

for c in cat_cols_raw:
    einzelner_relu = Model(inputs=model.input, outputs=model.get_layer(f"{c}_relu").output)
    einz_relu_test = einzelner_relu.predict(embeddings)
    pca_test[c] = pca2.fit_transform(einz_relu_test).astype(np.float64)

def plot_von_pca(ax, pca_daten, df_nutzer, categorie_col, titel):
    momentane_col = df_nutzer[categorie_col]
    momentane_codes = momentane_col.cat.codes.values
    labels = list(momentane_col.cat.categories)
    colors = ["red", "green", "blue"]

    for code, label, color in zip([0,1,2], labels, colors):
        mask = (momentane_codes == code)
        ax.scatter(
            pca_daten[mask, 0],
            pca_daten[mask, 1],
            s=20,
            alpha=0.7,
            color=color,
            label=label
        )

    ax.set_title(titel)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend()

def kmeans_and_plot(ax, pca_daten, categorie_col):
    x_min, x_max = pca_daten[:, 0].min() - 0.05, pca_daten[:, 0].max() + 0.05
    y_min, y_max = pca_daten[:, 1].min() - 0.05, pca_daten[:, 1].max() + 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    kmmeans = KMeans(n_clusters=3, random_state=42).fit(pca_daten)

    momentane_col = df_users[categorie_col]
    ids_cluster = kmmeans.predict(pca_daten)
    tabelle = pd.crosstab(ids_cluster, momentane_col)
    in_prozent = tabelle.div(tabelle.sum(axis=1), axis=0)
    print(in_prozent)

    Z_kmeans = kmmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_kmeans = Z_kmeans.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

    zentren = kmmeans.cluster_centers_

    ax.scatter(
        zentren[:, 0],
        zentren[:, 1],
        s=250,
        marker="X",
        color="black",
        label="KMeans Zentren"
    )

    ax.contourf(
        xx, yy, Z_kmeans, cmap=cmap_light, alpha=0.3
    )

    for b in range(3):
        punkte_im_clust = pca_daten[ids_cluster == b]

        x = punkte_im_clust[:, 0].mean()
        y = punkte_im_clust[:, 1].mean()

        text = (
            f"Cluster {b+1}\n"
            f"niedrig: {in_prozent.loc[b, 'niedrig']:.0%}\n"
            f"mittel: {in_prozent.loc[b, 'mittel']:.0%}\n"
            f"hoch: {in_prozent.loc[b, 'hoch']:.0%}\n"
        )
        ax.text(
            x,
            y,
            text,
            ha="center", #Text im Vergleich zum Punkt
            va="center",
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.75)
        )
    return in_prozent

max_prozent = []

for cat_col, raw_col in zip(cat_cols, cat_cols_raw):

    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(131)
    plot_von_pca(ax1, pca_test["embeddings"], df_users, cat_col, "Embeddings")
    in_prozent_embeddings = kmeans_and_plot(ax1, pca_test["embeddings"], cat_col)

    ax2 = fig.add_subplot(132)
    plot_von_pca(ax2, pca_test["shared"], df_users, cat_col, "Shared Hidden")
    in_prozent_shared = kmeans_and_plot(ax2, pca_test["shared"], cat_col)

    ax3 = fig.add_subplot(133)
    plot_von_pca(ax3, pca_test[raw_col], df_users, cat_col, f"{raw_col} Hidden")
    in_prozent_Hidden = kmeans_and_plot(ax3, pca_test[raw_col], cat_col)

    for label in ["niedrig", "mittel", "hoch"]:
        max_prozent.append({
            "Dimension": raw_col,
            "Repräsentation": "Embeddings",
            "Label": label,
            "Max_Anteil": in_prozent_embeddings[label].max()
        })
        max_prozent.append({
            "Dimension": raw_col,
            "Repräsentation": "Shared",
            "Label": label,
            "Max_Anteil": in_prozent_shared[label].max()
        })
        max_prozent.append({
            "Dimension": raw_col,
            "Repräsentation": f"{raw_col} Hidden",
            "Label": label,
            "Max_Anteil": in_prozent_Hidden[label].max()
        })

    plt.tight_layout()
    plt.show()

max_df = pd.DataFrame(max_prozent)
max_df.sort_values("Max_Anteil", ascending=False, inplace=True)

neues_df = max_df.groupby(["Dimension" , "Repräsentation"]).agg(Mittelwert=("Max_Anteil", "mean"), Maximum=("Max_Anteil", "max")).reset_index()
emb_ref = neues_df[neues_df["Repräsentation"] == "Embeddings"].set_index("Dimension")["Mittelwert"]

neues_df["Differenz zu Embeddings"] = neues_df["Mittelwert"] - neues_df["Dimension"].map(emb_ref)

neues_df["Mittelwert"] = neues_df["Mittelwert"].apply(lambda x: f"{x:.2%}")
neues_df["Maximum"] = neues_df["Maximum"].apply(lambda x: f"{x:.2%}")
neues_df["Differenz zu Embeddings"] = neues_df["Differenz zu Embeddings"].apply(lambda x: f"{x:+.2%}")

neues_df.sort_values("Mittelwert", ascending=False, inplace=True)

max_df["Max_Anteil"] = max_df["Max_Anteil"].apply(lambda x: f"{x:.2%}")
print(max_df)

fig, ax = plt.subplots(figsize=(19, 18), dpi=600)

ax.axis("off")


table = ax.table(
    cellText=max_df.values,
    colLabels=max_df.columns,
    loc='center',
    cellLoc='center',
)

table.scale(1.3, 1.8)
table.auto_set_font_size(False)
table.set_fontsize(10)

for (reihe, col), zelle in table.get_celld().items():
    if reihe == 0:
        zelle.set_text_props(weight="bold")
        zelle.set_facecolor("white")
        continue
    repräsentation = max_df.iloc[reihe-1]["Repräsentation"]
    if repräsentation.endswith("Hidden"):
        zelle.set_facecolor("#98F5FF")
    elif repräsentation == "Embeddings":
        zelle.set_facecolor("#FF7F24")
    elif repräsentation == "Shared":
        zelle.set_facecolor("#919191")

plt.title('Höchster Cluster Anteil pro Dimension, Repräsentation und Label', pad=20, fontsize=20, fontweight='bold')
plt.subplots_adjust(top=0.95)

plt.savefig(
    "tabelle_cluster_unaggregiert.png",
    dpi=1800,
    bbox_inches="tight",
)
plt.close()

fig, ax = plt.subplots(figsize=(14, 5), dpi=600)

ax.axis("off")


table = ax.table(
    cellText=neues_df.values,
    colLabels=neues_df.columns,
    loc='center',
    cellLoc='center',
)

table.scale(1, 1.5)
table.auto_set_font_size(False)
table.set_fontsize(10)

for (reihe, col), zelle in table.get_celld().items():
    if reihe == 0: #da es bei der kopfzeile anfängt
        zelle.set_text_props(weight="bold")
        zelle.set_facecolor("white")
        continue
    repräsentation = neues_df.iloc[reihe-1]["Repräsentation"]
    if repräsentation.endswith("Hidden"):
        zelle.set_facecolor("#98F5FF")
    elif repräsentation == "Embeddings":
        zelle.set_facecolor("#FF7F24")
    elif repräsentation == "Shared":
        zelle.set_facecolor("#919191")

plt.title('Aggregierte Form der Clusterinformationen über Dimension und Repräsentation', pad=20, fontweight='bold')

plt.tight_layout()
plt.savefig(
    "tabelle_cluster_aggregiert.png",
    dpi=1200,
    bbox_inches="tight",
)
plt.close()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

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

def collect_texts(group, menge=10):
    texts = group["body"].sample(n=min(len(group), menge), random_state=42).tolist()
    return " ".join(texts)

user_texts = comments.groupby("author").progress_apply(collect_texts)
user_texts = user_texts.reset_index()
user_texts.columns = ["author", "text"]

df_users = df.merge(user_texts, on="author", how="inner")

print("Finale Anzahl Nutzer:", len(df_users))

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda")

def embed(text):
    return model.encode(text, normalize_embeddings=True)

embeddings = np.vstack(df_users["text"].progress_apply(embed))

pca3 = PCA(n_components=3, random_state=42)
emb_pca3 = pca3.fit_transform(embeddings)


colors = ["tab:blue", "tab:orange", "tab:green"]

for cat_col, raw_col in zip(cat_cols, cat_cols_raw):
    momentane_col = df_users[cat_col]
    momentane_codes = momentane_col.cat.codes.values
    labels = list(momentane_col.cat.categories)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    for code, label, color in zip([0, 1, 2], labels, colors):
        mask = momentane_codes == code
        ax.scatter(
            emb_pca3[mask, 0],
            emb_pca3[mask, 1],
            emb_pca3[mask, 2],
            s=20,
            alpha=0.7,
            color=color,
            label=label
        )

    ax.set_xlabel("PCA Komponente 1")
    ax.set_ylabel("PCA Komponente 2")
    ax.set_zlabel("PCA Komponente 3")
    ax.set_title(f"3D PCA Embedding aller Nutzer gefärbt nach {raw_col}")

    plt.legend(title=raw_col)
    plt.show()


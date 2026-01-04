import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import torch
import joblib
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

BATCH_SIZE = 32

path = r"../Files/commentsfromoceanpeople.csv"
df = pd.read_csv(path)

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda")
model.max_seq_length = 256

def embed_mit_fortschritt(texts, batch_size=BATCH_SIZE):
    return model.encode(texts, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=True)

def konsistenz(emb):
    variance = np.mean(np.var(emb, axis=0))
    pairwise = np.mean(pairwise_distances(emb))
    return {"variance": variance, "pairwise": pairwise}

resultate = {}

for user, group in tqdm(df.groupby("author"), desc="Users"):
    texts = group["body"].sample(n=40, replace= True).tolist()
    emb = embed_mit_fortschritt(texts, batch_size=BATCH_SIZE)
    resultate[user] = konsistenz(emb)
    torch.cuda.empty_cache()

res_df = pd.DataFrame.from_dict(resultate, orient="index")
res_df.index.name = "author"

print(res_df.head())
print("Totale Nutzer:", len(res_df))


x = res_df["pairwise"]
y = res_df["variance"]

plt.scatter(x, y)
plt.xlabel("Durch. Distanz Paare")
plt.ylabel("Durch. Varianz")
plt.title("Varianz zu Distanz")
plt.show()


X = res_df[["pairwise", "variance"]]

kmeans = KMeans(n_clusters=2 , random_state=42)
labels = kmeans.fit_predict(X)

cut_pair = res_df["pairwise"].quantile(0.045)
cut_var = res_df["variance"].quantile(0.055)

plt.scatter(res_df["pairwise"], res_df["variance"], c=labels)
plt.xlabel("Durch. Distanz Paare")
plt.ylabel("Durch. Varianz")
plt.axvline(cut_pair, linestyle="--", color="red")
plt.axhline(cut_var, linestyle="--")
plt.title("KMeans Clustering 2 Cluster")
plt.show()

res_df["cluster"] = labels

#joblib.dump(res_df, "author_embedding_konsistenz.joblib")
cut_pair = res_df["pairwise"].quantile(0.70)
cut_var = res_df["variance"].quantile(0.70)
oceandf = pd.read_csv(r"C:\Users\mauri_9qhl4wd\OneDrive - SekII Zürich\Maturarbeit\PANDORA\oceanprofilesfile.csv")
dfoceannew = res_df[(res_df["pairwise"] < cut_pair) & (res_df["variance"] < cut_var)]
df_all_selected = oceandf[oceandf["author"].isin(dfoceannew.index)]
df_all_selected.to_csv(r"C:\Users\mauri_9qhl4wd\OneDrive - SekII Zürich\Maturarbeit\Top70%oceanprofiles.csv")
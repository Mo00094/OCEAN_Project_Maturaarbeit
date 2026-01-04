from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import normalize
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Activation
from tensorflow.keras.models import Model
from pathlib import Path
from time import strftime
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

#---------------------------------
#Spliting into the Sets
#---------------------------------

path1 = r"C:\Users\mauri_9qhl4wd\OneDrive - SekII Zürich\Maturarbeit\Top70%oceanprofiles.csv"

path2 = r"C:\Users\mauri_9qhl4wd\OneDrive - SekII Zürich\Maturarbeit\PANDORA\commentsfromoceanpeople.csv"

df = pd.read_csv(path1)

for i, col in enumerate(df.columns[14:19]):
    df[str(col) + "_cat"] = pd.cut(df[col], bins=[0, 33, 66, np.inf], labels=["niedrig", "mittel", "hoch"])

cat_cols = [
    "openness_cat",
    "conscientiousness_cat",
    "extraversion_cat",
    "agreeableness_cat",
    "neuroticism_cat"
]

df = df.dropna(subset=cat_cols)


df["Gesamt_strat_OCEAN"] = (df["openness_cat"].astype(str) + "_" + df["conscientiousness_cat"].astype(str) + "_" + df["extraversion_cat"].astype(str) + "_" + df["agreeableness_cat"].astype(str) + "_" + df["neuroticism_cat"].astype(str))


counts = df["Gesamt_strat_OCEAN"].value_counts()

richtige_kombinationen = counts[counts >= 2].index

df = df[df["Gesamt_strat_OCEAN"].isin(richtige_kombinationen)]

df = df.reset_index(drop=True)

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
stratifiezierte_splits = []
for trainings_indexe, test_indexe in splitter.split(df, df["Gesamt_strat_OCEAN"]):
    train_set_n = df.loc[trainings_indexe]
    temp_set = df.loc[test_indexe]

    val_set, test_set = train_test_split(
    temp_set, test_size=0.5, random_state=43
    )

    stratifiezierte_splits.append([train_set_n, val_set, test_set])

df1 = pd.read_csv(path2)

df2 = df.assign(body = df1['body'].fillna('').astype(str))
texte_pro_user = df2.groupby("author")["body"].apply(' '.join).reset_index()

all_splits = []

for i, splits in enumerate(stratifiezierte_splits):
    train_split = splits[0].set_index("author")
    test_split = splits[2].set_index("author")
    vald_split = splits[1].set_index("author")

    x_train = texte_pro_user.set_index("author").loc[train_split.index, "body"]
    x_vald = texte_pro_user.set_index("author").loc[vald_split.index, "body"]
    x_test = texte_pro_user.set_index("author").loc[test_split.index, "body"]

    y_cols = ["openness_cat", "conscientiousness_cat", "extraversion_cat", "agreeableness_cat", "neuroticism_cat"]
    y_train = train_split.loc[:, y_cols]
    y_vald = vald_split.loc[:, y_cols]
    y_test = test_split.loc[:, y_cols]

    split_data = {
        "x_train": x_train,
        "x_vald": x_vald,
        "x_test": x_test,
        "y_train": y_train,
        "y_vald": y_vald,
        "y_test": y_test,
    }

    all_splits.append(split_data)


#-----------------------------
#Qwen3 + MLP
#-----------------------------

vectorizer_qwen= SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    trust_remote_code=True,
    device="cuda"
)

def embeded_texte_pro_user(text):
    return vectorizer_qwen.encode(text, normalize_embeddings=True, device="cuda")

tqdm.pandas()

all_splits_vectorized_training = []

for split in all_splits:
    x_train_vec = np.vstack(split["x_train"].progress_apply(embeded_texte_pro_user))
    x_vald_vec = np.vstack(split["x_vald"].progress_apply(embeded_texte_pro_user))
    x_test_vec = np.vstack(split["x_test"].progress_apply(embeded_texte_pro_user))

    all_splits_vectorized_training.append({
        "x_train_vec": x_train_vec,
        "x_vald_vec": x_vald_vec,
        "x_test_vec": x_test_vec
    })


def get_run_logdir():
    base_logdir = Path(r"C:\TensorBoardLogs")
    base_logdir.mkdir(parents=True, exist_ok=True)
    return base_logdir / strftime("run_%Y_%m_%d_%H_%M_%S")

run_logdir = get_run_logdir()

tensorboard_cb = tf.keras.callbacks.TensorBoard(
    run_logdir,
    profile_batch=(100, 200),
    histogram_freq=1
)


x_train_vec = all_splits_vectorized_training[0]["x_train_vec"]
x_vald_vec = all_splits_vectorized_training[0]["x_vald_vec"]
x_test_vec = all_splits_vectorized_training[0]["x_test_vec"]


traits = [
    "openness_cat",
    "conscientiousness_cat",
    "extraversion_cat",
    "agreeableness_cat",
    "neuroticism_cat"
]

y_train_list = [all_splits[0]["y_train"][t].cat.codes.values for t in traits]
y_vald_list = [all_splits[0]["y_vald"][t].cat.codes.values for t in traits]
y_test_list = [all_splits[0]["y_test"][t].cat.codes.values for t in traits]


inputs = Input(shape=(x_train_vec.shape[1],), name="input")

geteilter_dense = Dense(256, name="geteilter_dense")(inputs)
geteilter_relu = Activation("relu", name="geteilter_relu")(geteilter_dense)
geteilter_dropout = Dropout(0.4, name="geteilter_dropout")(geteilter_relu)


traits_without_cat = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism"
]


outputs = []

for t in traits_without_cat:
    dense1 = Dense(128, name=f"{t}_dense_1")(geteilter_relu)
    relu1 = Activation("relu", name=f"{t}_relu_1")(dense1)
    logits = Dense(3, name=f"{t}_logits")(relu1)
    softmax = Activation("softmax", name=f"{t}_softmax")(logits)

    outputs.append(softmax)



model = Model(inputs=inputs, outputs=outputs)

Precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics= ["accuracy"] * 5)

history = model.fit(
    x_train_vec, y_train_list,
    epochs=40,
    validation_data=(x_vald_vec, y_vald_list),
    callbacks=[tensorboard_cb],
    batch_size=32
)

y_proba = model.predict(x_test_vec)
y_pred = [p.argmax(axis=1) for p in y_proba]

for i, trait in enumerate(traits_without_cat):
    y_true = y_test_list[i]
    acc = accuracy_score(y_true, y_pred[i])
    f1 = f1_score(y_true, y_pred[i], average="macro")

    loss_training = history.history[f"{trait}_softmax_loss"]
    loss_validation = history.history[f"val_{trait}_softmax_loss"]

    genau_training = history.history[f"{trait}_softmax_accuracy"]
    genau_validation = history.history[f"val_{trait}_softmax_accuracy"]

    epochen = range(1, len(loss_training) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    fig.suptitle(trait, fontsize=20, fontweight="bold")

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred[i],
        normalize="true",
        values_format=".0%",
        cmap="Blues",
        text_kw={"color": "black", "fontsize": 10},
        ax=axes[0]
    )

    axes[0].set_title("Confusion Matrix", fontsize=15, pad=15)

    axes[0].text(0.5, -0.13, f"Acc = {acc:.2f} F1 = {f1:.2f}",
                 transform=axes[0].transAxes, fontsize=11, ha="center")

    axes[1].plot(epochen, loss_training, label="Training Loss")
    axes[1].plot(epochen, loss_validation, label="Validation Loss")
    axes[1].set_xlabel("Epoche")
    axes[1].grid(True)
    axes[1].legend(loc="upper left")
    axes[1].set_title("Loss", fontsize=15, pad=15)

    axes[2].plot(epochen, genau_training, label="Training Genauigkeit")
    axes[2].plot(epochen, genau_validation, label="Validation Genauigkeit")
    axes[2].set_xlabel("Epoche")
    axes[2].grid(True)
    axes[2].legend(loc="upper left")
    axes[2].set_title("Genauigkeit", fontsize=15, pad=15)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    fig.show()

print(history.history.keys())


model.save("../Trained Models/model_version1_qwen+mlp.keras")



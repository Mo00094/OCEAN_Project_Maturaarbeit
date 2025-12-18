import joblib
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, make_scorer, ConfusionMatrixDisplay, confusion_matrix
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split #weil train_test_split nicht in der root von sklearn ist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


modele = joblib.load(r"alle_modelle_for_accuracy_score_para_version3_tf_idf_iter_50.joblib")

path2 = "../Files/oceanprofilesfile.csv"

df = pd.read_csv(path2)

for i, col in enumerate(df.columns[13:18]):
    df[str(col) + "_cat"] = pd.cut(df[col], bins=[0, 33, 66, np.inf], labels=["niedrig", "mittel", "hoch"]) #Speichert ein Categorical-Object in jeweils einer neuen Spalte für alle OCEAN-Werte, Gibt hier dann 5 Bins, die die Werte für OCEAN unterteilen in die passenden Kateogrien
cat_cols = [
    "openness_cat",
    "conscientiousness_cat",
    "extraversion_cat",
    "agreeableness_cat",
    "neuroticism_cat"
]

df = df.dropna(subset=cat_cols)

print(df.columns)
print(df.head())

df["Gesamt_strat_OCEAN"] = (df["openness_cat"].astype(str) + "_" + df["conscientiousness_cat"].astype(str) + "_" + df["extraversion_cat"].astype(str) + "_" + df["agreeableness_cat"].astype(str) + "_" + df["neuroticism_cat"].astype(str))


counts = df["Gesamt_strat_OCEAN"].value_counts() #Gibt eine Series zürck für alle Werte gezählt in counts

richtige_kombinationen = counts[counts >= 2].index #Nimmt nur den Index, also z.B. 1-4-3-4-5 aus der Series counts

df = df[df["Gesamt_strat_OCEAN"].isin(richtige_kombinationen)]

df = df.reset_index(drop=True) #Index neu berechnen, weil Punkte gelöscht wurden

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) #Spliiter Objekt wird vorbereitet mit jeweils 10 Splits, Test-Grösse von 20% und dem random_state vom 42, damit immer gleiche Test/Trainingssets generiet werden.
stratifiezierte_splits = [] #Liste für die Paare aus Test- und Trainingsdaten DataFrames
for trainings_indexe, test_indexe in splitter.split(df, df["Gesamt_strat_OCEAN"]): #Splitter Objekt hat die Methode split(), um Verteilungen zu berücksichtigen
    strat_train_set_n = df.loc[trainings_indexe]
    strat_test_set_n = df.loc[test_indexe]
    stratifiezierte_splits.append([strat_train_set_n, strat_test_set_n]) #Liste bekommt 10 Paare von Training und Test

print(df["Gesamt_strat_OCEAN"].value_counts())
print(df.head())

path1 = "../Files/commentsfromoceanpeople.csv"

df = pd.read_csv(path1)

df2 = df.assign(body = df['body'].fillna('').astype(str))
texte_pro_user = df2.groupby("author")["body"].apply(' '.join).reset_index()


dimensionen = ["openness_cat", "conscientiousness_cat", "extraversion_cat", "agreeableness_cat", "neuroticism_cat"]

test_split_y = stratifiezierte_splits[0][1].set_index('author')
train_split_y = stratifiezierte_splits[0][0].set_index('author')

x_test = texte_pro_user.set_index("author").loc[test_split_y.index, "body"]
x_train = texte_pro_user.set_index("author").loc[train_split_y.index, "body"]

traits = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism"
]

resultate = {}
LABELS = ["hoch", "mittel", "niedrig"]

for i, (trait_cat, trait) in enumerate(zip(dimensionen, traits)):
    y_true = test_split_y[trait_cat]
    y_pred = modele[trait_cat].predict(x_test)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        normalize="true",
        values_format=".0%",
        cmap="Blues",
        text_kw={"color": "black", "fontsize": 10}
    )
    plt.title(f"{trait}\nAcc={acc:.2f}, F1={f1:.2f}")
    plt.show()
    y_pred_tuned = modele[trait_cat].predict(x_test)
    cm_tuned = confusion_matrix(y_true, y_pred_tuned, labels=LABELS)

    resultate[trait] = {"trait": trait, "acc": acc, "f1": f1, "confusion_matrix": cm_tuned, "y_pred_tuned": y_pred_tuned, "y_true": y_true}

joblib.dump(resultate, r"Alle Auswertungen Modelle Naive Bayes/alle_auswertungen_for_accuracy_para_version3_tf_idf_iter_50.joblib")

print("\nTRAINING RESULTATE")
for i, (trait_cat, trait) in enumerate(zip(dimensionen, traits)):
    y_train_true = train_split_y[trait_cat]
    y_train_pred = modele[trait_cat].predict(x_train)

    train_acc = accuracy_score(y_train_true, y_train_pred)
    train_f1 = f1_score(y_train_true, y_train_pred, average="macro")

    print(f"{trait}: Acc={train_acc:.3f}, F1={train_f1:.3f}")


#modelle = joblib.load("alle_modelle_for_f1_score_para_version2_iter_18.joblib")
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, ConfusionMatrixDisplay, confusion_matrix
import joblib
path1 = "../Files/oceanprofilesfile.csv"

df = pd.read_csv(path1)

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

df["Gesamt_strat_OCEAN"] = (df["openness_cat"].astype(str) + "_" + df["conscientiousness_cat"].astype(str) + "_" + df["extraversion_cat"].astype(str) + "_" + df["agreeableness_cat"].astype(str) + "_" + df["neuroticism_cat"].astype(str))


counts = df["Gesamt_strat_OCEAN"].value_counts() #Gibt eine Series zürck für alle Werte gezählt in counts

richtige_kombinationen = counts[counts >= 2].index #Nimmt nur den Index, also z.B. 1-4-3-4-5 aus der Series counts

df = df[df["Gesamt_strat_OCEAN"].isin(richtige_kombinationen)]

df = df.reset_index(drop=True) #Index neu berechnen, weil Punkte gelöscht wurden

df = df.add_prefix("profile_")

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) #Spliiter Objekt wird vorbereitet mit jeweils 10 Splits, Test-Grösse von 20% und dem random_state vom 42, damit immer gleiche Test/Trainingssets generiet werden.
stratifiezierte_splits = [] #Liste für die Paare aus Test- und Trainingsdaten DataFrames
for trainings_indexe, test_indexe in splitter.split(df, df["profile_Gesamt_strat_OCEAN"]): #Splitter Objekt hat die Methode split(), um Verteilungen zu berücksichtigen
    strat_train_set_n = df.loc[trainings_indexe]
    strat_test_set_n = df.loc[test_indexe]
    stratifiezierte_splits.append([strat_train_set_n, strat_test_set_n]) #Liste bekommt 10 Paare von Training und Test

path1 = "../Files/commentsfromoceanpeople.csv"

df = pd.read_csv(path1)

df2 = df.assign(body = df['body'].fillna('').astype(str))
texte_pro_user = df2.groupby("author")["body"].apply(' '.join).reset_index()
texte_pro_user.index = texte_pro_user["author"]
texte_pro_user.drop(["author"], axis=1, inplace=True)

profile_users = set(df["author"])  # from profiles
text_users = set(texte_pro_user.index)

missing_users = profile_users - text_users
print("Users in OCEAN profiles but missing text:", len(missing_users))
print(list(missing_users)[:20])

print(texte_pro_user.columns)
vectorizer_count = CountVectorizer()

vectorizer_tfid = TfidfVectorizer(
    max_features=650000,
    ngram_range=(1,2),
    min_df=10,
    max_df=0.6
)

X = vectorizer_tfid.fit_transform(texte_pro_user["body"])

X_df = pd.DataFrame.sparse.from_spmatrix(X, columns=vectorizer_tfid.get_feature_names_out(), index=texte_pro_user.index)

del X

merged_trainings_daten = stratifiezierte_splits[0][0].merge(X_df, left_on="profile_author", right_index=True)
merged_test_daten = stratifiezierte_splits[0][1].merge(X_df, left_on="profile_author", right_index=True)

del df, df2, texte_pro_user, stratifiezierte_splits

merged_trainings_daten = merged_trainings_daten.reindex(columns = merged_test_daten.columns,fill_value = 0)

merged_test_daten = merged_test_daten.reindex(columns = merged_trainings_daten.columns,fill_value = 0)

dimensionen = ["profile_openness_cat", "profile_conscientiousness_cat", "profile_extraversion_cat", "profile_agreeableness_cat", "profile_neuroticism_cat"]
modele = {}
genauigkeiten = {}

print(merged_trainings_daten.head())
print(merged_test_daten.head())

feature_cols = vectorizer_tfid.get_feature_names_out().tolist()

print(len(feature_cols))
print(len(merged_test_daten.columns))
missing_features = [c for c in feature_cols if c not in merged_test_daten.columns]
print(len(missing_features))
print(missing_features[:20])
print(feature_cols[:50])


for dimension in dimensionen:
    y_train = merged_trainings_daten[dimension]

    model = MultinomialNB()
    model.fit(merged_trainings_daten[feature_cols], y_train)
    y_pred = model.predict(merged_trainings_daten[feature_cols])

    genauigkeit = accuracy_score(y_train, y_pred)
    modele[dimension] = model
    genauigkeiten[dimension] = genauigkeit

print("Genauigkeit pro Dimension Training", genauigkeiten)

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
    y_true = merged_test_daten[trait_cat]
    y_pred = modele[trait_cat].predict(merged_test_daten[feature_cols])
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
    y_pred_base = modele[trait_cat].predict(merged_test_daten[feature_cols])
    cm_base = confusion_matrix(y_true, y_pred_base, labels=LABELS)

    resultate[trait] = {"trait": trait, "acc": acc, "f1": f1, "confusion_matrix": cm_base, "y_pred_base": y_pred_base, "y_true": y_true}

joblib.dump(resultate, f"base_tfidf.joblib")
joblib.dump(modele, f"modelle_tfidf_base.joblib")
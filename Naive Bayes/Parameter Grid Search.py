from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, make_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split #weil train_test_split nicht in der root von sklearn ist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

path2 = "../Files/commentsfromoceanpeople.csv"

df = pd.read_csv(path2)

df2 = df.assign(body = df['body'].fillna('').astype(str))
texte_pro_user = df2.groupby("author")["body"].apply(' '.join).reset_index()


dimensionen = ["openness_cat", "conscientiousness_cat", "extraversion_cat", "agreeableness_cat", "neuroticism_cat"]

modele = {}
best_parameter = {}

train_split_y = stratifiezierte_splits[0][0].set_index('author')
test_split_y = stratifiezierte_splits[0][1].set_index('author')

x_train = texte_pro_user.set_index("author").loc[train_split_y.index, "body"]
x_test = texte_pro_user.set_index("author").loc[test_split_y.index, "body"]
LABELS = ["hoch", "mittel", "niedrig"]

def safe_macro_f1(y_true, y_pred):
    try:
        return f1_score(
            y_true,
            y_pred,
            average="macro",
            labels=LABELS,
            zero_division=0
        )
    except:
        return 0.0

def safe_accruacy(y_true, y_pred):
    try:
        return accuracy_score(y_true, y_pred)
    except:
        return 0.0


bewerter1 = make_scorer(safe_macro_f1, greater_is_better=True)
bewerter2 = make_scorer(safe_accruacy, greater_is_better=True)
rows = []

for dimension in dimensionen:
    y_train = train_split_y[dimension]
    y_test = test_split_y[dimension]

    whole_pipeline = Pipeline([("vectorizer1", TfidfVectorizer()), ("multiNB", MultinomialNB(fit_prior=False))])

    verteilung_parameter_version1_tfid = {
        "vectorizer1__max_features": randint(100000, 900000),
        "vectorizer1__ngram_range": [(1, 1), (1, 2)],
        "vectorizer1__min_df": uniform(0.0005, 0.005),
        "vectorizer1__max_df": uniform(0.75, 0.2),
        "multiNB__alpha": uniform(0.01, 1.0),
        "multiNB__fit_prior": [True, False]
    }

    verteilung_parameter_version2_tfid = {
        "vectorizer1__max_features": randint(50000, 300000),
        "vectorizer1__ngram_range": [(1, 1), (1, 2)],
        "vectorizer1__min_df": uniform(0.0001, 0.001),
        "vectorizer1__max_df": uniform(0.5, 0.49),
        "multiNB__alpha": uniform(0.01, 1.0)}

    verteilung_parameter_version3 = {
        "vectorizer1__max_features": randint(30000, 800000),  # vocabulary size
        "vectorizer1__ngram_range": [(1, 1), (1, 2)],
        "vectorizer1__min_df": randint(1, 10),
        "vectorizer1__max_df": uniform(0.8, 0.19),
        "multiNB__alpha": uniform(0.01, 1.5),
        "multiNB__fit_prior": [True, False]
    }

    rand_srch = RandomizedSearchCV(whole_pipeline, param_distributions=verteilung_parameter_version3, n_iter=50, cv=3,
                                   scoring=bewerter2, random_state=42, n_jobs=4, verbose=2
                                   )

    rand_srch.fit(x_train, y_train)

    modele[dimension] = rand_srch.best_estimator_
    best_parameter[dimension] = rand_srch.best_params_

    y_pred = modele[dimension].predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    labels = ["hoch", "mittel", "niedrig"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    cm_str = "; ".join([", ".join(map(str, row)) for row in cm])

    rows.append({
        "Dimension": dimension,
        "Accuracy": acc,
        "Macro-F1": f1,
        "Confusion Matrix (rows=hoch,mittel,niedrig)": cm_str,
        "Best Parameters": best_parameter[dimension]
    })

df_results = pd.DataFrame(rows)

print(df_results)

joblib.dump(modele, "alle_modelle_for_accuracy_score_para_version3_tf_idf_iter_50.joblib")

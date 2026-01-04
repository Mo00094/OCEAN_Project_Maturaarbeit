import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import ConfusionMatrixDisplay

traits = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism"
]
base = joblib.load(r"Alle Auswertungen Modelle Naive Bayes/Tf-IDF + MultinominalNB/base_tfidf.joblib")

tuned1 = joblib.load(r"Alle Auswertungen Modelle Naive Bayes/Tf-IDF + MultinominalNB/alle_auswertungen_for_f1_score_para_version3_tf_idf_iter_50.joblib")

tuned2 = joblib.load(r"Alle Auswertungen Modelle Naive Bayes/Tf-IDF + MultinominalNB/alle_auswertungen_for_accuracy_para_version3_tf_idf_iter_50.joblib")


for i, trait in enumerate(traits):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(trait, fontsize=20, fontweight="bold")
    ConfusionMatrixDisplay.from_predictions(
        base[trait]["y_true"], base[trait]["y_pred_base"],
        normalize="true",
        values_format=".0%",
        cmap="Blues",
        text_kw={"color": "black", "fontsize": 10},
        ax = axes[0]
    )
    axes[0].set_title("Baseline Confusion Matrix", fontsize=15, pad=15)
    axes[0].text(0.5, -0.15, f"Acc = {base[trait]['acc']:.2f} F1 = {base[trait]['f1']:.2f}", transform=axes[0].transAxes, fontsize=11, ha="center")

    ConfusionMatrixDisplay.from_predictions(
        tuned1[trait]["y_true"], tuned1[trait]["y_pred_tuned"],
        normalize="true",
        values_format=".0%",
        cmap="Blues",
        text_kw={"color": "black", "fontsize": 10},
        ax = axes[1]
    )
    axes[1].set_title("Tuned Confusion Matrix for F1 Score", fontsize=15, pad=15)
    axes[1].text(0.5, -0.15, f"Acc = {tuned1[trait]['acc']:.2f} F1 = {tuned1[trait]['f1']:.2f}", transform=axes[1].transAxes, fontsize=11, ha="center")

    ConfusionMatrixDisplay.from_predictions(
        tuned2[trait]["y_true"], tuned2[trait]["y_pred_tuned"],
        normalize="true",
        values_format=".0%",
        cmap="Blues",
        text_kw={"color": "black", "fontsize": 10},
        ax=axes[2]
    )
    axes[2].set_title("Tuned Confusion Matrix for Accuracy", fontsize=15, pad=15)
    axes[2].text(0.5, -0.15, f"Acc = {tuned2[trait]['acc']:.2f} F1 = {tuned2[trait]['f1']:.2f}",
                 transform=axes[2].transAxes, fontsize=11, ha="center")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    fig.show()


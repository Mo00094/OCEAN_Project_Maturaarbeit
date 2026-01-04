from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from argparse import Namespace
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from time import strftime
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
import numpy as np

args = Namespace(
 hidden_dim=300,
 num_epochs=100,
 learning_rate=0.001,
 batch_size=64, device="cuda"
)

#---------------------------------
#Spliting into the Sets
#---------------------------------

path1 = r"C:\Users\mauri_9qhl4wd\OneDrive - SekII Zürich\Maturarbeit\Top70%oceanprofiles.csv"

path2 = r"C:\Users\mauri_9qhl4wd\OneDrive - SekII Zürich\Maturarbeit\PANDORA\commentsfromoceanpeople.csv"

df = pd.read_csv(path1)

def get_run_logdir(base_dir=r"C:\TensorBoardLogs\OCEAN_pytorch"):
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    run_id = strftime("run_%Y_%m_%d_%H_%M_%S")
    return base_path / run_id
log_dir = get_run_logdir()
writer = SummaryWriter(log_dir=str(log_dir))

print("TensorBoard logs in:", log_dir)

for i, col in enumerate(df.columns[14:19]):
    df[str(col) + "_cat"] = pd.cut(df[col], bins=[0, 33, 66, np.inf], labels=[0, 1, 2])

cat_cols = [
    "openness_cat",
    "conscientiousness_cat",
    "extraversion_cat",
    "agreeableness_cat",
    "neuroticism_cat"
]

traits = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism"
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
        "y_test": y_test
    }

    all_splits.append(split_data)

#-----------------------------
#Vokb. + MLP
#-----------------------------

ocean_dimensionen  = ["openness_cat", "conscientiousness_cat", "extraversion_cat", "agreeableness_cat", "neuroticism_cat"]

class_gewichte = {}

for dimension in ocean_dimensionen:
    anzahl = df[dimension].value_counts().sort_index()
    gewichte = 1.0 / anzahl
    gewichte = gewichte / gewichte.sum()
    class_gewichte[dimension] = torch.tensor(gewichte.values, dtype=torch.float32).to(args.device)

df_wörter = pd.read_csv(r"C:\Users\mauri_9qhl4wd\OneDrive - SekII Zürich\Maturarbeit\PANDORA\top100_eachdimension.csv")
which_dim = ["O-Words", "C-Words", "E-Words", "A-Words", "N-Words"]
vocab_blocks = {
    dim: df_wörter[dim].dropna().astype(str).tolist()
    for dim in which_dim
}

full_vocab = vocab_blocks["O-Words"] + vocab_blocks["C-Words"] + vocab_blocks["E-Words"] + vocab_blocks["A-Words"] + vocab_blocks["N-Words"]

seen = set()
unique_vocab = []

for word in full_vocab:
    w = str(word).strip().lower()
    if w not in seen:
        seen.add(w)
        unique_vocab.append(w)

vocab_dict = {word: idx for idx, word in enumerate(unique_vocab)}

from sklearn.feature_extraction.text import CountVectorizer

vectorizer_count = CountVectorizer(
    vocabulary=vocab_dict,
    lowercase=True,
    token_pattern=r"(?u)\b\w+\b"
)


class Dataset_prep(Dataset):
    def __init__(self, ocean_df_all_splits, vectorizer):
        self.ocean_df = ocean_df_all_splits
        self._vectorizer = vectorizer
        self.train_df = self.ocean_df[0]["x_train"]
        self.train_df_y = self.ocean_df[0]["y_train"]
        self.train_size = len(self.train_df)
        self.val_df = self.ocean_df[0]["x_vald"]
        self.val_df_y = self.ocean_df[0]["y_vald"]
        self.validation_size = len(self.val_df)
        self.test_df = self.ocean_df[0]["x_test"]
        self.test_df_y = self.ocean_df[0]["y_test"]
        self.test_size = len(self.test_df)
        self._lookup_dict = {'train': (self.train_df, self.train_df_y, self.train_size),
        'val': (self.val_df, self.val_df_y, self.validation_size),
        'test': (self.test_df, self.test_df_y, self.test_size)}
        self.set_split('train')
    @classmethod
    def load_dataset_and_make_vectorizer(cls, ocean_df, vectorizer):
        return cls(ocean_df, vectorizer)
    def get_vectorizer(self):
        return self._vectorizer
    def set_split(self, split="train"):
        self._target_split = split
        self._target_df_x, self._target_df_y, self._target_size = self._lookup_dict[split]
    def __len__(self):
        return self._target_size
    def __getitem__(self, index):
        row_comments = self._target_df_x.iloc[index]
        row_oceanvalues = self._target_df_y.iloc[index]
        comments_vector = self._vectorizer.transform([row_comments]).toarray()[0]
        x_tensor = torch.tensor(comments_vector)
        ocean_values_per_row = torch.tensor(row_oceanvalues.values)
        return {'x_data': x_tensor, 'y_target': ocean_values_per_row}
    def get_num_batches(self, batch_size):
        return len(self) // batch_size



def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict


class OCEANClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(OCEANClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.4)
        )
        self.head_O = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.head_C = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.head_E = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.head_A = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.head_N = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x_in, apply_softmax=False):
        geteilt = self.fc1(x_in)
        out_O = self.head_O(geteilt)
        out_C = self.head_C(geteilt)
        out_E = self.head_E(geteilt)
        out_A = self.head_A(geteilt)
        out_N = self.head_N(geteilt)

        if apply_softmax:
            out_O = F.softmax(out_O, dim=1)
            out_C = F.softmax(out_C, dim=1)
            out_E = F.softmax(out_E, dim=1)
            out_A = F.softmax(out_A, dim=1)
            out_N = F.softmax(out_N, dim=1)

        return [out_O, out_C, out_E, out_A, out_N]

dataset = Dataset_prep.load_dataset_and_make_vectorizer(all_splits, vectorizer_count)
vectorizer = dataset.get_vectorizer()
classifier = OCEANClassifier(input_dim= len(vectorizer.vocabulary), hidden_dim=args.hidden_dim, num_classes=3)
classifier = classifier.to(args.device)
loss_O_func = nn.CrossEntropyLoss(weight=class_gewichte["openness_cat"])
loss_C_func = nn.CrossEntropyLoss(weight=class_gewichte["conscientiousness_cat"])
loss_E_func = nn.CrossEntropyLoss(weight=class_gewichte["extraversion_cat"])
loss_A_func = nn.CrossEntropyLoss(weight=class_gewichte["agreeableness_cat"])
loss_N_func = nn.CrossEntropyLoss(weight=class_gewichte["neuroticism_cat"])
optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)


train_state = {
   "train_loss": [],
   "val_loss": [],
   "train_acc": [],
   "val_acc": [],
   "epoch_index": 0
}

def accuracy_per_trait(outputs, y_true):
    accuracies = {}
    for i, trait in enumerate(traits):
        preds = outputs[i].argmax(dim=1)
        true = y_true[:, i]
        accuracies[trait] = (preds == true).float().mean().item()
    return accuracies


for epoch_index in range(args.num_epochs):
    train_state['epoch_index'] = epoch_index
    dataset.set_split('train')
    batch_generator = generate_batches(dataset, batch_size=args.batch_size, device=args.device)

    running_loss = 0.0
    running_train_acc = {t: 0.0 for t in traits}

    classifier.train()

    for batch_index, batch_dict in enumerate(batch_generator):
        optimizer.zero_grad()

        outputs = classifier(batch_dict['x_data'].float())
        y = batch_dict['y_target'].long()

        loss_O = loss_O_func(outputs[0], y[:,0])
        loss_C = loss_C_func(outputs[1], y[:,1])
        loss_E = loss_E_func(outputs[2], y[:,2])
        loss_A = loss_A_func(outputs[3], y[:,3])
        loss_N = loss_N_func(outputs[4], y[:,4])

        loss = (loss_O + loss_C + loss_E + loss_A + loss_N) / 5.0

        loss.backward()
        optimizer.step()

        running_loss = (running_loss + (loss.item() - running_loss) / (batch_index + 1))

        batch_accs = accuracy_per_trait(outputs, y)

        for tr in traits:
            running_train_acc[tr] = (running_train_acc[tr] + (batch_accs[tr] - running_train_acc[tr]) / (batch_index + 1))

    train_state["train_acc"].append(running_train_acc)
    train_state['train_loss'].append(running_loss)


    dataset.set_split('val')
    batch_generator = generate_batches(dataset, batch_size=args.batch_size, device=args.device)

    running_loss = 0.0
    running_val_acc = {t: 0.0 for t in traits}

    classifier.eval()

    for batch_index, batch_dict in enumerate(batch_generator):
        outputs = classifier(batch_dict['x_data'].float())
        y = batch_dict['y_target'].long()

        loss_O = loss_O_func(outputs[0], y[:,0])
        loss_C = loss_C_func(outputs[1], y[:,1])
        loss_E = loss_E_func(outputs[2], y[:,2])
        loss_A = loss_A_func(outputs[3], y[:,3])
        loss_N = loss_N_func(outputs[4], y[:,4])

        loss = (loss_O + loss_C + loss_E + loss_A + loss_N) / 5.0
        running_loss += (loss.item() - running_loss) / (batch_index + 1)

        batch_accs = accuracy_per_trait(outputs, y)
        for tr in traits:
            running_val_acc[tr] += (batch_accs[tr] - running_val_acc[tr]) / (batch_index + 1)

    train_state["val_acc"].append(running_val_acc)
    train_state['val_loss'].append(running_loss)

    writer.add_scalars(
        main_tag="loss",
        tag_scalar_dict={
            "train": train_state['train_loss'][-1],
            "val": train_state['val_loss'][-1]
        },
        global_step=epoch_index
    )
    for tr in traits:
        writer.add_scalars(
            main_tag="accuracy_" + tr,
            tag_scalar_dict={
                "train": train_state["train_acc"][-1][tr],
                "val": train_state["val_acc"][-1][tr]
            },
            global_step=epoch_index
        )
    for name, param in classifier.named_parameters():
        writer.add_histogram(name, param, epoch_index)

dataset.set_split("train")

dummy_batch = next(
    generate_batches(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        device=args.device
    )
)

dummy_x = dummy_batch["x_data"].float()
writer.add_graph(classifier, dummy_x)

dataset.set_split("test")
X_test_text = dataset._target_df_x
X_test_vec = vectorizer.transform(X_test_text).toarray()
X_test_tensor = torch.tensor(X_test_vec).float().to(args.device)
classifier.eval()
outputs = classifier(X_test_tensor, apply_softmax=True)
y_pred = [p.argmax(dim=1).cpu().numpy() for p in outputs]

cols = ["openness_cat", "conscientiousness_cat", "extraversion_cat", "agreeableness_cat", "neuroticism_cat"]

epoche = range(1, args.num_epochs + 1)

y_test_list = dataset._target_df_y

for i, (col, trait) in enumerate(zip(cols, traits)):
    y_true = y_test_list[col]
    y_hat = y_pred[i]

    acc = accuracy_score(y_true, y_hat)
    f1 = f1_score(y_true, y_hat, average="macro")

    loss_training = train_state["train_loss"]
    loss_validation = train_state["val_loss"]

    genau_training = []
    for epoche_genau in train_state["train_acc"]:
        wert = epoche_genau[trait]
        genau_training.append(wert)
    genau_validation = []
    for epoche_genau in train_state["val_acc"]:
        wert = epoche_genau[trait]
        genau_validation.append(wert)

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

traced = torch.jit.trace(classifier, dummy_x)
#torch.jit.save(traced, "model_version1_vokab+mlp.pt")

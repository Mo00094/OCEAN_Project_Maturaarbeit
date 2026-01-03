import pandas as pd
import matplotlib.pyplot as plt

path = r"../Files/oceanprofilesfile.csv"
df = pd.read_csv(path)

fig, axes = plt.subplots(1, 5, figsize=(20, 10))

for i, col in enumerate(df.columns[13:18]):
    ax = axes[i]

    counts, bin_edges, _ = ax.hist(
        df[col],
        bins=8,
        linewidth=0.5,
        edgecolor="white"
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax.plot(bin_centers, counts, marker="o")

    ax.set_title(col)
    ax.set_ylabel("Anzahl")
    ax.set_xlabel("Wert")
    ax.set_ylim(0, 500)

plt.tight_layout()
plt.show()
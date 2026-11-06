import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# ---------------------------------------------------------------
# Confusion matrices on the test patient chb08 (fixed split)
# ---------------------------------------------------------------
matrices = {
    "Original Logistic Regression\n(1 channel)": np.array([
        [1229, 479],
        [20, 69]
    ]),
    "Full pipeline Logistic Regression\n(18 channels)": np.array([
        [1619, 89],
        [62, 27]
    ])
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

for ax, (title, cm) in zip(axes, matrices.items()):
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-seizure", "Seizure"]
    ).plot(ax=ax, colorbar=False)
    ax.set_title(title)

fig.suptitle("Test patient chb08: confusion matrices")
plt.tight_layout()

plt.savefig(
    "figures/logistic_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ---------------------------------------------------------------
# Precision, recall and F1 on the test patient chb08
# ---------------------------------------------------------------
methods = [
    "Non-ML\nBaseline",
    "Original LR\n(1 channel)",
    "Improved LR\n(1 channel)",
    "Full pipeline LR\n(18 channels)"
]

precision_values = [0.1043, 0.1259, 0.142, 0.233]
recall_values = [0.1348, 0.7753, 0.180, 0.303]
f1_values = [0.1176, 0.2166, 0.158, 0.263]

x = np.arange(len(methods))
width = 0.25

plt.figure(figsize=(9, 4.5))

plt.bar(x - width, precision_values, width, label="Precision")
plt.bar(x, recall_values, width, label="Recall")
plt.bar(x + width, f1_values, width, label="F1")

plt.xticks(x, methods)
plt.ylabel("Score")
plt.title("Test patient chb08 (fixed split)")
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()

plt.savefig(
    "figures/model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

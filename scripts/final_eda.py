import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/final_features.csv")

class_counts = (
    df.groupby(["split", "label"])
    .size()
    .unstack(fill_value=0)
)

print(class_counts)

class_counts.plot(
    kind="bar"
)

plt.xlabel("Dataset split")
plt.ylabel("Number of 4-second windows")
plt.title("Class distribution across train, validation, and test")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("figures/class_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure()

df.boxplot(
    column="line_length",
    by="label"
)

plt.xlabel("Class")
plt.ylabel("Line length")
plt.title("Line length distribution by class")
plt.suptitle("")
plt.xticks(
    [1, 2],
    ["Non-seizure", "Seizure"]
)

plt.tight_layout()

plt.savefig(
    "figures/line_length_by_class.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


plt.figure()

df.boxplot(
    column="rms",
    by="label"
)

plt.xlabel("Class")
plt.ylabel("RMS (V)")
plt.title("RMS distribution by class")
plt.suptitle("")

plt.xticks(
    [1, 2],
    ["Non-seizure", "Seizure"]
)

plt.tight_layout()

plt.savefig(
    "figures/rms_by_class.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
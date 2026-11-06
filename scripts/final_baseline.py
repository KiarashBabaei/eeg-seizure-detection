import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("results/final_features.csv")

print(df.groupby(["split", "label"]).size())
validation_df = df[df["split"] == "validation"]

thresholds = [0.005, 0.01, 0.015, 0.02, 0.025]

best_threshold = None
best_f1 = -1

for threshold in thresholds:
    predictions = (
        validation_df["line_length"] >= threshold
    ).astype(int)

    precision = precision_score(
        validation_df["label"],
        predictions,
        zero_division=0
    )

    recall = recall_score(
        validation_df["label"],
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        validation_df["label"],
        predictions,
        zero_division=0
    )

    print(
        "threshold =", threshold,
        "precision =", precision,
        "recall =", recall,
        "f1 =", f1
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\nBest threshold:", best_threshold)
print("Best validation F1:", best_f1)
test_df = df[df["split"] == "test"]

test_predictions = (
    test_df["line_length"] >= best_threshold
).astype(int)

test_precision = precision_score(
    test_df["label"],
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    test_df["label"],
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    test_df["label"],
    test_predictions,
    zero_division=0
)

test_cm = confusion_matrix(
    test_df["label"],
    test_predictions
)

print("\nTEST RESULTS")
print("precision =", test_precision)
print("recall =", test_recall)
print("f1 =", test_f1)
print(test_cm)

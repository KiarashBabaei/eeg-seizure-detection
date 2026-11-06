import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("results/final_features.csv")

feature_columns = [
    "variance",
    "rms",
    "line_length"
]

train_df = df[df["split"] == "train"]
validation_df = df[df["split"] == "validation"]
test_df = df[df["split"] == "test"]

X_train = train_df[feature_columns]
y_train = train_df["label"]

X_validation = validation_df[feature_columns]
y_validation = validation_df["label"]

X_test = test_df[feature_columns]
y_test = test_df["label"]

print("Train:")
print(y_train.value_counts())

print("\nValidation:")
print(y_validation.value_counts())

print("\nTest:")
print(y_test.value_counts())
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic_regression", LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000
    ))
])

model.fit(X_train, y_train)

print("\nModel trained successfully")
validation_predictions = model.predict(X_validation)

validation_precision = precision_score(
    y_validation,
    validation_predictions,
    zero_division=0
)

validation_recall = recall_score(
    y_validation,
    validation_predictions,
    zero_division=0
)

validation_f1 = f1_score(
    y_validation,
    validation_predictions,
    zero_division=0
)

validation_cm = confusion_matrix(
    y_validation,
    validation_predictions
)

print("\nVALIDATION RESULTS")
print("precision =", validation_precision)
print("recall =", validation_recall)
print("f1 =", validation_f1)
print(validation_cm)
test_predictions = model.predict(X_test)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_cm = confusion_matrix(
    y_test,
    test_predictions
)

print("\nTEST RESULTS")
print("precision =", test_precision)
print("recall =", test_recall)
print("f1 =", test_f1)
print(test_cm)
logistic_model = model.named_steps["logistic_regression"]

coefficients = logistic_model.coef_[0]

print("\nCOEFFICIENTS")

for feature, coefficient in zip(feature_columns, coefficients):
    print(feature, coefficient)
    
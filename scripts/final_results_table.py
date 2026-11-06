import pandas as pd

# Values copied from the printed outputs of:
#   final_baseline.py, final_logistic_regression.py   (original models)
#   improved_model.py      -> improved_model_results.txt
#   full_pipeline.py       -> full_pipeline_results.txt
#
# "Test" columns: fixed split, test patient chb08.
# "LOPO" columns: mean over 5 patients, leave-one-patient-out.
# The baseline was not run with leave-one-patient-out.

results = pd.DataFrame({
    "Method": [
        "Non-ML Baseline",
        "Original Logistic Regression",
        "Improved Logistic Regression",
        "Improved Gradient Boosting",
        "Full pipeline Logistic Regression",
        "Full pipeline Gradient Boosting"
    ],
    "Channels": [1, 1, 1, 1, 18, 18],
    "Test_Precision": [0.1043, 0.1259, 0.142, 0.196, 0.233, 0.189],
    "Test_Recall": [0.1348, 0.7753, 0.180, 0.438, 0.303, 0.236],
    "Test_F1": [0.1176, 0.2166, 0.158, 0.271, 0.263, 0.210],
    "LOPO_mean_AUC": [None, 0.881, 0.938, 0.941, 0.957, 0.953],
    "LOPO_mean_F1": [None, 0.209, 0.397, 0.332, 0.327, 0.290]
})

print(results.to_string(index=False))

results.to_csv(
    "results/final_results_table.csv",
    index=False
)

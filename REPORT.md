# Explainable Machine Learning for Epileptic Seizure Detection from EEG Signals

> **Note.** Sections 1 to 9 describe the original course project and its results. Section 10 reports follow-up work with improved features, multichannel EEG and leave-one-patient-out evaluation. The original text is kept unchanged so the two versions can be compared.

## 1. Introduction

Epilepsy is a neurological disorder characterized by recurrent seizures caused by abnormal electrical activity in the brain. Electroencephalography (EEG) is widely used to record brain electrical activity and can provide useful information for identifying seizure episodes.

Manual inspection of EEG recordings can be time-consuming, especially when long recordings must be reviewed. Machine learning methods can potentially support this task by identifying patterns in EEG signals that differ between seizure and non-seizure periods.

The objective of this project is to investigate whether simple and interpretable features extracted from short EEG windows can distinguish seizure from non-seizure activity. The project focuses on seizure detection rather than seizure prediction. In other words, the goal is to identify whether a given EEG window corresponds to an ongoing seizure, not to predict a future seizure.

To keep the analysis suitable for a basic machine-learning project, the study uses a limited subset of the CHB-MIT Scalp EEG Database and compares two simple approaches: a non-machine-learning threshold-based baseline and a trained Logistic Regression model. The baseline uses line length as a single interpretable EEG feature, while the Logistic Regression model uses variance, root mean square (RMS), and line length.

The main research question is whether a trained machine-learning model can improve seizure detection compared with a simple threshold rule when evaluated on an unseen patient. Particular attention is given to class imbalance, patient-level data separation, interpretable features, and the limitations of the resulting model.

This project is an offline educational analysis and is not intended to represent a clinical or real-time seizure detection system.

## 2. Dataset

The CHB-MIT Scalp EEG Database from PhysioNet was used for this project. The database contains scalp EEG recordings together with seizure timing information provided in summary files.

To keep the project computationally manageable and suitable for a basic machine-learning study, only a limited subset of patients and recordings was selected instead of analyzing the full database.

Five patients were included in the final experiment:

- Training patients: chb01, chb03, and chb05
- Validation patient: chb07
- Test patient: chb08

This patient-level separation was used to reduce data leakage between the different parts of the experiment. In particular, the final test patient was not included in either the training or validation sets.

The selected EEG recordings were sampled at 256 Hz. For consistency across the selected recordings, the FP1-F7 EEG channel was used in the final analysis.

The recordings were divided into non-overlapping 4-second windows. Each window was assigned a binary label:

- 0: non-seizure
- 1: seizure

Windows fully contained inside an annotated seizure interval were labeled as seizure. Windows fully outside seizure intervals were labeled as non-seizure. Windows that partially overlapped seizure boundaries were discarded to avoid ambiguous labeling.

The resulting feature dataset contained 8,119 EEG windows:

- Training: 5,294 non-seizure and 99 seizure windows
- Validation: 906 non-seizure and 23 seizure windows
- Test: 1,708 non-seizure and 89 seizure windows

The dataset therefore showed substantial class imbalance.

In this project, “non-seizure” means only that no seizure was annotated during that specific window. It does not mean that the patient was neurologically healthy.

## 3. Exploratory Data Analysis

Exploratory data analysis was performed on the extracted EEG-window features to understand the class distribution and examine whether seizure and non-seizure windows showed different feature patterns.

The first important observation was strong class imbalance. In all dataset splits, non-seizure windows greatly outnumbered seizure windows. This is shown in `figures/class_distribution.png`. Because of this imbalance, accuracy alone would not be an appropriate evaluation metric, since a model could achieve high accuracy simply by predicting most windows as non-seizure.

The distribution of line length was examined for seizure and non-seizure windows. As shown in `figures/line_length_by_class.png`, seizure windows generally tended to have higher line-length values. However, substantial overlap remained between the two classes, and some non-seizure windows also showed high line-length values. This suggested that line length could provide a useful simple baseline but was unlikely to separate the classes perfectly using a single threshold.

RMS showed a similar pattern. In `figures/rms_by_class.png`, seizure windows generally tended toward higher RMS values than non-seizure windows, although considerable overlap remained.

Overall, the EDA suggested that simple time-domain features contained useful information related to seizure activity but that no single feature provided complete separation between seizure and non-seizure windows. This motivated comparison of a single-feature threshold baseline with a trained multivariable Logistic Regression model.

## 4. Methods

### 4.1 EEG Windowing and Labeling

Each selected EEG recording was divided into non-overlapping 4-second windows. With a sampling rate of 256 Hz, each complete window contained 1,024 samples from the FP1-F7 channel.

A simple labeling rule was applied using the seizure start and end times provided in the CHB-MIT summary files:

- Windows fully outside a seizure interval were labeled non-seizure.
- Windows fully inside a seizure interval were labeled seizure.
- Windows partially overlapping the beginning or end of a seizure were discarded.

This conservative rule was used to avoid assigning ambiguous labels to windows containing only part of a seizure interval.

### 4.2 Feature Extraction

Three simple time-domain features were extracted from each EEG window:

- Variance
- Root Mean Square (RMS)
- Line Length

Variance describes variability of signal amplitude within the window.

RMS summarizes the overall magnitude of the EEG signal.

Line length measures the cumulative absolute change between consecutive signal samples and therefore reflects both signal amplitude and fluctuations over time.

These features were selected because they are computationally simple, interpretable, and suitable for a basic machine-learning project.

Each row of the final feature table represented one EEG window. Patient ID, recording ID, dataset split, window start time, and window end time were retained as metadata but were not used as model input features.

### 4.3 Data Splitting Strategy

A patient-level split was used to reduce the risk of data leakage and evaluate generalization to an unseen patient.

The final split was:

- Training: chb01, chb03, chb05
- Validation: chb07
- Test: chb08

No patient appeared in more than one split.

The training set was used to fit the Logistic Regression model. The validation set was used to select the non-ML baseline threshold. The test set was kept separate until final evaluation.

This design is stricter than randomly splitting individual EEG windows because neighboring windows from the same patient or recording can be highly similar.

### 4.4 Non-ML Baseline

A simple threshold-based rule using line length was implemented as the non-machine-learning baseline.

A small predefined set of candidate thresholds was evaluated on the validation set:

- 0.005
- 0.010
- 0.015
- 0.020
- 0.025

For each threshold, seizure predictions were generated using the rule:

`predict seizure if line length >= threshold`

The threshold producing the highest validation F1 score was selected. The selected threshold was 0.025.

The threshold was then fixed and evaluated on the test patient without further adjustment.

### 4.5 Logistic Regression

The main trained model was Logistic Regression using three input features:

- Variance
- RMS
- Line length

A scikit-learn Pipeline was used to combine feature scaling and model training.

`StandardScaler` was applied before Logistic Regression so that the three features were represented on comparable scales.

Because the training data were strongly imbalanced, the model used:

`class_weight="balanced"`

The final model settings were:

- LogisticRegression
- `class_weight="balanced"`
- `random_state=42`
- `max_iter=1000`

The model was fitted using only the training set.

Performance was evaluated using precision, seizure recall, F1 score, and the confusion matrix. Accuracy was not used as the main metric because of the strong class imbalance.

## 5. Results

Final evaluation was performed on the unseen test patient, chb08.

The non-ML baseline achieved:

- Precision: 0.1043
- Recall: 0.1348
- F1 score: 0.1176

Its confusion matrix contained:

- True Negatives: 1605
- False Positives: 103
- False Negatives: 77
- True Positives: 12

The Logistic Regression model achieved:

- Precision: 0.1259
- Recall: 0.7753
- F1 score: 0.2166

Its confusion matrix contained:

- True Negatives: 1229
- False Positives: 479
- False Negatives: 20
- True Positives: 69

The trained Logistic Regression model therefore detected substantially more seizure windows than the simple threshold baseline. Seizure recall increased from 0.1348 to 0.7753.

F1 score also increased from 0.1176 to 0.2166, while precision increased only slightly from 0.1043 to 0.1259.

However, the increased seizure sensitivity was accompanied by a much larger number of false-positive windows. Logistic Regression produced 479 false positives compared with 103 for the non-ML baseline.

The results therefore demonstrate a clear trade-off: the trained model was considerably more sensitive to seizure activity but generated substantially more false positives.

The final performance comparison is summarized in `results/final_results_table.csv` and visualized in `figures/model_comparison.png`. The Logistic Regression confusion matrix is shown in `figures/logistic_confusion_matrix.png`.

## 6. Explainability

To provide a simple interpretation of the trained Logistic Regression model, its coefficients were examined after feature standardization.

The final coefficients were:

- Variance: -0.9238
- RMS: +2.0933
- Line length: +1.0026

Because the input features were standardized before model fitting, the relative magnitude and direction of the coefficients can be compared more directly.

RMS had the largest positive coefficient, indicating that higher standardized RMS values contributed most strongly toward seizure predictions in the fitted model.

Line length also had a positive coefficient, meaning that larger line-length values tended to increase the model's prediction toward the seizure class.

Variance had a negative coefficient in the final model. This does not imply that higher EEG variance is physiologically protective or directly associated with non-seizure activity. Logistic Regression coefficients describe the contribution of each feature while the other features are simultaneously present in the model. In addition, variance and RMS are mathematically related measures of signal magnitude, which may influence their individual coefficient values.

The coefficients should therefore be interpreted as model-specific associations rather than causal physiological relationships.

## 7. Discussion

The results show that the trained Logistic Regression model performed better than the simple non-ML threshold baseline on the unseen test patient, especially in seizure recall.

The most important improvement was the increase in recall from 0.1348 for the baseline to 0.7753 for Logistic Regression. The trained model therefore detected a substantially larger proportion of seizure windows.

However, this improvement came with a clear cost. Logistic Regression generated substantially more false-positive windows than the baseline. False positives increased from 103 to 479, and precision remained low at 0.1259.

The use of class weighting gave greater importance to the minority seizure class during Logistic Regression training. This is consistent with the observed high seizure sensitivity, although the experiment also demonstrates the resulting trade-off with false-positive predictions.

The results are consistent with the exploratory data analysis. Seizure windows generally showed higher RMS and line-length values, but substantial overlap remained between seizure and non-seizure distributions. These features therefore contain useful predictive information without completely separating the classes.

The weaker generalization of the threshold baseline to the test patient also suggests that a simple threshold selected using one patient may not transfer effectively to another patient. This illustrates the difficulty of patient-independent seizure detection and the variability of EEG signals between individuals.

Overall, this experiment supports the hypothesis that combining several simple EEG features in a trained model can outperform a single-feature threshold rule. However, the low precision and high number of false-positive windows indicate that the model remains limited and should not be interpreted as a clinically reliable seizure detector.

## 8. Limitations

This project has several important limitations.

First, only five patients from the CHB-MIT database were included. Although the train, validation, and test sets were separated at the patient level, final performance was measured on only one unseen patient. The reported results may therefore depend strongly on the patient selected for testing and should not be generalized to the full CHB-MIT population.

Second, only the FP1-F7 EEG channel was analyzed. EEG is a multichannel signal, and seizure-related activity may be more visible in other channels or combinations of channels. Restricting the analysis to one channel simplified the project but may have reduced detection performance.

Third, only three simple time-domain features were used: variance, RMS, and line length. These features are interpretable and computationally inexpensive but do not capture all temporal or frequency-domain information present in EEG signals.

Fourth, the selected subset was based on seizure-containing recordings. Consequently, the non-seizure windows in this project do not represent the complete range of non-seizure recordings available in the full CHB-MIT dataset.

Fifth, the dataset was strongly imbalanced, with far fewer seizure windows than non-seizure windows. Class weighting was used in Logistic Regression to address this imbalance, but the resulting model showed a clear sensitivity–false-positive trade-off.

Sixth, evaluation was performed at the window level. A false-positive window is not equivalent to a clinical false-alarm event, and multiple consecutive positive windows could correspond to a single alarm event. False alarms per hour and event-level seizure detection performance were therefore not reported.

Finally, this study was designed as a basic educational machine-learning project. It was not intended to develop or validate a clinically deployable seizure detection system.

## 9. Conclusion

This project investigated whether simple features extracted from short EEG windows could distinguish seizure from non-seizure activity and whether a trained machine-learning model could outperform a simple non-ML baseline.

Using a patient-level split on a limited subset of the CHB-MIT dataset, a threshold-based baseline using line length was compared with a Logistic Regression model using variance, RMS, and line length.

The Logistic Regression model achieved better performance on the unseen test patient, particularly in seizure recall, which increased from 0.1348 for the baseline to 0.7753. F1 score also increased from 0.1176 to 0.2166.

However, precision remained low and the Logistic Regression model produced many false-positive windows. The trained model was therefore more sensitive to seizure activity but was not sufficiently reliable for practical or clinical use.

Overall, the results show that simple explainable EEG features can provide useful information for seizure detection and that combining multiple features in a trained model can improve performance compared with a single-feature threshold rule. At the same time, the results highlight the difficulty of generalizing seizure detection across patients using a small dataset and simple features.

## 10. Follow-up: Improved Models

After the original project, two further pipelines were tested. The aim was to raise performance without leaking information from the test patient.

### 10.1 Changes

The first pipeline (`scripts/improved_model.py`) used the same FP1-F7 feature table. RMS and line length were log-transformed because EEG amplitudes are strongly skewed. Variance was removed because it is almost equal to RMS squared. A ratio of line length to RMS was added as a simple proxy for signal frequency. Each window also received the mean of its 5 and 9 neighboring windows, since a seizure lasts much longer than 4 seconds. Predicted probabilities were smoothed across 3 windows.

The second pipeline (`scripts/full_pipeline.py`) read the raw EDF files. It used 18 bipolar channels filtered at 0.5 to 40 Hz. For each channel it computed log line length, log power, relative power in five frequency bands and spectral entropy. Each feature was then summarized across channels by its median, maximum and standard deviation.

### 10.2 Evaluation

The original split tested the model on one patient only. For this reason, leave-one-patient-out testing was added. Each of the five patients was held out once, and the model was trained on the other four. The decision threshold was chosen on the training patients only.

Two checks were used to detect leakage and overfitting. First, labels were randomly shuffled and the full procedure was repeated. Second, training AUC was compared with test AUC.

### 10.3 Results

| Model | Channels | chb08 F1 | Mean F1 (5 patients) | Mean AUC (5 patients) |
|---|---:|---:|---:|---:|
| Original Logistic Regression | 1 | 0.217 | 0.209 | 0.881 |
| Improved Logistic Regression | 1 | 0.158 | 0.397 | 0.938 |
| Full pipeline Logistic Regression | 18 | 0.263 | 0.327 | 0.957 |

The improved single-channel model almost doubled the mean F1 across patients. It performed worse on chb08 alone, which shows how much one test patient can bias the result. The 18-channel model gave the highest mean AUC and the best chb08 F1. On chb08 it detected both seizures with 0.5 false alarms per hour.

With shuffled labels, mean AUC dropped to about 0.52 for all models. This is chance level, so the pipelines do not leak test information. The improved single-channel model had almost equal training and test AUC (0.94). The 18-channel model reached a training AUC near 0.99, which suggests mild overfitting with 30 features and five patients.

### 10.4 Remaining limitations

The threshold did not transfer well between patients. For chb01 and chb03, AUC was close to 0.99 but recall at the chosen threshold stayed below 0.1. Precision also remained low. Adding more CHB-MIT patients is the most direct way to address both problems.

## 11. References

1. PhysioNet. CHB-MIT Scalp EEG Database.

2. Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation. 2000;101(23):e215-e220.

3. Shoeb AH. Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment. PhD thesis, Massachusetts Institute of Technology, 2009.

4. scikit-learn documentation. LogisticRegression, StandardScaler, Pipeline, and classification metrics.

5. MNE-Python documentation. Reading and processing EDF EEG recordings.

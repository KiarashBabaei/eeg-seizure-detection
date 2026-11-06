import mne
import numpy as np
import pandas as pd


recordings = [
    # Train
    {
        "patient_id": "chb01",
        "recording_id": "chb01_03",
        "file": "data/chb01/chb01_03.edf",
        "seizure_start": 2996,
        "seizure_end": 3036,
        "split": "train"
    },
    {
        "patient_id": "chb01",
        "recording_id": "chb01_04",
        "file": "data/chb01/chb01_04.edf",
        "seizure_start": 1467,
        "seizure_end": 1494,
        "split": "train"
    },
    {
        "patient_id": "chb03",
        "recording_id": "chb03_01",
        "file": "data/chb03/chb03_01.edf",
        "seizure_start": 362,
        "seizure_end": 414,
        "split": "train"
    },
    {
        "patient_id": "chb03",
        "recording_id": "chb03_02",
        "file": "data/chb03/chb03_02.edf",
        "seizure_start": 731,
        "seizure_end": 796,
        "split": "train"
    },
    {
        "patient_id": "chb05",
        "recording_id": "chb05_06",
        "file": "data/chb05/chb05_06.edf",
        "seizure_start": 417,
        "seizure_end": 532,
        "split": "train"
    },
    {
        "patient_id": "chb05",
        "recording_id": "chb05_13",
        "file": "data/chb05/chb05_13.edf",
        "seizure_start": 1086,
        "seizure_end": 1196,
        "split": "train"
    },

    # Validation
    {
        "patient_id": "chb07",
        "recording_id": "chb07_13",
        "file": "data/chb07/chb07_13.edf",
        "seizure_start": 3285,
        "seizure_end": 3381,
        "split": "validation"
    },

    # Test
    {
        "patient_id": "chb08",
        "recording_id": "chb08_02",
        "file": "data/chb08/chb08_02.edf",
        "seizure_start": 2670,
        "seizure_end": 2841,
        "split": "test"
    },
    {
        "patient_id": "chb08",
        "recording_id": "chb08_05",
        "file": "data/chb08/chb08_05.edf",
        "seizure_start": 2856,
        "seizure_end": 3046,
        "split": "test"
    }
]


window_seconds = 4
all_rows = []


for recording in recordings:
    raw = mne.io.read_raw_edf(
        recording["file"],
        preload=False
    )

    sampling_rate = int(raw.info["sfreq"])
    recording_seconds = int(raw.times[-1]) + 1

    seizure_start = recording["seizure_start"]
    seizure_end = recording["seizure_end"]

    for start_time in range(
        0,
        recording_seconds,
        window_seconds
    ):
        end_time = start_time + window_seconds

        # Skip incomplete final window
        if end_time > recording_seconds:
            continue

        # Label window
        if end_time <= seizure_start or start_time >= seizure_end:
            label = 0

        elif start_time >= seizure_start and end_time <= seizure_end:
            label = 1

        else:
            # Partial seizure overlap
            continue

        start_sample = start_time * sampling_rate
        end_sample = end_time * sampling_rate

        # Use first EEG channel: FP1-F7
        window_data, _ = raw[
            0,
            start_sample:end_sample
        ]

        signal = window_data[0]

        # Feature extraction
        variance_value = np.var(signal)

        rms_value = np.sqrt(
            np.mean(signal ** 2)
        )

        line_length_value = np.sum(
            np.abs(np.diff(signal))
        )

        all_rows.append([
            recording["patient_id"],
            recording["recording_id"],
            recording["split"],
            start_time,
            end_time,
            label,
            variance_value,
            rms_value,
            line_length_value
        ])


df = pd.DataFrame(
    all_rows,
    columns=[
        "patient_id",
        "recording_id",
        "split",
        "start_time",
        "end_time",
        "label",
        "variance",
        "rms",
        "line_length"
    ]
)


print("Dataset shape:")
print(df.shape)

print("\nClass distribution by split:")
print(
    df.groupby(
        ["split", "label"]
    ).size()
)

print("\nPatients by split:")
print(
    df.groupby("split")[
        "patient_id"
    ].unique()
)
df.to_csv(
    "results/final_features.csv",
    index=False
)

print("\nSaved results/final_features.csv")

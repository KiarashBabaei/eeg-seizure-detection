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

df_metadata = pd.DataFrame(recordings)

print(df_metadata)
print()
print(df_metadata.groupby("split")["patient_id"].unique())

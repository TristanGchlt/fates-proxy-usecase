import pandas as pd

def nullify_values(dataset : pd.DataFrame, old_value : str) -> pd.DataFrame :
    dataset_clean = dataset.copy()
    dataset_clean.replace(old_value, pd.NA, inplace=True)
    return dataset_clean

def fill_missing_values(dataset : pd.DataFrame, fill_value : str = "Unknown") -> pd.DataFrame :
    dataset_clean = dataset.copy()
    for col in dataset_clean.columns:
        dataset[col] = dataset[col].fillna(fill_value)
    return dataset
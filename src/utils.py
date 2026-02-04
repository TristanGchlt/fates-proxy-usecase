import pandas as pd

def load_csv(path : str) -> pd.DataFrame :
    dataset = pd.read_csv(path)
    return dataset

def save_csv(dataset : pd.DataFrame, path : str) :
    dataset.to_csv(path, index=None)
    return None
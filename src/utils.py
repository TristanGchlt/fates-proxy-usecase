import pandas as pd
import yaml

def load_csv(path : str) -> pd.DataFrame :
    dataset = pd.read_csv(path)
    return dataset

def save_csv(dataset : pd.DataFrame, path : str) :
    dataset.to_csv(path, index=None)
    return None

def read_config(config_path : str) :
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config
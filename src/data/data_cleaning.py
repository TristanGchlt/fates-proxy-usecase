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

def create_binary_column(dataset : pd.DataFrame, column_name : str, target_value : str) -> pd.DataFrame:
    dataset_clean = dataset.copy()
    dataset_clean[column_name] = dataset_clean[column_name] == target_value
    return dataset_clean

def create_dummy_columns(dataset : pd.DataFrame, column_names : list[str]) -> pd.DataFrame :
    dataset_clean = dataset.copy()
    dataset_clean = pd.get_dummies(dataset_clean, columns = column_names)
    return dataset_clean

def convert_columns_type(dataset : pd.DataFrame, type_to_convert : str, new_type : str) -> pd.DataFrame :
    dataset_clean = dataset.copy()

    cols_to_convert = dataset_clean.select_dtypes(include=type_to_convert).columns

    if len(cols_to_convert) > 0:
        dataset_clean[cols_to_convert] = dataset_clean[cols_to_convert].astype(new_type)

    return dataset_clean

def remove_columns(dataset : pd.DataFrame, columns : list[str]) -> pd.DataFrame :
    dataset_clean = dataset.copy()

    dataset_clean = dataset_clean.drop(columns=columns)

    return dataset_clean
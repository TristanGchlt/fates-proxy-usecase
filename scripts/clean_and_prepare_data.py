import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.data_cleaning import nullify_values, fill_missing_values
from src.data.data_preparation import create_binary_column, create_dummy_columns, convert_columns_type, remove_columns
from src.utils import load_csv, save_csv

RAW_DATA_FILE = PROJECT_ROOT / "data" / "raw" / "data.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"


def main() :
    dataset = load_csv(RAW_DATA_FILE)

    # Missing values replacement
    dataset = nullify_values(dataset, "?")
    dataset = fill_missing_values(dataset, fill_value="Unknown")

    # From qualitative to binary
    dataset = create_binary_column(dataset, "sex", "Male")
    dataset = create_binary_column(dataset, "income", ">50K")
    categorical_columns = [
        'workclass',
        'marital.status',
        'occupation',
        'relationship',
        'native.country'
    ]
    dataset = create_dummy_columns(dataset, categorical_columns)

    # Type transformation
    dataset = convert_columns_type(dataset, "int", "float64")

    # Agregated feature
    dataset['capital_diff'] = dataset['capital.gain'] - dataset['capital.loss']

    # Columns to be removed
    columns_to_be_removed = [
        'race',
        'fnlwgt',
        'education',
        'capital.gain',
        'capital.loss'
    ]
    dataset = remove_columns(dataset, columns_to_be_removed)

    # Save the data
    save_csv(dataset, OUTPUT_PATH)

if __name__ == "__main__" :
    main()
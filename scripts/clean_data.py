import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.data_cleaning import clean
from src.utils import load_csv, save_csv

RAW_DATA_FILE = PROJECT_ROOT / "data" / "raw" / "data.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"


def main(raw_data_file=RAW_DATA_FILE, output_path=OUTPUT_PATH) :
    
    # LOAD
    dataset = load_csv(raw_data_file)

    # CLEAN

    clean_dataset = clean(dataset)

    # SAVE
    save_csv(clean_dataset, output_path)

    return clean_dataset

if __name__ == "__main__" :
    main()
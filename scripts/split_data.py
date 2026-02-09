import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_csv, save_csv
from src.data.split import train_test, x_y_p

PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"

# To be sent to a config file
PROTECTED_FEATURE = "sex"
TARGET_FEATURE = "income"
RANDOM_STATE = 1
SPLIT_TEST_SIZE = 0.8
SPLIT_PATH = PROJECT_ROOT / "data" / "split"


def main(processed_data_path=PROCESSED_DATA_PATH, 
         test_size=SPLIT_TEST_SIZE, 
         seed=RANDOM_STATE, 
         target_feature=TARGET_FEATURE, 
         protected_feature=PROTECTED_FEATURE,
         split_path=SPLIT_PATH
         ) :
    
    dataset = load_csv(processed_data_path)

    train, test = train_test(dataset, test_size, seed)

    X_train, y_train, p_train = x_y_p(train, target_feature, protected_feature)
    X_test, y_test, p_test = x_y_p(test, target_feature, protected_feature)

    save_csv(X_train, split_path / "X_train.csv") 
    save_csv(y_train, split_path / "y_train.csv")
    save_csv(p_train, split_path / "p_train.csv")
    save_csv(X_test, split_path / "X_test.csv") 
    save_csv(y_test, split_path / "y_test.csv")
    save_csv(p_test, split_path / "p_test.csv")

    return X_train, y_train, p_train, X_test, y_test, p_test

if __name__ == "__main__" :
    main()
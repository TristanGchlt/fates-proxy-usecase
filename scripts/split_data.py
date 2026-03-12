import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_csv, save_csv, read_config
from src.data.split import train_test, x_y_p

PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
SPLIT_PATH = PROJECT_ROOT / "data" / "split"


def main(config_path=CONFIG_PATH,
         processed_data_path=PROCESSED_DATA_PATH,
         split_path=SPLIT_PATH
         ) :
    
    config = read_config(config_path)
    test_size=config['test_size']
    seed = config['random_state']
    target_feature = config['target_feature']
    protected_feature = config['protected_feature']
    split_strategy = config['f_split_strategy']

    dataset = load_csv(processed_data_path)

    train, test = train_test(dataset, test_size, protected_feature, split_strategy, seed)

    X_train, y_train, p_train = x_y_p(train, target_feature, protected_feature)
    X_test, y_test, p_test = x_y_p(test, target_feature, protected_feature)

    save_csv(X_train, split_path / "X_train.csv") 
    save_csv(y_train, split_path / "y_train.csv")
    save_csv(p_train, split_path / "p_train.csv")
    save_csv(X_test, split_path / "X_test.csv") 
    save_csv(y_test, split_path / "y_test.csv")
    save_csv(p_test, split_path / "p_test.csv")

    return {
        "test_size" : test_size,
        "split_seed" : seed,
        "target" : target_feature,
        "protected" : protected_feature,
        "split_strategy" : split_strategy
    }

if __name__ == "__main__" :
    main()
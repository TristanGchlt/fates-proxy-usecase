import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_csv, save_csv, read_config
from src.data.data_preparation import split

PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
SPLIT_PATH = PROJECT_ROOT / "data" / "split"


def main(config_path=CONFIG_PATH,
         processed_data_path=PROCESSED_DATA_PATH,
         split_path=SPLIT_PATH
         ) :

    # LOAD

    dataset = load_csv(processed_data_path)
    

    # SPLIT

    config = read_config(config_path)

    X_train, y_train, p_train, X_test, y_test, p_test = split(dataset, 
                                                              test_size = config['test_size'], 
                                                              protected_feature = config['protected_feature'], 
                                                              split_strategy = config['f_split_strategy'], 
                                                              seed = config['split_random_seed'], 
                                                              target_feature = config['target_feature'], 
                                                              balance_strategy = config['f_balance'], 
                                                              balance_seed = config['f_balance_seed'], 
                                                              hide_protected = config["f_hide_protected"],
                                                              hide_proxies= config['f_hide_proxies'])

    # SAVE

    save_csv(X_train, split_path / "X_train.csv") 
    save_csv(y_train, split_path / "y_train.csv")
    save_csv(p_train, split_path / "p_train.csv")
    save_csv(X_test, split_path / "X_test.csv") 
    save_csv(y_test, split_path / "y_test.csv")
    save_csv(p_test, split_path / "p_test.csv")

    return {
        "X_train" : X_train,
        "y_train" : y_train,
        "p_train" : p_train,
        "X_test" : X_test,
        "y_test" : y_test,
        "p_test" : p_test
    }

if __name__ == "__main__" :
    main()
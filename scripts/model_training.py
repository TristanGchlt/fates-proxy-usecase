import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_csv, read_config
from src.models.training import train
from src.models.utils import predict
from src.metrics.utils import compute_measures

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
SPLIT_PATH = PROJECT_ROOT / "data" / "split"


def main(
        config_path = CONFIG_PATH,
        split_path = SPLIT_PATH,
        ) :
    
    config = read_config(config_path)
    model_type = config['model_type']
    hyperparameters = config['model_hyperparameters']
    metrics = config['metrics']
    
    data = {}

    for name in ["X_train", 
                 "y_train", 
                 "p_train",
                 "X_test",
                 "y_test",
                 "p_test"] :
        data[name] = load_csv(split_path / f"{name}.csv")

    model = train(model_type, data, hyperparameters)

    data['y_pred'] = predict(model, data)

    measures = compute_measures(data, metrics)

    return {
        "model" : model,
        "measures" : measures,
        "hyperparameters" : hyperparameters
    }

if __name__ == "__main__" :
    main()


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_csv
from src.models.training import train
from src.models.utils import save_model, predict
from src.metrics.utils import compute_measures, save_measures

SPLIT_PATH = PROJECT_ROOT / "data" / "split"

# To be moved to a config file
MODEL_NAME = "RFC_no_fairness"
MODEL_TYPE = "Random Forest Classifier"
MODEL_HYPERPARAMETERS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 1
}
METRICS = []

MODEL_PATH = PROJECT_ROOT / "models" / MODEL_NAME


def main(
        model_type=MODEL_TYPE, 
        hyperparameters=MODEL_HYPERPARAMETERS,
        split_path=SPLIT_PATH,
        model_path=MODEL_PATH,
        metrics=METRICS
        ) :

    X_train = load_csv(split_path / "X_train.csv")
    y_train = load_csv(split_path / "y_train.csv")
    p_train = load_csv(split_path / "p_train.csv")

    X_test = load_csv(split_path / "X_test.csv")
    y_test = load_csv(split_path / "y_test.csv")
    p_test = load_csv(split_path / "p_test.csv")

    data = {
        'X_train' : X_train,
        'y_train' : y_train,
        'p_train' : p_train,
        'X_test' : X_test,
        'y_test' : y_test,
        'p_test' : p_test
    }

    model = train(model_type, data, hyperparameters)

    data['y_pred'] = predict(model, data)

    save_model(model, model_type, model_path)

    measures = compute_measures(model, data, metrics)

    save_measures(measures, model_path)

    return model

if __name__ == "__main__" :
    main()


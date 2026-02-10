import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.model.training import train

MODEL_TYPE = "Random Forest Classifier"
MODEL_HYPERPARAMETERS = {
    "n_estimators": 100.0,
    "max_depth": 10.0,
    "random_state": 1.0
}


def main(model_type=MODEL_TYPE) :
    data = {
        'X_train' : None,
        'y_train' : None,
        'p_train' : None,
        'X_test' : None,
        'y_test' : None,
        'p_test' : None
    }
    hyperparameters = {
        'a' : None
    }
    model = train(model_type, data, hyperparameters)
    return model

if __name__ == "__main__" :
    main()
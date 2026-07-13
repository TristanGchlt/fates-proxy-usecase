import sys
from pathlib import Path
import mlflow
import mlflow.sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

from src.data.data_cleaning import clean
from src.data.data_preparation import split
from src.models.training import train
from src.models.utils import predict
from src.metrics.utils import compute_measures
from src.utils import read_config, clean_folder, load_csv
from src.models.utils import save_model, save_model_type

RAW_DATA_FILE = PROJECT_ROOT / "data" / "raw" / "data.csv"

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

MODEL_PATH = PROJECT_ROOT / "model"

def run_pipeline() :

    # Préparation du tracking
    mlflow.set_experiment("my_experiment")

    # Récupération des paramètres
    config = read_config(CONFIG_PATH)
    model_name = config['model_name']

    # Chargement des données brutes
    dataset = load_csv(RAW_DATA_FILE)

    with mlflow.start_run(run_name=model_name):

        ####
        # Nettoyage et préparation des données
        ####

        # Nettoyage
        clean_dataset = clean(dataset)
        
        # Préparation des données
        X_train, y_train, p_train, X_test, y_test, p_test = split(clean_dataset, 
                                                              test_size = config['test_size'], 
                                                              protected_feature = config['protected_feature'], 
                                                              split_strategy = config['f_split_strategy'], 
                                                              seed = config['split_random_seed'], 
                                                              target_feature = config['target_feature'], 
                                                              balance_strategy = config['f_balance'], 
                                                              balance_seed = config['f_balance_seed'], 
                                                              hide_protected = config["f_hide_protected"],
                                                              hide_proxies = config['f_hide_proxies'])
        

        # Suivi de la config dans mlflow
        split_logs = {
            "test_size" : config['test_size'],
            "protected_feature" : config['protected_feature'], 
            "split_strategy" : config['f_split_strategy'], 
            "seed" : config['split_random_seed'], 
            "target_feature" : config['target_feature'], 
            "balance_strategy" : config['f_balance'], 
            "balance_seed" : config['f_balance_seed'], 
            "hide_protected" : config["f_hide_protected"]
        }
        for key, value in split_logs.items():
            mlflow.log_param(key, value)

        ####
        # Entrainement et évaluation du modèle
        ####

        # Entrainement
        model_type = config['model_type']
        hyperparameters = config['model_hyperparameters']
        model = train(model_type, X_train, y_train, hyperparameters)
        
        # prediction
        y_pred = predict(model, model_type, X_test)

        # evaluation
        metrics = config['metrics']
        data = {
            "y_test" : y_test,
            "y_pred" : y_pred,
            "p_test" : p_test
        }
        measures = compute_measures(data, metrics)
        
        # Suivi des paramètres du modèles dans mlflow
        mlflow.log_params(hyperparameters)

        # Evaluation du modele et suivi dans mlflow
        for metric, value in measures.items():
            mlflow.log_metric(metric, value)

        # Suivi du modele lui même dans mlflow
        mlflow.sklearn.log_model(model, 
                                 name="model")
        
    ####
    # Replacement du modèle principal, celui évalué en intégration continue
    ####

    clean_folder(MODEL_PATH)
    save_model(model, model_type, MODEL_PATH)
    save_model_type(model_type, MODEL_PATH)
    
    return 0

if __name__ == "__main__" :
    run_pipeline()
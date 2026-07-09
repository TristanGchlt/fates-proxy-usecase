import sys
from pathlib import Path
import mlflow
import mlflow.sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

from scripts.clean_data import main as clean
from scripts.prepare_data import main as split
from scripts.model_training import main as train
from src.utils import read_config, clean_folder
from src.models.utils import save_model, save_model_type

RAW_DATA_FILE = PROJECT_ROOT / "data" / "raw" / "data.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
SPLIT_PATH = PROJECT_ROOT / "data" / "split"

PROD_PATH = PROJECT_ROOT / "model"

def run_pipeline() :

    # Préparation du tracking
    mlflow.set_experiment("my_experiment")

    # Récupération des paramètres
    config = read_config(CONFIG_PATH)
    model_name = config['model_name']

    with mlflow.start_run(run_name=model_name):

        ####
        # Nettoyage et préparation des données
        ####

        # Nettoyage : Adresse du fichier des données brut et écrit le fichier de données nettoyées.
        clean(raw_data_file=RAW_DATA_FILE,
              output_path=CLEAN_DATA_PATH)
        
        # Préparation des données : Récupère le fichier des données nettoyées, applique les transformations selon la config, écrit le fichier des données splitées.
        split_logs = split(config_path=CONFIG_PATH,
                            processed_data_path=CLEAN_DATA_PATH,
                            split_path=SPLIT_PATH)
        
        # Suivi de la config dans mlflow
        for key, value in split_logs.items():
            mlflow.log_param(key, value)

        ####
        # Entrainement et évaluation du modèle
        ####
        
        # Récupère les données split et entraine le modele
        model_logs = train(config_path=CONFIG_PATH,
                            split_path=SPLIT_PATH)
        
        # Suivi des paramètres du modèles dans mlflow
        mlflow.log_params(model_logs['hyperparameters'])

        # Evaluation du modele et suivi dans mlflow
        for metric, value in model_logs["measures"].items():
            mlflow.log_metric(metric, value)

        # Suivi du modele lui même dans mlflow
        mlflow.sklearn.log_model(model_logs['model'], 
                                 name="model")
        
    ####
    # Replacement du modèle principal, celui évalué en intégration continue
    ####
        
    if config['prod'] :
        model = model_logs['model']
        model_type = config['model_type']
        clean_folder(PROD_PATH)
        save_model(model, model_type, PROD_PATH)
        save_model_type(model_type, PROD_PATH)
        
    return model_logs

if __name__ == "__main__" :
    run_pipeline()
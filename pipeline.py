import sys
from pathlib import Path
import mlflow
import mlflow.sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

from scripts.clean_and_prepare_data import main as clean
from scripts.split_data import main as split
from scripts.model_training import main as train
from src.utils import read_config, clean_folder
from src.models.utils import save_model, save_model_type

RAW_DATA_FILE = PROJECT_ROOT / "data" / "raw" / "data.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"

DATA_CONFIG_PATH = PROJECT_ROOT / "config" / "data_config.yaml"
SPLIT_PATH = PROJECT_ROOT / "data" / "split"

MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.yaml"

PROD_PATH = PROJECT_ROOT / "model"

def run_pipeline() :
    mlflow.set_experiment("my_experiment")

    config = read_config(MODEL_CONFIG_PATH)
    model_name = config['model_name']

    with mlflow.start_run(run_name=model_name):

        clean(raw_data_file=RAW_DATA_FILE,
              output_path=CLEAN_DATA_PATH)
        
        split_logs = split(config_path=DATA_CONFIG_PATH,
                            processed_data_path=CLEAN_DATA_PATH,
                            split_path=SPLIT_PATH)
        
        for key, value in split_logs.items():
            mlflow.log_param(key, value)
        
        model_logs = train(config_path=MODEL_CONFIG_PATH,
                            split_path=SPLIT_PATH)
        
        mlflow.log_params(model_logs['hyperparameters'])
        for metric, value in model_logs["measures"].items():
            mlflow.log_metric(metric, value)
        mlflow.sklearn.log_model(model_logs['model'], 
                                 name="model")
        
    if config['prod'] :
        model = model_logs['model']
        model_type = config['model_type']
        clean_folder(PROD_PATH)
        save_model(model, model_type, PROD_PATH)
        save_model_type(model_type, PROD_PATH)
        
    return model_logs

if __name__ == "__main__" :
    run_pipeline()
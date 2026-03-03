install_dependencies:
	poetry install --no-root

preprocess:
	poetry run python scripts/clean_and_prepare_data.py

split:
	poetry run python scripts/split_data.py

train:
	poetry run python scripts/model_training.py

pipeline:
	poetry run python pipeline.py

mlflow:
	poetry run mlflow ui
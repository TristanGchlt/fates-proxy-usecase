install_dependencies:
	poetry install --no-root

preprocess:
	poetry run python scripts/clean_and_prepare_data.py

split:
	poetry run python scripts/split_data.py
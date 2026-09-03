.PHONY: setup repos fetch encode train eval stream all

setup:
	pip install -r requirements.txt

repos:
	python scripts/00_create_repos.py

fetch:
	python scripts/01_fetch_data.py

encode:
	python scripts/02_encode_data.py

train:
	python scripts/03_train.py

train-multi:
	accelerate launch scripts/03_train.py

eval:
	python scripts/04_evaluate.py --push

stream:
	python scripts/05_stream_demo.py --model-dir $(MODEL) --wav $(WAV) --lang $(or $(LANG),auto)

# full data->model pipeline (single GPU)
all: repos fetch encode train eval

.PHONY: setup repos fetch fetch-hub encode encode-hub train train-hub train-multi eval stream all

setup:
	pip install -r requirements.txt

repos:
	python scripts/00_create_repos.py

fetch:
	python scripts/01_fetch_data.py

# CPU VM (300 GB disk): stream sources, push `audio`, delete local shards
fetch-hub:
	python scripts/01_fetch_data.py --hub-only

encode:
	python scripts/02_encode_data.py

# GPU VM: pull `audio` from Hub if local is missing (or pass --from-hub)
encode-hub:
	python scripts/02_encode_data.py --from-hub

train:
	python scripts/03_train.py

train-hub:
	python scripts/03_train.py --from-hub

train-multi:
	accelerate launch scripts/03_train.py

eval:
	python scripts/04_evaluate.py --push

stream:
	python scripts/05_stream_demo.py --model-dir $(MODEL) --wav $(WAV) --lang $(or $(LANG),auto)

# full data->model pipeline (single GPU)
all: repos fetch encode train eval

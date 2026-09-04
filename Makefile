.PHONY: setup repos fetch fetch-status fetch-hub encode encode-hub compact train train-hub train-multi eval stream all

setup:
	pip install -r requirements.txt

repos:
	python scripts/00_create_repos.py

fetch:
	python scripts/01_fetch_data.py

fetch-status:
	python scripts/01_fetch_data.py --status

# CPU VM: upload leftover local shards, then fetch; delete each shard after Hub upload
fetch-hub:
	python scripts/01_fetch_data.py --hub-only

# download only, ZERO Hub commits (never rate-limited); parquet collects in pending/
fetch-defer:
	python scripts/01_fetch_data.py --defer-upload

# pack pending/ into bunches and upload
upload:
	python scripts/01_fetch_data.py --upload-only

encode:
	python scripts/02_encode_data.py

# GPU VM: pull `audio` from Hub if local is missing (or pass --from-hub)
encode-hub:
	python scripts/02_encode_data.py --from-hub

# pack 200 tiny Hub parquet files into 1 bunch (fixes 429 Too Many Requests)
compact:
	python scripts/06_compact_hub.py

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

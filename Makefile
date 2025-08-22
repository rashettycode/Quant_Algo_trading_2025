.PHONY: setup run data report lint test clean format

setup:
	python -m pip install -r requirements.txt

# Unified pipeline: data + features + model + predictions + sims
data:
	python -m scripts.run_pipeline --config configs/base.yaml --k 5 --file-mode

# End-to-end pipeline entrypoint
run:
	python -m scripts.run_pipeline --config configs/base.yaml --k 5

# Generate summaries + plots from all backtests
report:
	python -m scripts.report

lint:
	flake8 src/ scripts/
	black --check src/ scripts/

format:
	black src/ scripts/

test:
	pytest -q

clean:
	rm -rf outputs/* data/interim/* data/processed/*

tune:
	python -m scripts.tune_dt --config configs/base.yaml


# ---------------------------
# Sample data & data syncing
# ---------------------------
# Generate tiny, deterministic sample Parquet files in data/sample/
sample-data:
	python -m scripts.make_sample_data

# Generate samples AND mirror them to data/processed/{prices.parquet,features.parquet}
# (won't overwrite existing files unless you use 'sample-to-processed-force')
sample-to-processed:
	python -m scripts.make_sample_data --mirror-processed

# Same as above, but overwrite processed files if they already exist
sample-to-processed-force:
	python -m scripts.make_sample_data --mirror-processed --overwrite

# Download big real Parquet(s) from Google Drive as configured in configs/base.yaml
pull-processed:
	python -m scripts.pull_data_from_drive --config configs/base.yaml --which processed

pull-features:
	python -m scripts.pull_data_from_drive --config configs/base.yaml --which features

pull-all:
	$(MAKE) pull-processed
	$(MAKE) pull-features

make pull-processed    # downloads prices.parquet → data/processed/prices.parquet
make pull-features     # downloads features.parquet → data/processed/features.parquet
make pull-all          # both


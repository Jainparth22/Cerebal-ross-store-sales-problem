.PHONY: install test test-verbose pipeline-test pipeline-predict run-api docker-build docker-up clean help

help:
	@echo "Solari Demand Forecasting Engine - Available Commands:"
	@echo "  make install         Install all dependencies from requirements.txt"
	@echo "  make test            Execute the complete unit test suite (13 tests)"
	@echo "  make pipeline-test   Run fast end-to-end integration test on 2% sample"
	@echo "  make pipeline-predict Run full production inference pipeline"
	@echo "  make run-api         Start FastAPI production server on http://localhost:8000"
	@echo "  make docker-build    Build Docker container image"
	@echo "  make docker-up       Run forecasting service with Docker Compose"
	@echo "  make clean           Remove temporary cache files and artifacts"

install:
	pip install -r requirements.txt

test:
	python -m unittest discover tests -v

pipeline-test:
	python -m src.pipeline --mode predict --sample-frac 0.02

pipeline-predict:
	python -m src.pipeline --mode predict

run-api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t solari-demand-forecasting:latest .

docker-up:
	docker-compose up -d

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "catboost_info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

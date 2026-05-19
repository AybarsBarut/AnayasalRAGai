.PHONY: install run test lint format eval docker-build docker-run compose-up

install:
	python -m pip install -r requirements.txt

run:
	uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000

test:
	pytest --cov=backend tests/

lint:
	flake8 . --max-line-length=120
	mypy backend/

format:
	black .

eval:
	python scripts/evaluate_queries.py

docker-build:
	docker build -t anayasa-ai .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env anayasa-ai

compose-up:
	docker compose up --build

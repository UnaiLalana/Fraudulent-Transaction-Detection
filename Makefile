install:
	pip install uv &&\
	uv sync

test:
	uv run python -m pytest tests/ -vv --cov=src --cov=api

format:	
	uv run black src/*.py api/*.py

lint:
	uv run pylint --disable=R,C --ignore-patterns=test_.*\.py src/*.py api/*.py 

refactor: format lint

train:
	uv run src/train_CICD.py

all: install format lint test train
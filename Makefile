# Makefile for stt-api-test-10
# Simple & tydlig

PORT ?= 8000

.PHONY: install run dev clean fmt

install:
	uv pip install --upgrade pip
	uv pip install -r requirements.txt

run:
	uvicorn app.main:app --host 0.0.0.0 --port $(PORT)

dev: run

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +; \
	rm -rf .pytest_cache; \
	rm -rf dist build; \
	rm -f *.pyc *.pyo

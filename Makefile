.PHONY: install lint format typecheck test test-unit clean benchmark-pdfs eval build run-api run-ui

PYTHON ?= python

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy --strict src scripts evals

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

benchmark-pdfs:
	$(PYTHON) scripts/build_benchmark_pdfs.py

eval:
	$(PYTHON) -m evals.run_multimodal_benchmark --output evals/results/benchmark_report.json

build:
	$(PYTHON) -m pip wheel . --no-deps -w dist

# These targets become runnable in Phases 5 and 6.
run-api:
	uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8001

run-ui:
	chainlit run app.py -w

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

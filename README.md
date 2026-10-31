# OmniDoc-Vision

Multimodal document intelligence for complex engineering PDFs: Docling layout
and table reconstruction, visual figure captioning, hybrid dense/BM25 retrieval,
cross-encoder reranking, and adaptive routing between text and vision-language
generation.

## Status

Phase 1 provides the project scaffold, runtime settings, structured logging,
multimodal schemas, and a deterministic synthetic benchmark corpus. The ingestion,
retrieval, graph, API, and Chainlit layers are added in later phases.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

On Windows PowerShell, use `.venv\Scripts\Activate.ps1` instead.

Generate the three benchmark PDFs and the 30-question bilingual golden dataset:

```bash
python scripts/build_benchmark_pdfs.py
```

The PDFs are written to `data/raw_pdfs/` and the dataset to
`data/golden_multimodal_dataset.json`. PDF output is deliberately excluded from
version control so the corpus remains reproducible from the generator.

## Quality gates

```bash
pytest tests/ -q
ruff check .
mypy --strict src scripts
python -m pip wheel . --no-deps -w dist
```

The vLLM service is optional for these checks; offline paths use deterministic
fallback behavior in the later ingestion and generation phases.

## Configuration

Copy `.env.example` to `.env` and adjust paths, model endpoints, and security
bounds. Relative data paths are resolved against the project root. The FastAPI
service defaults to port `8001`; the local vLLM endpoint defaults to port `8000`.

## Repository layout

```text
src/common/       settings, logging, exceptions
src/schemas/      multimodal and API contracts
scripts/          deterministic benchmark generation
data/             generated PDFs and golden dataset
docs/             product, architecture, database, API, security, and testing docs
tests/unit/       unit tests
tests/integration/ integration tests
```

## License

Apache-2.0

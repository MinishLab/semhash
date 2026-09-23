clean:


venv:
	uv venv

install: venv
	uv sync --all-extras
	uv run pre-commit install

install-no-pre-commit:
	uv pip install ".[dev,all]"

fix:
	uv run pre-commit run --all-files

test:
	uv run pytest --cov=semhash --cov-report=term-missing

benchmark-text:
	uv run python -m benchmarks.run_text_benchmarks

benchmark-image:
	uv run python -m benchmarks.run_image_benchmarks

benchmark-lexical:
	uv run --with datasets --with datasketch python -m benchmarks.run_lexical_benchmarks

benchmark: benchmark-text benchmark-image benchmark-lexical

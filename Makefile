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

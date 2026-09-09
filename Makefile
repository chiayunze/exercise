fmt:
	uv run ruff check --select I --fix .
	uv run ruff format .
run:
	uv run src/main.py
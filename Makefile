include src/activation_steering_study/Makefile

.PHONY: typecheck mutation

typecheck:
	uv run --locked mypy src tests

mutation:
	uv run --locked mutmut run --max-children 1

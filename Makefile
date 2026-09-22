include src/activation_steering_study/Makefile

.PHONY: typecheck

typecheck:
	uv run --locked mypy src tests

include src/activation_steering_study/Makefile

.PHONY: typecheck mutation test

typecheck:
	uv run --locked mypy src tests scripts/report_tables.py scripts/project_run_record.py scripts/run_condition.py

mutation:
	uv run --locked mutmut run --max-children 1

test:
	uv run --locked pytest src tests

.PHONY: results figures validate-results

results:
	uv run --locked python scripts/report_tables.py

figures:
	uv run --script --locked scripts/make_figures.py

validate-results:
	uv run --locked python scripts/report_tables.py --check-only

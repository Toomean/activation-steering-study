.PHONY: smoke capture intervention generation steer test-steering scoring sample-refusal refusal analyze-refusal refusal-random

smoke:
	uv run --locked python -m activation_steering_study.smoke

capture:
	uv run --locked pytest -s tests/test_capture.py

intervention:
	uv run --locked pytest -s tests/test_intervention.py

generation:
	uv run --locked pytest -s tests/test_generation.py

steer:
	uv run --locked python -m activation_steering_study.steering.generation

test-steering:
	uv run --locked pytest -s tests/test_steered_generation.py

scoring:
	uv run --locked pytest -s tests/test_scoring.py

sample-refusal:
	uv run --locked python -m activation_steering_study.data.refusal

refusal:
	uv run --locked python -m activation_steering_study.steering.refusal

analyze-refusal:
	uv run --locked python -m activation_steering_study.analysis.refusal

refusal-random:
	uv run --locked python -m activation_steering_study.steering.random_control

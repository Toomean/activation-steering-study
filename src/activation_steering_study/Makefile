.PHONY: smoke capture intervention

smoke:
	uv run --locked python -m activation_steering_study.smoke

capture:
	uv run --locked pytest -s tests/test_capture.py

intervention:
	uv run --locked pytest -s tests/test_intervention.py

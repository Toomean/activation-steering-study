.PHONY: smoke capture

smoke:
	uv run --locked python -m activation_steering_study.smoke

capture:
	uv run --locked pytest -s tests/test_capture.py

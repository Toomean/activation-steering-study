.PHONY: smoke capture intervention generation steer test-steering scoring sample-refusal refusal analyze-refusal refusal-random sycophancy-baseline sycophancy-direction sycophancy-pilot sycophancy-norm-matched refusal-mmlu sycophancy-sweep honesty-sweep refusal-domain-extract refusal-domain-baseline

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
	uv run --locked pytest -s src/activation_steering_study/steering/test_generation.py

scoring:
	uv run --locked pytest -s src/activation_steering_study/evaluation/test_scoring.py

sample-refusal:
	uv run --locked python -m activation_steering_study.data.refusal

refusal:
	uv run --locked python -m activation_steering_study.steering.refusal

analyze-refusal:
	uv run --locked python -m activation_steering_study.analysis.refusal

refusal-random:
	uv run --locked python -m activation_steering_study.steering.random_control

sycophancy-baseline:
	uv run --locked python -m activation_steering_study.evaluation.sycophancy

sycophancy-direction:
	uv run --locked python -m activation_steering_study.extraction.sycophancy

sycophancy-pilot:
	uv run --locked python -m activation_steering_study.steering.sycophancy

sycophancy-norm-matched:
	uv run --locked python -m activation_steering_study.steering.sycophancy --match-refusal-norm

refusal-mmlu:
	uv run --locked python -m activation_steering_study.evaluation.mmlu

sycophancy-sweep:
	uv run --locked python -m activation_steering_study.steering.choice_sweep --behaviour sycophancy

honesty-sweep:
	uv run --locked python -m activation_steering_study.steering.choice_sweep --behaviour honesty

refusal-domain-extract:
	uv run --locked python -m activation_steering_study.steering.refusal_domain extract

refusal-domain-baseline:
	uv run --locked python -m activation_steering_study.steering.refusal_domain run --control baseline

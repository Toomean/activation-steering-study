import pytest

from activation_steering_study.analysis.bootstrap import bootstrap_mean_interval


def test_bootstrap_returns_expected_mean_interval() -> None:
    interval = bootstrap_mean_interval([-1.0, 0.0, 0.0, 1.0])

    assert interval == pytest.approx((-0.75, 0.75)), (
        "symmetric signed changes should produce the known BCa interval"
    )


@pytest.mark.parametrize("values", ([0.0, 0.0], [2.0, 2.0, 2.0], [-1.0, -1.0]))
def test_constant_changes_have_no_bca_interval(values: list[float]) -> None:
    assert bootstrap_mean_interval(values) is None, (
        "constant changes cannot produce a BCa interval"
    )


@pytest.mark.parametrize("values", ([], [1.0]))
def test_bootstrap_requires_two_observations(values: list[float]) -> None:
    with pytest.raises(
        ValueError,
        match="each sample in `data` must contain two or more observations",
    ):
        bootstrap_mean_interval(values)

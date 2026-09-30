"""Check geometry values against vectors with hand-computable directions."""

import re
from typing import Any, cast

import numpy as np
import pytest
import torch
from scipy.spatial.distance import cosine as cosine_distance

from activation_steering_study.analysis.geometry import (
    _cosine,
    _split_half_geometry,
    _summary,
    analyze_geometry,
)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ([0.0, 0.0], [1.0, 0.0]),
        ([1.0, 0.0], [0.0, 0.0]),
        ([0.0, 0.0], [0.0, 0.0]),
    ],
)
def test_zero_vector_cosine_is_undefined(left: list[float], right: list[float]) -> None:
    left_array = np.asarray(left)
    right_array = np.asarray(right)

    with pytest.warns(RuntimeWarning):
        assert np.isnan(cosine_distance(left_array, right_array)), (
            "SciPy reports zero-vector cosine as NaN"
        )
    assert _cosine(left_array, right_array) is None, (
        "geometry represents undefined zero-vector cosine as None"
    )


def test_summary_counts_none_and_interpolates_defined_percentiles() -> None:
    # Linear percentiles over the defined values [0, 1] are 0.05 and 0.95; the median is 0.5.
    assert _summary([1.0, None, 0.0]) == pytest.approx(
        {"median": 0.5, "p05": 0.05, "p95": 0.95, "undefined_count": 1}, abs=1e-12
    ), "None should be counted as undefined and excluded from the median and percentiles"


def test_geometry_centering_removes_a_shared_translation() -> None:
    harmless = torch.zeros((32, 2))
    harmful = {
        "east": torch.tensor([[1.0, 0.0]]).repeat(24, 1),
        "north": torch.tensor([[0.0, 1.0]]).repeat(24, 1),
        "west": torch.tensor([[-1.0, 0.0]]).repeat(24, 1),
    }
    original = cast(dict[str, Any], analyze_geometry(harmful, harmless))
    common_shift = torch.tensor([3.0, 2.0])
    shifted = cast(
        dict[str, Any],
        analyze_geometry(
            {name: rows + common_shift for name, rows in harmful.items()}, harmless
        ),
    )

    assert original["raw_cosine_matrix"][0][1] == 0.0
    assert shifted["raw_cosine_matrix"][0][1] != original["raw_cosine_matrix"][0][1]
    torch.testing.assert_close(
        torch.tensor(shifted["centered_cosine_matrix"]),
        torch.tensor(original["centered_cosine_matrix"]),
    )
    # Mean (0, 1/3) leaves residuals (1, -1/3), (0, 2/3), (-1, -1/3) with norms sqrt(10)/3, 2/3,
    # sqrt(10)/3; the unequal residual norms make these expected values non-degenerate.
    expected_centered = np.array([
        [1.0, -1 / np.sqrt(10), -0.8],
        [-1 / np.sqrt(10), 1.0, -1 / np.sqrt(10)],
        [-0.8, -1 / np.sqrt(10), 1.0],
    ])
    assert np.asarray(original["centered_cosine_matrix"]) == pytest.approx(
        expected_centered, abs=1e-12
    ), "centered cosines should divide each residual dot product by both residual norms"
    assert original["counts"] == {"harmful_per_domain": 24, "shared_harmless": 32}
    assert original["split_halves"]["replicates"] == 100
    assert original["split_halves"]["harmful_per_domain_per_half"] == 12
    assert original["split_halves"]["shared_harmless_per_half"] == 16
    assert original["split_halves"]["shared_harmless_partition_across_domains"] is True
    expected_cross = {"east|north": 0.0, "east|west": -1.0, "north|west": 0.0}
    for pair, expected in expected_cross.items():
        summary = original["split_halves"]["cross_domain_raw"][pair]
        assert summary == {
            "median": expected, "p05": expected, "p95": expected, "undefined_count": 0,
        }, (
            f"{pair} should match its constant-domain cosine in every partition"
        )
    assert original["domain_allocation_null"]["replicates"] == 199
    assert original["domain_allocation_null"]["pooled_harmful_rows"] == 72
    assert original["domain_allocation_null"]["allocation_per_domain"] == 24
    assert original["domain_allocation_null"]["retains_shared_harmless_mean"] is True


def test_zero_directions_are_undefined_and_counted() -> None:
    harmless = torch.ones((32, 2))
    harmful = {
        "one": torch.ones((24, 2)),
        "two": torch.ones((24, 2)),
        "three": torch.ones((24, 2)),
    }
    result = cast(dict[str, Any], analyze_geometry(harmful, harmless))

    assert result["raw_cosine_matrix"] == [[None] * 3 for _ in range(3)]
    assert result["raw_matrix_undefined_count"] == 9
    for summary in result["split_halves"]["within_domain_raw"].values():
        assert summary["median"] is None
        assert summary["undefined_count"] == 100
    for pair, summary in result["split_halves"]["cross_domain_raw"].items():
        assert summary["median"] is None, f"{pair} has no defined opposite-half cosine"
        assert summary["undefined_count"] == 100, f"{pair} must count each undefined partition once"


def test_disjoint_basis_rows_keep_half_sample_cosines_separate() -> None:
    rows = torch.eye(104, dtype=torch.float64)
    harmful = {
        name: rows[start : start + 24]
        for name, start in zip(("east", "north", "west"), (0, 24, 48))
    }
    result = cast(dict[str, Any], analyze_geometry(harmful, rows[72:]))

    for left in range(3):
        for right in range(left):
            assert abs(result["raw_cosine_matrix"][left][right] - 3 / 7) < 1e-12, (
                "shared harmless background should give full-data raw cosine 3/7"
            )
            assert abs(result["centered_cosine_matrix"][left][right] + 0.5) < 1e-12, (
                "three centered orthogonal domain means should have cosine -1/2"
            )
    for group in ("within_domain_raw", "cross_domain_raw"):
        for name, summary in result["split_halves"][group].items():
            assert summary == {
                "median": 0.0, "p05": 0.0, "p95": 0.0, "undefined_count": 0,
            }, (
                f"{group}[{name}] should compare disjoint opposite halves"
            )
    # Any allocation gives 1/24 on 24 own and -1/32 on 32 shared coordinates: dot 32 * (1/32)^2
    # over squared norm 1/24 + 1/32 = 7/96 is 3/7; three such directions center to cosine -1/2.
    for group, expected in (("raw_cross_cosines", 3 / 7), ("centered_cross_cosines", -0.5)):
        for pair, summary in result["domain_allocation_null"][group].items():
            assert summary == pytest.approx(
                {"median": expected, "p05": expected, "p95": expected, "undefined_count": 0},
                abs=1e-12,
            ), f"{group}[{pair}] should be identical for every allocation of one-hot rows"


def test_cross_domain_raw_averages_both_half_orientations() -> None:
    harmful = [
        np.tile([1.0, 0.0], (24, 1)),
        np.concatenate((np.tile([1.0, 1.0], (12, 1)), np.tile([-1.0, 1.0], (12, 1)))),
        np.tile([0.0, -1.0], (24, 1)),
    ]
    result = _split_half_geometry(
        ["east", "middle", "south"], harmful, np.zeros((32, 2)), np.random.default_rng(42)
    )

    summary = result["cross_domain_raw"]["east|middle"]
    assert (summary["median"], summary["p05"], summary["p95"]) == pytest.approx(
        (0.0, 0.0, 0.0), abs=1e-15
    ), "both orientations should cancel apart from distance-conversion roundoff"
    assert summary["undefined_count"] == 0, "all orientations have nonzero directions"


def test_cross_domain_raw_requires_both_orientations_to_be_defined() -> None:
    harmful = [
        np.tile([1.0, 0.0], (24, 1)),
        np.concatenate((np.zeros((23, 2)), [[0.0, 1.0]])),
        np.tile([-1.0, 0.0], (24, 1)),
    ]
    result = _split_half_geometry(
        ["east", "sparse", "west"], harmful, np.zeros((32, 2)), np.random.default_rng(42)
    )

    assert result["cross_domain_raw"]["east|sparse"] == {
        "median": None, "p05": None, "p95": None, "undefined_count": 100,
    }, "one undefined orientation makes the pair undefined in every partition"


def test_split_half_geometry_advances_supplied_rng() -> None:
    rng = np.random.default_rng(42)
    initial_state = rng.bit_generator.state
    harmful = [np.tile(row, (24, 1)) for row in ([1.0, 0.0], [0.0, 1.0], [-1.0, 0.0])]

    _split_half_geometry(["east", "north", "west"], harmful, np.zeros((32, 2)), rng)

    assert rng.bit_generator.state != initial_state, "split halves should consume the caller's RNG"


def test_split_half_geometry_uses_twelve_harmful_rows_per_half() -> None:
    harmful = [np.concatenate((np.tile([1.0, 0.0], (23, 1)), [[1.0, 12.0]]))] * 3
    result = _split_half_geometry(
        ["one", "two", "three"], harmful, np.zeros((32, 2)), np.random.default_rng(42)
    )

    # With 12 rows per half, the (1, 12) row gives half means (1, 1) and (1, 0): cosine 1/sqrt(2).
    expected = 1 / np.sqrt(2)
    for name, summary in result["within_domain_raw"].items():
        assert summary == pytest.approx(
            {"median": expected, "p05": expected, "p95": expected, "undefined_count": 0},
            abs=1e-12,
        ), f"{name} should use 12 harmful rows per half; 11 or 13 would change 1/sqrt(2)"


def test_split_half_geometry_uses_sixteen_harmless_rows_per_half() -> None:
    harmful = [np.tile([1.0, 0.0], (24, 1))] * 3
    harmless = np.concatenate((np.zeros((31, 2)), [[0.0, 16.0]]))
    result = _split_half_geometry(
        ["one", "two", "three"], harmful, harmless, np.random.default_rng(42)
    )

    # With 16 rows per half, harmless means (0, 1) and (0, 0) give directions (1, -1) and (1, 0).
    expected = 1 / np.sqrt(2)
    for group in (result["within_domain_raw"], result["cross_domain_raw"]):
        for name, summary in group.items():
            assert summary == pytest.approx(
                {"median": expected, "p05": expected, "p95": expected, "undefined_count": 0},
                abs=1e-12,
            ), f"{name} should use 16 harmless rows per half; 15 or 17 would change 1/sqrt(2)"


def _nonconstant_activations() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Return rows that vary within each domain, so different partitions give different cosines."""
    x = torch.arange(24, dtype=torch.float64)
    y = torch.arange(32, dtype=torch.float64)
    harmful = {
        "east": torch.stack((2 + 0.07 * x, 0.2 * torch.sin(x), 0.1 * (x % 3)), dim=1),
        "north": torch.stack((0.1 * torch.cos(x), 2 + 0.05 * x, 0.2 * (x % 4)), dim=1),
        "west": torch.stack((-2 + 0.04 * x, 0.3 * torch.sin(x / 2), 1 + 0.06 * x), dim=1),
    }
    harmless = torch.stack((0.03 * y, 0.02 * (y % 5), 0.01 * torch.sin(y)), dim=1)
    return harmful, harmless


def test_nonconstant_rows_produce_varying_seeded_partitions_and_allocations() -> None:
    """Nonconstant rows make a fixed-identity partition visibly wrong."""
    harmful, harmless = _nonconstant_activations()
    result = cast(dict[str, Any], analyze_geometry(harmful, harmless))

    for name in ("within_domain_raw", "within_domain_centered_diagnostic"):
        summary = result["split_halves"][name]["east"]
        assert summary["undefined_count"] == 0, f"{name} unexpectedly lost defined halves"
        assert summary["p95"] - summary["p05"] > 0.0001, (
            f"{name} repeated the same partition across 100 draws"
        )
    for name in ("raw_cross_cosines", "centered_cross_cosines"):
        summary = result["domain_allocation_null"][name]["east|north"]
        assert summary["undefined_count"] == 0, f"{name} unexpectedly lost defined allocations"
        assert summary["p95"] - summary["p05"] > 0.1, (
            f"{name} repeated the same harmful-domain allocation across 199 draws"
        )


def test_analyze_geometry_is_repeatable_without_global_numpy_draws() -> None:
    harmful, harmless = _nonconstant_activations()
    original_state = np.random.get_state()
    try:
        # Use a global seed distinct from geometry's local seed 42.
        np.random.seed(7)
        before = np.random.get_state()
        assert analyze_geometry(harmful, harmless) == analyze_geometry(harmful, harmless), (
            "the fixed local seed should reproduce every partition and allocation summary"
        )
        np.testing.assert_equal(
            np.random.get_state(), before, err_msg="geometry must not touch the global NumPy stream"
        )
    finally:
        np.random.set_state(original_state)


@pytest.mark.parametrize(
    ("harmful_shapes", "harmless_shape", "message"),
    [
        ([(24, 2), (24, 2)], (32, 2), "exactly three domains"),
        ([(24, 2), (23, 2), (24, 2)], (32, 2), "(24, H)"),
        ([(24, 2), (24, 2), (24, 2)], (31, 2), "(32, H)"),
        ([(24, 2), (24, 3), (24, 2)], (32, 2), "(24, H)"),
    ],
    ids=["two-domains", "domain-with-23-rows", "harmless-with-31-rows", "domain-hidden-size-3"],
)
def test_analyze_geometry_rejects_wrong_domain_count_or_shapes(
    harmful_shapes: list[tuple[int, int]], harmless_shape: tuple[int, int], message: str
) -> None:
    harmful = {f"domain_{index}": torch.zeros(shape) for index, shape in enumerate(harmful_shapes)}

    with pytest.raises(ValueError, match=re.escape(message)):
        analyze_geometry(harmful, torch.zeros(harmless_shape))

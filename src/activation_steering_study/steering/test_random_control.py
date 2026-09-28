import pytest
import torch

from activation_steering_study.steering.random_control import sample_random_direction


def test_random_direction_shape_dtype_and_norm() -> None:
    direction = sample_random_direction(7, 3.5, seed=42)

    assert direction.shape == (7,), "random direction should match hidden size"
    assert direction.device.type == "cpu", "random direction should stay on CPU"
    assert direction.dtype == torch.float32, "random direction should use float32"
    assert torch.linalg.vector_norm(direction).item() == pytest.approx(3.5), (
        "random direction should match the requested norm"
    )


def test_random_direction_seed_controls_orientation() -> None:
    first = sample_random_direction(7, 3.5, seed=42)
    repeated = sample_random_direction(7, 3.5, seed=42)
    different = sample_random_direction(7, 3.5, seed=43)

    assert torch.equal(first, repeated), "the same seed should reproduce the direction"
    assert not torch.equal(first, different), "different seeds should change the direction"


def test_random_direction_does_not_change_global_rng() -> None:
    before = torch.random.get_rng_state()
    sample_random_direction(7, 3.5, seed=42)
    after = torch.random.get_rng_state()

    assert torch.equal(before, after), "local sampling should preserve global torch RNG state"

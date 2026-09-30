"""Descriptive geometry for three harmful-domain difference-in-means vectors."""

from itertools import combinations
from typing import TypedDict

import numpy as np
import torch
from scipy.spatial.distance import cosine as cosine_distance


class CosineSummary(TypedDict):
    median: float | None
    p05: float | None
    p95: float | None
    undefined_count: int

class SplitHalfGeometry(TypedDict):
    replicates: int
    harmful_per_domain_per_half: int
    shared_harmless_per_half: int
    shared_harmless_partition_across_domains: bool
    within_domain_raw: dict[str, CosineSummary]
    cross_domain_raw: dict[str, CosineSummary]
    within_domain_centered_diagnostic: dict[str, CosineSummary]


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    if np.linalg.norm(left) == 0.0 or np.linalg.norm(right) == 0.0:
        return None
    # SciPy returns cosine distance; geometry reports similarity.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html
    return float(1 - cosine_distance(left, right))


def _cosine_matrix(directions: np.ndarray) -> tuple[list[list[float | None]], int]:
    matrix: list[list[float | None]] = []
    undefined_count = 0
    for left in directions:
        row = []
        for right in directions:
            value = _cosine(left, right)
            row.append(value)
            undefined_count += value is None
        matrix.append(row)
    return matrix, undefined_count


def _summary(values: list[float | None]) -> CosineSummary:
    defined = np.asarray([value for value in values if value is not None], dtype=float)
    if defined.size == 0:
        return {
            "median": None,
            "p05": None,
            "p95": None,
            "undefined_count": len(values),
        }
    return {
        "median": float(np.median(defined)),
        "p05": float(np.percentile(defined, 5)),
        "p95": float(np.percentile(defined, 95)),
        "undefined_count": len(values) - int(defined.size),
    }


def _center(directions: np.ndarray) -> np.ndarray:
    return directions - directions.mean(axis=0, keepdims=True)


def _split_half_geometry(
    domain_names: list[str],
    harmful_rows: list[np.ndarray],
    harmless_rows: np.ndarray,
    rng: np.random.Generator,
) -> SplitHalfGeometry:
    """Use the caller's RNG and the same harmless halves across domains in each split."""
    domain_pairs = list(combinations(range(3), 2))
    pair_names = [f"{domain_names[left]}|{domain_names[right]}" for left, right in domain_pairs]
    split_values: dict[str, list[float | None]] = {name: [] for name in domain_names}
    cross_split_values: dict[str, list[float | None]] = {pair: [] for pair in pair_names}
    centered_split_values: dict[str, list[float | None]] = {
        name: [] for name in domain_names
    }
    for _ in range(100):
        harmless_order = rng.permutation(32)
        harmless_halves = (
            harmless_rows[harmless_order[:16]].mean(axis=0),
            harmless_rows[harmless_order[16:]].mean(axis=0),
        )
        half_directions: list[list[np.ndarray]] = [[], []]
        for rows in harmful_rows:
            harmful_order = rng.permutation(24)
            for half, indices in enumerate((harmful_order[:12], harmful_order[12:])):
                half_directions[half].append(rows[indices].mean(axis=0) - harmless_halves[half])
        first, second = np.stack(half_directions[0]), np.stack(half_directions[1])
        centered_first, centered_second = _center(first), _center(second)
        for index, name in enumerate(domain_names):
            split_values[name].append(_cosine(first[index], second[index]))
            centered_split_values[name].append(
                _cosine(centered_first[index], centered_second[index])
            )
        # Match within-domain halves using disjoint harmless halves, averaging both orientations.
        for pair, (left, right) in zip(pair_names, domain_pairs):
            forward = _cosine(first[left], second[right])
            reverse = _cosine(first[right], second[left])
            cross_split_values[pair].append(
                None if forward is None or reverse is None else (forward + reverse) / 2
            )

    return {
        "replicates": 100,
        "harmful_per_domain_per_half": 12,
        "shared_harmless_per_half": 16,
        "shared_harmless_partition_across_domains": True,
        "within_domain_raw": {
            name: _summary(split_values[name]) for name in domain_names
        },
        "cross_domain_raw": {
            pair: _summary(cross_split_values[pair]) for pair in pair_names
        },
        "within_domain_centered_diagnostic": {
            name: _summary(centered_split_values[name]) for name in domain_names
        },
    }


def analyze_geometry(
    harmful_by_domain: dict[str, torch.Tensor],
    harmless: torch.Tensor,
) -> dict[str, object]:
    """Summarize raw/centered geometry, split-halves, and domain-allocation nulls.

    Inputs are saved row activations with shapes ``24 x H`` per domain and
    ``32 x H`` for the single shared harmless sample. All random partitions
    use a local NumPy generator, leaving process-wide random state untouched.
    """
    domain_names = list(harmful_by_domain)
    if len(domain_names) != 3:
        raise ValueError("Expected harmful activations for exactly three domains")
    harmful_rows = [tensor.detach().double().cpu().numpy() for tensor in harmful_by_domain.values()]
    harmless_rows = harmless.detach().double().cpu().numpy()
    hidden_size = harmless_rows.shape[1]

    if harmless_rows.shape != (32, hidden_size):
        raise ValueError(f"Expected harmless activations with shape (32, H), got {harmless_rows.shape}")
    if any(rows.shape != (24, hidden_size) for rows in harmful_rows):
        shapes = [rows.shape for rows in harmful_rows]
        raise ValueError(f"Expected three harmful activation arrays with shape (24, H), got {shapes}")

    harmless_mean = harmless_rows.mean(axis=0)
    harmful_pool = np.concatenate(harmful_rows, axis=0)
    raw_directions = np.stack([rows.mean(axis=0) - harmless_mean for rows in harmful_rows])
    centered_directions = _center(raw_directions)
    raw_matrix, raw_undefined = _cosine_matrix(raw_directions)
    centered_matrix, centered_undefined = _cosine_matrix(centered_directions)

    seed = 42
    rng = np.random.default_rng(seed)
    split_halves = _split_half_geometry(domain_names, harmful_rows, harmless_rows, rng)
    domain_pairs = list(combinations(range(3), 2))
    pair_names = [f"{domain_names[left]}|{domain_names[right]}" for left, right in domain_pairs]

    null_values: dict[str, dict[str, list[float | None]]] = {
        "raw": {pair: [] for pair in pair_names},
        "centered": {pair: [] for pair in pair_names},
    }
    for _ in range(199):
        allocation = rng.permutation(72).reshape(3, 24)
        permuted_directions = np.stack(
            [harmful_pool[indices].mean(axis=0) - harmless_mean for indices in allocation]
        )
        permuted_centered = _center(permuted_directions)
        for pair_index, (left, right) in enumerate(domain_pairs):
            pair = pair_names[pair_index]
            null_values["raw"][pair].append(
                _cosine(permuted_directions[left], permuted_directions[right])
            )
            null_values["centered"][pair].append(
                _cosine(permuted_centered[left], permuted_centered[right])
            )

    return {
        "domains": domain_names,
        "counts": {"harmful_per_domain": 24, "shared_harmless": 32},
        "rng": {"library": "NumPy default_rng", "seed": seed},
        "split_halves": split_halves,
        "raw_cosine_matrix": raw_matrix,
        "raw_matrix_undefined_count": raw_undefined,
        "centered_cosine_matrix": centered_matrix,
        "centered_matrix_undefined_count": centered_undefined,
        "centered_object": (
            "Each of the three directions minus their shared mean; the three residuals "
            "sum to zero and have rank at most two. This is not a criterion for "
            "functional distinctness."
        ),
        "domain_allocation_null": {
            "replicates": 199,
            "pooled_harmful_rows": 72,
            "allocation_per_domain": 24,
            "question": "Geometry under shuffled allocation of pooled harmful rows to domains.",
            "not_a_refusal_existence_null": True,
            "retains_shared_harmless_mean": True,
            "raw_cross_cosines": {
                pair: _summary(null_values["raw"][pair]) for pair in pair_names
            },
            "centered_cross_cosines": {
                pair: _summary(null_values["centered"][pair]) for pair in pair_names
            },
        },
        "interpretation": (
            "Descriptive partitions and a domain-allocation null only; row-level semantic "
            "dependence limits exchangeability. These are not confidence bounds, p-values, "
            "or evidence of distinct refusal mechanisms. Shared harmless background can "
            "contribute to raw cross-domain alignment. Compare cross_domain_raw with "
            "within_domain_raw at matching half-sample sizes, not with the full-data matrix."
        ),
    }

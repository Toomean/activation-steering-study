"""Check frozen refusal-domain IDs and the shared harmless pool."""

from activation_steering_study.data.refusal_domains import load_domain_extraction


def test_frozen_domain_ids_are_disjoint_and_share_harmless_pool() -> None:
    domains, harmless = load_domain_extraction()
    ids = [row["sample_index"] for rows in domains.values() for row in rows]
    assert [len(rows) for rows in domains.values()] == [24, 24, 24]
    assert len(ids) == len(set(ids)) == 72
    assert len(harmless) == 32

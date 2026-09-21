"""Bootstrap a mean interval for paired refusal changes."""

import numpy as np
from scipy.stats import bootstrap


def bootstrap_mean_interval(values: list[float]) -> tuple[float, float] | None:
    """Return a BCa mean interval, or None when its bounds are undefined."""
    # The single sample contains one signed change per paired prompt.
    # np.mean is the statistic across each resampled set of prompts.
    # Integer seed 42 makes the resampling repeatable.
    # The defaults are 95% BCa with 9,999 resamples.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
    interval = bootstrap(
        (values,),
        np.mean,
        rng=42,
    ).confidence_interval
    if np.isnan(interval.low) or np.isnan(interval.high):
        return None
    return float(interval.low), float(interval.high)

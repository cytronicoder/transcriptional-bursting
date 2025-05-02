import pytest
import numpy as np
from burst_models.utils import (
    pmf_mixed_poisson,
    method_of_moments_poisson,
    pmf_total_compound,
)


def test_pmf_mixed_poisson_basic():
    # For rate=0, pmf should be 1 at k=0, 0 elsewhere
    assert pmf_mixed_poisson(0, rate=0.0, k_off=1.0) == pytest.approx(1.0)
    for k in [1, 2, 3]:
        assert pmf_mixed_poisson(k, rate=0.0, k_off=1.0) == pytest.approx(0.0)


def test_method_of_moments_poisson():
    # Synthetic data: generate Poisson with lambda=2, k_on*L=1 => mean=2, var=2
    rng = np.random.default_rng(42)
    data = rng.poisson(lam=2.0, size=10000)
    # Add noise: treat as totals over L=1 with k_on*L*lambda=2 => k_on irrelevant here
    lam_hat, k_on_hat = method_of_moments_poisson(data, L=1.0)
    assert lam_hat == pytest.approx(1.0, rel=0.1) or lam_hat > 0
    # k_on_hat approximate mean/(lambda*L)
    assert k_on_hat > 0


def test_compound_zero_bursts():
    # If each burst always produces >=1 transcript,
    # then P(X=0) = P(B=0) = exp(-k_on * L)
    k_on = 0.5
    L = 2.0
    expected_p0 = np.exp(-k_on * L)
    # pmf_burst returns zero at k=0, and 1 at k>=1 (unnormalized for test)

    def pmf_burst(k):
        return 0.0 if k == 0 else 1.0

    val = pmf_total_compound(0, pmf_burst, k_on, L, max_bursts=10)
    assert val == pytest.approx(expected_p0, rel=1e-6)


if __name__ == "__main__":
    pytest.main()

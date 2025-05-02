import pytest
from burst_models.negbin_model import NegBinBurstModel


def test_negbin_burst_model_moments():
    alpha = 3.0
    mu_b = 2.0
    k_on = 0.5
    model = NegBinBurstModel(k_on=k_on, mu_b=mu_b, alpha=alpha)
    mean, var, index = model.first_moments(L=1.0)
    # E_N=mu_b=2, Var_N=2+2^2/3=2+4/3=10/3
    # E_B=0.5 => mean=1.0, var=0.5*(10/3) + 0.5*(4) = 5/3 + 2 = 11/3
    assert mean == pytest.approx(1.0)
    assert var == pytest.approx(11 / 3)
    assert index == pytest.approx(var / mean)


if __name__ == "__main__":
    pytest.main()

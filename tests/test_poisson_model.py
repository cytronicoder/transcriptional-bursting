import pytest
from burst_models.poisson_model import PoissonBurstModel


def test_poisson_burst_model_moments():
    model = PoissonBurstModel(k_on=0.5, k_off=2.0, r=4.0, p=0.5)
    mean, var, index = model.first_moments(L=1.0)
    # expected lambda_b = p*r/k_off = 0.5*4/2 = 1
    # E_B = 0.5*1 = 0.5, so mean=0.5*1=0.5
    assert mean == pytest.approx(0.5)
    # Var_N = 1+1^2=2; var = E_B*2 + E_B*1^2 = 0.5*2 + 0.5*1 = 1.5
    assert var == pytest.approx(1.5)
    assert index == pytest.approx(var / mean)


if __name__ == "__main__":
    pytest.main()

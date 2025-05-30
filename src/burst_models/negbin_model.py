from typing import List, Tuple
from scipy.special import comb
from .utils import pmf_total_compound


class NegBinBurstModel:
    """
    Negative Binomial + Markov model with hierarchical variability.
    """

    def __init__(self, k_on: float, mu_b: float, alpha: float) -> None:
        """
        Initialize NB bursting model.

        Parameters:
            k_on (float): Burst initiation rate
            mu_b (float): Mean burst size
            alpha (float): NB dispersion parameter
        """
        self.k_on = k_on
        self.mu_b = mu_b
        self.alpha = alpha

    def burst_size_pmf(self, k: int) -> float:
        """PMF for NB-distributed burst sizes P(N=k)"""
        p = self.alpha / (self.alpha + self.mu_b)
        result = comb(k + self.alpha - 1, k) * p**self.alpha * (1 - p) ** k
        return float(result)

    def total_transcripts_pmf(self, x: int, L: float, max_bursts: int = 100) -> float:
        """Compound Poisson PMF for total transcript count"""
        return pmf_total_compound(x, self.burst_size_pmf, self.k_on, L, max_bursts)

    def first_moments(self, L: float) -> Tuple[float, float, float]:
        """
        Return (mean, variance, burstiness index) of total counts over window L.

        Returns:
            tuple: (mean, variance, index)
        """
        E_N = self.mu_b
        Var_N = self.mu_b + self.mu_b**2 / self.alpha
        E_B = self.k_on * L
        mean = E_B * E_N
        var = E_B * Var_N + E_B * E_N**2
        index = var / mean
        return mean, var, index

    @staticmethod
    def fit_moments(data: List[int], L: float) -> Tuple[float, float, float]:
        """
        Placeholder for fitting NB model parameters from data.

        Parameters:
            data (list): Observed total counts
            L (float): Observation window length

        Returns:
            NotImplementedError
        """
        raise NotImplementedError("Moment fitting for NB model not implemented yet")

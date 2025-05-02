from .utils import pmf_mixed_poisson, method_of_moments_poisson, pmf_total_compound


class PoissonBurstModel:
    """
    Poisson + Two-State Markov model with detection thinning
    for transcriptional bursting.
    """

    def __init__(self, k_on, k_off, r, p):
        """
        Initialize model parameters.

        Parameters:
            k_on (float): Burst initiation rate
            k_off (float): Transition rate from On to Off
            r (float): Transcription rate while On
            p (float): Detection probability per transcript
        """
        self.k_on = k_on
        self.k_off = k_off
        self.r = r
        self.p = p
        self.lambda_b = p * r / k_off

    def burst_size_pmf(self, k):
        """Marginal burst-size PMF P(N=k)"""
        return pmf_mixed_poisson(k, self.p * self.r, self.k_off)

    def total_transcripts_pmf(self, x, L, max_bursts=100):
        """PMF of total observed transcripts over window length L"""
        return pmf_total_compound(x, self.burst_size_pmf, self.k_on, L, max_bursts)

    def first_moments(self, L):
        """
        Return (mean, variance, burstiness index) of total counts over window L.

        Returns:
            tuple: (mean, variance, index)
        """
        E_N = self.lambda_b
        Var_N = self.lambda_b + self.lambda_b**2
        E_B = self.k_on * L
        mean = E_B * E_N
        var = E_B * Var_N + E_B * E_N**2
        index = var / mean
        return mean, var, index

    @staticmethod
    def fit_moments(data, L):
        """
        Estimate lambda and k_on from observed data.

        Parameters:
            data (list): Observed total counts
            L (float): Observation window length

        Returns:
            tuple: (lambda_hat, k_on_hat)
        """
        return method_of_moments_poisson(data, L)

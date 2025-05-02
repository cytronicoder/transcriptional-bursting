import numpy as np
from scipy.stats import poisson


def pmf_mixed_poisson(k, rate, k_off):
    """
    Compute marginal burst-size PMF where burst duration is exponentially distributed.

    Parameters:
        k (int): Burst size
        rate (float): Effective production rate (p * r)
        k_off (float): Off-rate for promoter switching

    Returns:
        float: Probability P(N = k)
    """
    lam = rate / k_off
    p0 = k_off / (rate + k_off)
    return (lam**k / np.math.factorial(k)) * p0 * (rate / (rate + k_off)) ** k


def method_of_moments_poisson(xs, L):
    """
    Estimate lambda and k_on using method of moments.

    Parameters:
        xs (list of int): Observed transcript totals in windows of length L
        L (float): Length of observation window

    Returns:
        tuple: (lambda_hat, k_on_hat)
    """
    x = np.array(xs)
    mean_x = np.mean(x)
    var_x = np.var(x, ddof=1)
    lambda_hat = var_x / mean_x - 1
    k_on_hat = mean_x / (lambda_hat * L)
    return lambda_hat, k_on_hat


def pmf_total_compound(x, pmf_burst, k_on, L, max_bursts=100):
    """
    Compound Poisson PMF: total observed transcripts X = sum of B burst sizes.

    Parameters:
        x (int): Total transcript count
        pmf_burst (function): Function returning P(N=k) for individual bursts
        k_on (float): Burst initiation rate
        L (float): Observation window length
        max_bursts (int): Maximum number of bursts to consider in convolution

    Returns:
        float: Probability P(X = x)
    """
    lam_b = k_on * L
    p = 0.0
    for b in range(0, max_bursts + 1):
        pb = poisson.pmf(b, lam_b)
        if b == 0:
            p += pb * (1.0 if x == 0 else 0.0)
        else:
            pmf_sum = np.zeros(x + 1)
            pmf_sum[0] = 1.0
            for _ in range(b):
                new = np.zeros(x + 1)
                for i in range(x + 1):
                    for j in range(x - i + 1):
                        new[i + j] += pmf_sum[i] * pmf_burst(j)
                pmf_sum = new
            p += pb * pmf_sum[x]
    return p

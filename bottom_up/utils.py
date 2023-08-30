#!/usr/bin/env python3
# Lanxin Zhang

from scipy import special
import numpy as np


def p_inoculum(r, v0, vl=10001):
    """
    This function computes the probability of inoculum size v0 given a success rate r.
    parameters:
    r: float
        success rate of one exposure mode
    v0: int
        inoculum size of interest
    vl: int
        upper bound for donor viral load
    return:
    The probability of inoculum size v0 under the corresponding exposure mode. For details please check
    online Method.
    """
    mu = 4.51
    sigma = 0.98
    m = 0.3892

    def cdf(x):        # the CDF of donor viral load (log normal distribution)
        return (1 + special.erf((np.log10(x ** (1 / m)) - mu) / (2 ** 0.5 * sigma))) / 2

    f0 = np.arange(1, vl)
    import warnings
    warnings.filterwarnings("ignore")
    f_array = cdf(f0) - cdf(f0-1)
    row_mat, col_mat = np.meshgrid(v0, f0)
    matrix = special.comb(col_mat, row_mat) * (r ** row_mat) * ((1 - r) ** (col_mat - row_mat))
    return f_array @ matrix


def expected_pe(x, mode_dict):
    """
    Calculate the expected extinction probability for different exposure mode, according to the probability
    distribution of inoculum size.
    parameters:
    x: float
        the extinction probability of interest
    mode_dict: dict
        a dictionary which contains the probability (value) of corresponding inoculum size (key)
    """
    res = 0
    for k, v in mode_dict.items():
        res += v * x ** k
    return res

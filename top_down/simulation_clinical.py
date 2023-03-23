#!/usr/bin/env python3
# Lanxin Zhang

import math
from mpmath import *
import numpy as np


def compute_def_integral(n_tot, py, n_inf):
    """
    Compute the distribution of infection incidence. Details in supplementary text 1.
    parameters:
        n_tot: initial population number
        n_inf: infection number
        py: total person-years
    Return: a list of probability corresponding to infection incidence ranging in [0, n_tot/py] with step width 1e-5.
    """

    mp.dps = 100

    def fun(x):  # function to integrate
        return (power(x, n_inf)) * (power(1 - py * x / n_tot, n_tot - n_inf))

    stepwidth = mpf(str(1e-5))  # use mpf for floats with higher precision
    steps = math.floor(n_tot / py / stepwidth)
    res = list()
    for i in range(steps):
        a = i * stepwidth
        b = (i + 1) * stepwidth
        fa = fun(a)
        fb = fun(b)
        res.append((fa + fb) * stepwidth / 2)
    res = np.array(res)
    res = res / res.sum()
    return res


def simulation_clinical_trial(n_individuals, py_followup, r_infection_incidence, phi=0):
    """
    Run simulation for clinical studies.
    For every individual enrolled, there's 1/avg_py probability to drop off and 1/infection_incidence
    probability to get infected.
    Parameters:
        n_individuals: total number of individuals
        py_followup: average followup years per person
        r_infection_incidence: infection rate
    Return: trajectories of time, total number of individuals and number of infections
    """

    np.random.seed()
    r_dropoff = 1 / py_followup - r_infection_incidence      # drop off rate in /person_year
    r_infection_incidence = r_infection_incidence * (1-phi)
    t_list = [0]
    n_total = [n_individuals]
    n_infection_list = [0]
    t = 0
    n_infection = 0
    while n_individuals > 0:
        B = (r_dropoff + r_infection_incidence) * n_individuals
        tau = np.random.exponential(1/B)
        t = t+tau
        r = np.random.random()
        if r_dropoff * n_individuals > r * B:      # someone drop off
            n_individuals -= 1
        else:                      # someone got infected
            n_individuals -= 1
            n_infection += 1
        n_total.append(n_individuals)
        n_infection_list.append(n_infection)
        t_list.append(t)
    return t_list, n_total, n_infection_list


def clinical_simulation_incidence_sampled(n_individuals, py_followup, r_inf, n_simul=100000, phi=0):
    """
    Run simulation for one clinical trial with infection incidence sampled from the distribution computed
    by function 'compute_def_integral'. By default the simulation will be repeated 100000 times.
    Parameters:
        n_individuals: total number of individuals
        py_followup: average followup years per person
        r_inf: an array with length n_simul containing infection incidences sampled
    """
    t_mat = list()
    n_mat = list()
    inf_mat = list()
    for i in range(n_simul):
        t, total, infection = simulation_clinical_trial(n_individuals, py_followup, r_inf[i], phi)
        t_mat.append(t)
        n_mat.append(total)
        inf_mat.append(infection)
    return t_mat, inf_mat
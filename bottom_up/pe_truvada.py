#!/usr/bin/env python3
# Lanxin Zhang

import os
import sys
import torch
import numpy as np
import pandas as pd
import psutil
from multiprocessing import Pool
import time
from copy import deepcopy

from Vectorized_clean.PEPV import *


indicators = [[0, i, j, 1] for i in range(2) for j in range(2)]


def compute_pe_truvada(indicator, adh_pattern=None, tend=50, colon_FTC=0.036584415522861,
                       time_step=0.02, ode_solver=euler):
    """
    Compute the extinction probability of Truvada for different hypotheses.
    Hypothesis is indicated by the indicator array (boolean array with 4 elems)
    colon_FTC: FTC-TP level in colon tissue. Colon FTC-TP (Cottrell et al., 2016)= 0.00547091049
    arr[0]: adherence, if 0: adherence = 1, always 0 in this case
    arr[1]: local PK
    arr[2]: local dNTP level
    arr[3]: exposure type RAI/RVI
    tend: end of running, day
    """
    e_rvi = EfficacyPredictor()
    if indicator[0]:
        # indicator: [1, 1, 1, ...], [1, 1, 0, ...], [1, 0, 1, ...], [1, 0, 0, ...]
        r_ftc = Regimen('FTC', 24, (0, 24 * tend), 200, 10, 1, adh_pattern=adh_pattern)
        r_tdf = Regimen('TDF', 24, (0, 24 * tend), 300, 10, 1, adh_pattern=adh_pattern)
    else:
        # indicator: [0, 1, 1, ...], [0, 1, 0, ...], [0, 0, 1, ...], [0, 0, 0, ...]
        r_ftc = Regimen('FTC', 24, (0, 24 * tend), 200, 40, 1, adh_pattern=adh_pattern)
        r_tdf = Regimen('TDF', 24, (0, 24 * tend), 300, 40, 1, adh_pattern=adh_pattern)
    # rvi is by default included
    e_rvi.add_regimen(r_ftc)
    e_rvi.add_regimen(r_tdf)
    e_rvi.add_sample_files('../data/ftcmax.csv')
    e_rvi.add_sample_files('../data/burnssimparam.csv')

    if time_step and ode_solver:
        e_rvi.set_pk_ode_solver('TDF', ode_solver)
        e_rvi.set_pk_time_step('TDF', time_step)
    e_rvi.compute_concentration()
    e_rai = deepcopy(e_rvi)

    if indicator[1]:
        e_rai.set_concentration_proportion('FTC', colon_FTC)
        e_rai.set_concentration_proportion('TDF', 2.92370504001375)
        e_rvi.set_concentration_proportion('FTC', 0.059229012963058)
        e_rvi.set_concentration_proportion('TDF', 0.071077979280555)
    if indicator[2]:
        e_rvi.set_pd_file('../data/mmoa/comb_output_VagChen.csv')
        e_rai.set_pd_file('../data/mmoa/comb_output_ColChen.csv')
    e_rvi.compute_extinction_probability()
    if (not indicator[1]) and (not indicator[2]):
        return e_rvi.get_extinction_probability()[::50, :, 0, 0, 0], e_rvi.get_extinction_probability()[::50, :, 0, 0, 0]
    e_rai.compute_extinction_probability()
    return e_rai.get_extinction_probability()[::50, :, 0, 0, 0], e_rvi.get_extinction_probability()[::50, :, 0, 0, 0]


def process(dose, dosing_file='../data/dosing.csv'):
    """
    compute the extinction probability for dosing schemes in dosing_file (here an example file with 90 days) and
    save the results in npy.
    """
    adh = pd.read_csv(dosing_file, index_col=0).iloc[[6-dose]].values
    for indicator in indicators:
        pe_rai, pe_rvi = compute_pe_truvada(indicator, adh_pattern=adh, tend=110)
        key = ''.join(map(str, indicator[:-1]))
        np.save("rai_{}_dose{}_a".format(key, dose), pe_rai.numpy())
        np.save("rvi_{}_dose{}_a".format(key, dose), pe_rvi.numpy())
    return 0


def main():
    pool = Pool(psutil.cpu_count(logical=False))
    t = time.time()
    for i in pool.imap_unordered(process, range(1, 7)):
        pass
    print(time.time() - t)


if __name__ == '__main__':
    main()
    print('Done')

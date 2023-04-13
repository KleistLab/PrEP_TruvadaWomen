#!/usr/bin/env python3
# Lanxin Zhang

from copy import deepcopy

from Vectorized_clean.PEPV import *


def compute_pe_truvada(indicator, tend, adh_pattern=None, time_step=0.02, ode_solver=euler):
    """
    Compute the extinction probability of Truvada for different hypotheses.
    indicator: boolean array with 4 elems, hypothesis is indicated by this indicator array
        indicator[0]: adherence, if 0: adherence = 1, always 1 in this case
        indicator[1]: local PK
        indicator[2]: local dNTP level
        indicator[3]: exposure type RAI/RVI (doesn't matter in this function)
    tend: int
        end of running, day
    return: tuple
        a tuple of two arrays: extinction probability of RAI and RVI, respectively
    """
    e_rvi = EfficacyPredictor()
    # indicator: [1, 1, 1, ...], [1, 1, 0, ...], [1, 0, 1, ...], [1, 0, 0, ...]
    r_ftc = Regimen('FTC', 24, (0, 24 * tend), 200, 40, 1, adh_pattern=adh_pattern)
    r_tdf = Regimen('TDF', 24, (0, 24 * tend), 300, 40, 1, adh_pattern=adh_pattern)
    # rvi is by default included
    e_rvi.add_regimen(r_ftc)
    e_rvi.add_regimen(r_tdf)
    e_rvi.add_sample_files('../data/pk/ftcmax.csv')
    e_rvi.add_sample_files('../data/pk/burnssimparam.csv')

    if time_step and ode_solver:
        e_rvi.set_pk_ode_solver('TDF', ode_solver)
        e_rvi.set_pk_time_step('TDF', time_step)
    e_rvi.compute_concentration()
    e_rai = deepcopy(e_rvi)

    if indicator[1]:    # set local PK ratio
        e_rai.set_concentration_proportion('FTC', 0.036584415522861)
        e_rai.set_concentration_proportion('TDF', 2.92370504001375)
        e_rvi.set_concentration_proportion('FTC', 0.059229012963058)
        e_rvi.set_concentration_proportion('TDF', 0.071077979280555)
    if indicator[2]:    # set drug effect file (MMOA) for different local dNTP level
        e_rvi.set_pd_file('../data/mmoa/comb_output_VagChen.csv')
        e_rai.set_pd_file('../data/mmoa/comb_output_ColChen.csv')
    e_rvi.compute_extinction_probability()
    if (not indicator[1]) and (not indicator[2]):
        return e_rvi.get_extinction_probability()[::50, :, 0, 0, 0], e_rvi.get_extinction_probability()[::50, :, 0, 0, 0]
    e_rai.compute_extinction_probability()
    return e_rai.get_extinction_probability()[::50, :, 0, 0, 0], e_rvi.get_extinction_probability()[::50, :, 0, 0, 0]


def process_one_dose(dose, indicator,  dosing_file='../data/dosing.csv'):
    """
    compute the extinction probability for dosing schemes in dosing_file (here an example file with 90 days) and
    save the results in npy.
    """
    adh = pd.read_csv(dosing_file, index_col=0).iloc[[7-dose]].values
    pe_rai, pe_rvi = compute_pe_truvada(indicator, adh_pattern=adh, tend=110)
    key = ''.join(map(str, indicator[:-1]))
    np.save("pe_rai_{}_d{}".format(key, dose), pe_rai.numpy())
    np.save("pe_rvi_{}_d{}".format(key, dose), pe_rvi.numpy())



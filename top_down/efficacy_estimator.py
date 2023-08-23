#!/usr/bin/env python3
# Lanxin Zhang

import sys
import math
import numpy as np
import pickle
from collections import OrderedDict
from simulation_utils import *

def run_simulations_efficacy_uniform_sampled(n_tot_nodrug, n_tot_drug, py_nodrug, py_followup, n_inf_nodrug, n_simul=1000000):
    """
    sample efficacy from uniform distribution and run the simulation on group with detected drug
    parameters:
        n_tot_nodrug: total number of individuals of drug undetected subgroup
        n_tot_drug: total number of individuals of drug detected subgroup
        py_nodrug: total time of drug undetected subgroup  in person-years
        py_followup: total time in person-years
        n_inf_nodrug: number on infections in drug undetected subgroup
        n_siml: times of simulation performed
    """
    print('Simulation for the drug-detected group initialized: efficacy sampled from uniform distribution')
    dict_infection2phi = dict()
    t_total = 0
    phi = np.random.choice(np.arange(0, 1, 0.01), n_simul)
    pdf = compute_def_integral(n_tot_nodrug, py_nodrug, n_inf_nodrug)
    r_inf = np.random.choice([i/1e5 for i in range(len(pdf))], n_simul, p=pdf.astype(np.float64))
    for i in range(n_simul):
            t, total, infection = simulation_clinical_trial(n_tot_drug, py_followup, r_inf[i], phi[i])
            if infection[-1] not in dict_infection2phi:
                dict_infection2phi[infection[-1]] = []
            dict_infection2phi[infection[-1]].append(phi[i])
            t_total += sum(t)
    # print('Average PY:', t_total/n_simul)  
    print('Simulation done')
    return dict_infection2phi


def compute_efficacy_distribution(inf_dict, inf_nodrug_file, n_inf): 
    """
    process the data and compute the distribution of efficacy for each clinical trial
    parameters:
        inf_dict: dict containing the simulation data of infection numbers assuming a uniform distributed efficacy, 
                  generated using func: run_simulations_efficacy_uniform_sampled() 
        inf_nodrug_file: name of npy file containing the infection numbers of no-drug group in intervention arm, should be generated before
                         using stochastic simulation (here the data are pre-computed and stored in ../data/inf)
        n_inf: total number of infection in intervention arm for each clinical trial
    return:
        A dictionary contains the probability distribution of efficacy:  efficacy (key): frequency (value)
        Efficacy is discretely divided into 100 intervals, spanning from 0% to 100%.
    """
    data = np.load('../data/inf/{}.npy'.format(inf_nodrug_file))
    p, infection = np.histogram(data, bins=np.unique(data), density=True)
    infection_drug = n_inf - np.array(infection)
    n_inf_larger_than0 = sum(infection_drug < 0 )
    infection_drug = infection_drug[:-n_inf_larger_than0]
    p_drug = p[:-n_inf_larger_than0]
    p_drug= np.append(p_drug, 1-sum(p_drug))

    efficacy_dict = OrderedDict({i:0 for i in np.arange(0, 1, 0.01)})
    n_efficacy_total = sum([len(inf_dict[i]) for i in infection_drug if i in inf_dict])
    

    for idx, inf in enumerate(infection_drug):
        if inf in inf_dict:
            w1 = p_drug[idx]

            phi, counts = np.unique(inf_dict[inf], return_counts=True)
            for i, p in enumerate(phi):
                efficacy_dict[p] += (w1 * counts[i] / len(inf_dict[inf])) 
        else: print(inf)
    return efficacy_dict


def main():
    # infection incidence sampled from distribution
    study = input('Please choose one clinical study '
                  '(Options: HPTN084, Partners, TDF2, VOICE, FEM): ')
    if study.lower() == 'hptn084':
        n_tot_nodrug, n_tot_drug, py_nodrug, py_followup, n_inf_nodrug, n_inf, inf_nodrug_file = \
            698, 888, 858, 1.23, 32, 36, 'HPTN084'
    elif study.lower() == 'partners':
        n_tot_nodrug, n_tot_drug, py_nodrug, py_followup, n_inf_nodrug, n_inf, inf_nodrug_file = \
            110, 456, 181, 1.649, 6.75, 9, 'Partners'
    elif study.lower() == 'tdf2':
        n_tot_nodrug, n_tot_drug, py_nodrug, py_followup, n_inf_nodrug, n_inf, inf_nodrug_file = \
            56, 224, 72, 1.282, 3.5, 7, 'TDF2'
    elif study.lower() == 'voice':
        n_tot_nodrug, n_tot_drug, py_nodrug, py_followup, n_inf_nodrug,n_inf, inf_nodrug_file = \
            699, 286, 911, 1.303, 53, 61, 'VOICE'
    elif study.lower() == 'fem':
        n_tot_nodrug, n_tot_drug, py_nodrug, py_followup, n_inf_nodrug,n_inf, inf_nodrug_file = \
            658, 366, 451, 0.685, 25, 33, 'FEM'
    else:
        sys.stderr.write('Invalid name of study. Please check and run again. \n')
    
    inf_dict = run_simulations_efficacy_uniform_sampled(n_tot_nodrug, n_tot_drug, py_nodrug, py_followup, n_inf_nodrug)
    efficacy_dict =  compute_efficacy_distribution(inf_dict, inf_nodrug_file, n_inf)
    with open('dict_{}.pkl'.format(study), 'wb') as fp:
        pickle.dump(efficacy_dict, fp)


if __name__ == '__main__':
    main()
    print('Done')




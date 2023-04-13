#!/usr/bin/env python3
# Lanxin Zhang
import sys

from simulation_clinical import *
import pandas as pd


def simulations_nodrug_and_drug_all_hypotheses(n_tot_nodrug, n_tot_drug, py_nodrug, py_followup, n_inf_nodrug, filename,
                                             df_conc, phi_dict, n_simul=100000):
    """
    Run simulation for different hypotheses (drug undetected subgroup and 8 hypotheses for drug detected group)
    and check the distribution of infection numbers. For each dosing scheme
    Parameters:
        n_tot_nodrug: total number of individuals of drug undetected subgroup
        n_tot_drug: total number of individuals of drug detected subgroup
        py_nodrug: total time of drug undetected subgroup  in person-years
        py_followup: total time in person-years
        n_inf_nodrug: number on infections in drug undetected subgroup
        filename: name of the file to save the results
        df_conc: a dataframe containing the probability of detectable plasma TFV (check supplementary text 1)
        phi_dict: dict containing the efficacy value of every hypothesis

    """
    pdf = compute_def_integral(n_tot_nodrug, py_nodrug, n_inf_nodrug)
    r_inf = np.random.choice([i / 1e5 for i in range(len(pdf))], n_simul, p=pdf.astype(np.float64))

    # drug undetected subgroup in intervention arm
    t, inf = clinical_simulation_incidence_sampled(n_tot_nodrug, py_followup, r_inf, n_simul)
    a = [inf[i][-1] for i in range(len(inf))]
    py = [np.sum(t[i]) for i in range(len(t))]
    print("average PYs: ", np.sum(py) / len(py), np.quantile(py, (0.025, 0.975)))
    print("Mean:", np.mean(a), "median", np.median(a), "95% quantile range: ", np.quantile(a, (0.025, 0.975)))
    print('--------------------------')
    res = [a]

    # drug detected subgroup in intervention arm
    for key, value in phi_dict.items():
        # calculate the weighted efficacy of 7 dosing schemes based on the percentage-concentration plots
        phi = sum([df_conc.iloc[1, i] * value[i] for i in range(7)]) / sum(df_conc.iloc[1])
        print(key, phi)
        t, inf = clinical_simulation_incidence_sampled(n_tot_drug, py_followup, r_inf, n_simul, phi=phi)
        py = [np.sum(t[i]) for i in range(len(t))]
        print("average PYs: ", np.sum(py) / len(py), np.quantile(py, (0.025, 0.975)))
        a = [inf[i][-1] for i in range(len(inf))]
        print("Mean:", np.mean(a), "95% quantile range: ", np.quantile(a, (0.025, 0.975)))
        res.append(a)
    np.save('{}'.format(filename), np.array(res))


# read the efficacy value of each hypothesis from npy file and save the mean value in a dict.
# each binary number represents a hypothesis. Check the code in ../bottom_up/main.py for details.
indicators = [[0, i, j, k] for i in range(2) for j in range(2) for k in range(2)]
phi_dict = dict()
for i, indicator in enumerate(indicators):
    key = ''.join(map(str, indicator))
    phi_dict[key] = list()
    data = np.load('../data/phi/phi_{}.npy'.format(key))
    for i in range(7):
        phi_dict[key].append(data[:, i].mean())

# read the probability of detectable plasma TFV from file
df_001 = pd.read_csv('../data/TFV_percentage/PercentageConcOver001.csv', index_col=0)  # LLOQ = 0.001uM
df_035 = pd.read_csv('../data/TFV_percentage/PercentageConcOver035.csv', index_col=0)  # LLOQ = 0.035uM


# take HPTN084 as an example to run the simulations and check the results
def main():
    # infection incidence sampled from distribution
    # HPTN84, total infection=36
    study = input('Please choose one clinical study '
                  '(Options: HPTN084, Partners, TDF2, VOICE, FEM): ')
    if study == 'HPTN084':
        n_tot_nodrug, n_tot_drug, py_nodrug, py_followup, n_inf_nodrug, filename, df_conc = \
            698, 888, 858, 1.23, 32, 'HPTN084', df_001
    if study == 'Partners':
        n_tot_nodrug, n_tot_drug, py_nodrug, py_followup, n_inf_nodrug, filename, df_conc = \
            110, 456, 181, 1.649, 6.75, 'Partners', df_001
    if study == 'TDF2':
        n_tot_nodrug, n_tot_drug, py_nodrug, py_followup, n_inf_nodrug, filename, df_conc = \
            56, 224, 72, 1.282, 3.5, 'TDF2', df_001
    if study == 'VOICE':
        n_tot_nodrug, n_tot_drug, py_nodrug, py_followup, n_inf_nodrug, filename, df_conc = \
            699, 286, 911, 1.303, 53, 'VOICE', df_001
    if study == 'FEM':
        n_tot_nodrug, n_tot_drug, py_nodrug, py_followup, n_inf_nodrug, filename, df_conc = \
            658, 366, 451, 0.685, 25, 'FEM', df_035
    else:
        sys.stderr.write('Invalid name of study. Please check and run again. \n')
    simulations_nodrug_and_drug_all_hypotheses(n_tot_nodrug, n_tot_drug, py_nodrug, py_followup, n_inf_nodrug,
                                                   filename, df_conc, phi_dict)


if __name__ == '__main__':
    main()
    print('Done')


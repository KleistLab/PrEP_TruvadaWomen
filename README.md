# Synthesis of protective oral PrEP adherence levels in cisgender women using convergent clinical- and bottom-up modeling.
> This repo contains the codes that were used to analyze the prophylactic efficacy of Truvada on cis-gender women in paper DOI: 10.21203/rs.3.rs-2772765/v1. 
> The top-down and bottom-up analysis can be reproduced using corresponding codes provided here.  

[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/) [![DOI](https://zenodo.org/badge/616463573.svg)](https://zenodo.org/badge/latestdoi/616463573)

## Table of Contents
-   [System requirements](#system-requirements)
      -   [Operating systems](#operating-systems)
      -   [Prerequisites](#prerequisites)
      -   [Dependencies](#dependencies)
- [Top-down](#Top-down)
- [Bottom-up](#Bottom-up)
- [About data](#About-data)

## System requirements

### Operating systems
This workflow was tested on Ubuntu 20.04.5 LTS.

### Prerequisites
Some tools have to be installed to run the analysis. We recommend following the steps below.

#### Install Conda/Miniconda

Conda will manage the dependencies of our program. Instructions can be found here: https://docs.conda.io/projects/conda/en/latest/user-guide/install.


#### Create the working environment

Create a new environment from the given environment config in [`env.yml`](./env/env.yml), where the pipeline will be executed.
Go to the main folder of the repository and call:

```
conda env create -f env/env.yml
```

This step may take a few minutes.

To activate the environment type:

```
conda activate prep
```

### Dependencies

This workflow uses the following dependencies:

```
  - numpy
  - scipy
  - pandas
  - torch
```
They can be installed automatically by creating the conda environment above. 

## Top-down
In the folder 'top_down' there are several scripts for different purposes:
* simulation_clinical.py: contains several functions for the simulation of clinical trials: 
compute the PDF (probability density function) of infection incidence, sample infection 
incidence from the PDF and then run the clinical simulations (see Supplementary Text 1). 
* run_simulation.py: call function in simulation_clinical.py, run the simulations and return 
the number of infections for each hypothesis. 

To run the top-down analysis:
```
cd top_down/
./run_simulation.py
```
The desired clinical study has to be chosen: 
```commandline
Please choose one clinical study (Options: HPTN084, Partners, TDF2, VOICE, FEM): 
```
After choosing the clinical study, the simulations will run for the corresponding study.

## Bottom-up
The folder 'bottom_up' contains scripts and packages for the 'bottom-up' modelling 
in the paper. The main task of this part is to compute the PrEP efficacy of Truvada under 
the hypothesized mechanism. 
* Package 'Vectorized_clean': compute the prophylactic efficacy 
in a vectorized way, i.e. this package can compute the PrEP efficacy trajectory of multiple 
regimens for multiple individuals in a single run. For detailed usage of this package you can 
check the examples in 'bottom_up/Vectorized_clean/example.ipynb'. 
* pe_truvada.py: compute the extinction probability for Truvada on women, for different hypothesis
and their combinations (see section Results). Here we use an example file (data/dosing.csv) containing 
7 dosing regimens for 90 days long, i.e. once per week to 7 times per week. The pharmacokinetic parameters 
of 1000 virtual individuals are used (parameters in data/pk). This computation can be slow since it computes the extinction probability profile for 1000 individuals. 
* utils.py: contains helper functions for the computation of PrEP efficacy, e.g. function to calculate 
the probability of different inoculum size.
* compute_efficacy: compute the PrEP efficacy for a given dose and regimen and save the efficacy profile 
as a numpy array. 
To run the bottom-up  analysis:
```
cd bottom_up/
./compute_efficacy.py
```
The dose of interest and length of regimen will be asked:
```
Please enter the number of doses per week: 
Please enter the duration of regimen in days: 
```
After entering the numbers the computation will run and the efficacy profile of each combination of hypotheses 
will be stored automatically in npy files.  

## About data
The folder 'data' contains pre-generated data that are necessary for the top-down and bottom-up analysis. 
* dosing.csv: contains 7 boolean arrays which represent respectively the dosing regimens that 1, 2, ... 7 doses will be taken every week. It's used to compute the PrEP efficacy in the bottom-up approach.
* pk: there are two files containing the pharmacokinetic parameters of 1000 virtual patients for TDF (burnssimparam.csv) and FTC (ftcmax.csv), respectively. They are utilized in the bottom-up approach.
* mmoa: two files containing the matrices computed from molecular mode of action: the matrices represent the relation between TFV-DP and FTC-TP level and direct drug effect. They are used in the bottom-up approach.
* phi: contains several numpy arrays which are precomputed PrEP efficacy trajectories under different hypothesis. 
* tfv_percentage: two files containing the probability that TFV is detectable (above the lower limit of quantification, LLOQ) within different dosing adherence for LLOQ = 0.001uM and 0.035uM respectively. 


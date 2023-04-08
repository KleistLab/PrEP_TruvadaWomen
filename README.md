# Truvada as PrEP for women
>This repo contains the codes that were used to generate the 
> results of the top-down and bottom-up approaches.  

[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)

## Table of Contents
-   [System requirements](#system-requirements)
      -   [Operating systems](#operating-systems)
      -   [Prerequisites](#prerequisites)
      -   [Dependencies](#dependencies)
- [Top-down](#Top-down)
- [Bottom-up](#Bottom-up)

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

## Top-down
In the folder 'top_down' there are several scripts for different purposes:
* simulation_clinical.py: contains functions for the simulation of clinical trials. 
* run_simulation.py: call function in simulation_clinical.py, run the simulations and return 
the number of infections for each hypothesis. 

## Bottom-up
The folder 'bottom_up' contains scripts and packages for the 'bottom-up' modelling 
in the paper. 
* There is a package  called 'Vectorized_clean' to compute the prophylactic efficacy 
in a vectorized way, i.e. this package can compute the PrEP efficacy trajectory of multiple 
regimens for multiple individuals in a single run. For detailed usage of this package you can 
check the examples in 'bottom_up/Vectorized_clean/example.ipynb'. 
* PrEP efficacy for different hypotheses ... ...  


